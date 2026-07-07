# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
import os
import sys
import traceback
from datetime import datetime

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vllm"))
from _ktest import run_test, logger

from vllm.triton_utils import tl, triton

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "cp_mha_gather_cache_kernel"


@triton.jit
def cp_mha_gather_cache_kernel(
    key_cache_ptr,
    value_cache_ptr,
    key_ptr,
    value_ptr,
    block_table_ptr,
    cu_seqlens_kv_ptr,
    token_to_batch_ptr,
    seq_start_ptr,
    k_scale_ptr,
    v_scale_ptr,
    num_heads,
    head_size,
    x,
    max_block_num,
    DEQUANT: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    CACHE_FORMAT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    key_ptr_offset = (key_ptr + token_id * head_size * num_heads
                      + head_id * head_size)
    value_ptr_offset = (value_ptr + token_id * head_size * num_heads
                        + head_id * head_size)
    batch_idx = tl.load(token_to_batch_ptr + token_id)
    batch_start = tl.load(seq_start_ptr + batch_idx)
    token_start = tl.load(cu_seqlens_kv_ptr + batch_idx)
    batch_offset = token_id - token_start + batch_start
    block_offset = batch_offset // PAGE_SIZE
    block_id = tl.load(
        block_table_ptr + max_block_num * batch_idx + block_offset
    ).to(tl.int64)
    slot_id = batch_offset % PAGE_SIZE

    if CACHE_FORMAT == "NHD":
        key_cache_ptr_offset = (
            key_cache_ptr
            + block_id * num_heads * head_size * PAGE_SIZE
            + slot_id * num_heads * head_size
            + head_id * head_size
        )
        value_cache_ptr_offset = (
            value_cache_ptr
            + block_id * num_heads * head_size * PAGE_SIZE
            + slot_id * num_heads * head_size
            + head_id * head_size
        )
        k_reg = tl.load(key_cache_ptr_offset + col_offsets)
        v_reg = tl.load(value_cache_ptr_offset + col_offsets)
        if DEQUANT:
            k_scale = tl.load(k_scale_ptr)
            v_scale = tl.load(v_scale_ptr)
            k_dtype = k_reg.dtype
            v_dtype = v_reg.dtype
            k_reg = (k_reg.to(tl.float32) * k_scale).to(k_dtype)
            v_reg = (v_reg.to(tl.float32) * v_scale).to(v_dtype)
        tl.store(key_ptr_offset + col_offsets, k_reg)
        tl.store(value_ptr_offset + col_offsets, v_reg)


def main(log_path):
    torch.manual_seed(42)
    batch_size = 2
    page_size = 4
    num_heads = 2
    head_size = 8
    num_blocks = 8
    # Each batch entry has 2 pages
    max_block_num = 2
    total_tokens = batch_size * page_size  # 8

    # key_cache, value_cache: [num_blocks, page_size, num_heads, head_size] (NHD)
    key_cache = torch.randn(num_blocks, page_size, num_heads, head_size,
                            dtype=torch.float32, device=DEVICE)
    value_cache = torch.randn_like(key_cache)

    # Output tensors
    key_out = torch.zeros(total_tokens, num_heads, head_size,
                          dtype=torch.float32, device=DEVICE)
    value_out = torch.zeros_like(key_out)

    # block_table: [batch_size, max_block_num] — physical block indices
    block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=DEVICE)

    # cu_seqlens_kv: cumulative token start per batch [0, page_size, 2*page_size]
    cu_seqlens_kv = torch.tensor([0, page_size, 2 * page_size],
                                  dtype=torch.int32, device=DEVICE)

    # token_to_batch: which batch entry each output token belongs to
    token_to_batch = torch.tensor([0] * page_size + [1] * page_size,
                                   dtype=torch.int32, device=DEVICE)

    # seq_start: where each batch starts in the logical sequence (same as cu_seqlens here)
    seq_start = torch.tensor([0, page_size], dtype=torch.int32, device=DEVICE)

    k_scale = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
    v_scale = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)

    grid = (total_tokens, num_heads)

    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  key_cache: {tuple(key_cache.shape)} [blocks,page_size,heads,head_size], device={key_cache.device}")
            log(f"  value_cache: {tuple(value_cache.shape)}")
            log(f"  batch_size={batch_size}, total_tokens={total_tokens}")
            log(f"  page_size={page_size}, num_heads={num_heads}, head_size={head_size}")
            log(f"  CACHE_FORMAT='NHD', DEQUANT=False")

            cp_mha_gather_cache_kernel[grid](
                key_cache,
                value_cache,
                key_out,
                value_out,
                block_table,
                cu_seqlens_kv,
                token_to_batch,
                seq_start,
                k_scale,
                v_scale,
                num_heads,
                head_size,
                1,  # x (unused for NHD)
                max_block_num,
                DEQUANT=False,
                PAGE_SIZE=page_size,
                CACHE_FORMAT="NHD",
                BLOCK_SIZE=head_size,
                num_warps=1,
                num_stages=1,
            )

            ko = key_out.cpu()
            vo = value_out.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: {grid}  (total_tokens={total_tokens}, num_heads={num_heads})")
            log(f"  PAGE_SIZE={page_size} (constexpr), BLOCK_SIZE={head_size} (constexpr)")
            log("\nOutput:")
            log(f"  key_out: shape={tuple(ko.shape)}, mean={ko.mean().item():.4f}")
            log(f"  value_out: shape={tuple(vo.shape)}, mean={vo.mean().item():.4f}")

            # reference: gather from cache using block_table
            kc_cpu = key_cache.cpu()
            vc_cpu = value_cache.cpu()
            ref_ko = torch.zeros_like(ko)
            ref_vo = torch.zeros_like(vo)
            for tok in range(total_tokens):
                b = token_to_batch[tok].item()
                bs = seq_start[b].item()
                ts = cu_seqlens_kv[b].item()
                batch_off = tok - ts + bs
                blk_off = batch_off // page_size
                slot = batch_off % page_size
                blk_id = block_table[b, blk_off].item()
                ref_ko[tok] = kc_cpu[blk_id, slot]
                ref_vo[tok] = vc_cpu[blk_id, slot]
            max_k = (ko - ref_ko).abs().max().item()
            max_v = (vo - ref_vo).abs().max().item()
            log("\nValidation (vs manual gather reference):")
            log(f"  key_out max abs diff: {max_k:.6f}")
            log(f"  value_out max abs diff: {max_v:.6f}")
            assert max_k < 1e-5 and max_v < 1e-5, f"k={max_k}, v={max_v}"
            log("  assert: PASSED")
            log("\nSummary:")
            log(f"  Kernel execution: SUCCESS (grid {grid})")
            log("  Validation vs reference: PASSED")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Kernel execution: FAILED")
            fp.write("\n" + "-" * 36 + "\n")
            sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "cp_mha_gather_cache")
