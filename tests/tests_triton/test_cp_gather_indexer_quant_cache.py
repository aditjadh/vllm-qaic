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
KERNEL_NAME = "_cp_gather_indexer_quant_cache_kernel"


@triton.jit
def _cp_gather_indexer_quant_cache_kernel(
    kv_cache_ptr,
    kv_cache_scale_ptr,
    k_fp8_ptr,
    k_scale_ptr,
    block_table_ptr,
    cu_seqlen_ptr,
    token_to_seq_ptr,
    block_size,
    block_table_stride,
    kv_cache_stride,
    kv_cache_scale_stride,
    LAYOUT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TILE_SIZE: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr,
):
    tid = tl.program_id(0)
    offset = tl.arange(0, HEAD_DIM)
    batch_id = tl.load(token_to_seq_ptr + tid)
    batch_start = tl.load(cu_seqlen_ptr + batch_id)
    batch_end = tl.load(cu_seqlen_ptr + batch_id + 1)
    batch_offset = tid - batch_start
    if tid >= batch_end:
        return
    block_table_id = batch_offset // block_size
    block_offset = batch_offset % block_size
    block_table_offset = batch_id * block_table_stride + block_table_id
    block_id = tl.load(block_table_ptr + block_table_offset)
    tiled_block_id = block_offset // BLOCK_TILE_SIZE
    tiled_block_offset = block_offset % BLOCK_TILE_SIZE
    if LAYOUT == "SHUFFLE":
        src_cache_offset = (block_id * kv_cache_stride
                            + tiled_block_id * HEAD_DIM * BLOCK_TILE_SIZE
                            + tiled_block_offset * HEAD_TILE_SIZE)
    else:
        src_cache_offset = block_id * kv_cache_stride + block_offset * HEAD_DIM
    src_scale_offset = block_id * kv_cache_scale_stride + block_offset
    dst_offset = tid * HEAD_DIM
    src_scale_ptr = kv_cache_scale_ptr + src_scale_offset
    src_cache_ptr = kv_cache_ptr + src_cache_offset
    dst_k_ptr = k_fp8_ptr + dst_offset
    scale_val = tl.load(src_scale_ptr)
    tl.store(k_scale_ptr + tid, scale_val)
    if LAYOUT == "SHUFFLE":
        tiled_src_offset = (offset // HEAD_TILE_SIZE * HEAD_TILE_SIZE * BLOCK_TILE_SIZE
                            + offset % HEAD_TILE_SIZE)
    else:
        tiled_src_offset = offset
    val = tl.load(src_cache_ptr + tiled_src_offset)
    tl.store(dst_k_ptr + offset, val)


def main(log_path):
    torch.manual_seed(42)
    batch_size = 2
    seq_lens = [3, 4]
    total_tokens = sum(seq_lens)
    HEAD_DIM = 8
    block_size = 4
    num_blocks = 8
    BLOCK_TILE_SIZE = 4
    HEAD_TILE_SIZE = 8

    # block_table: [batch_size, max_blocks_per_seq]
    max_blocks = 2
    block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=DEVICE)

    # cu_seqlen: cumulative sequence lengths
    cu_seqlen = torch.tensor([0, seq_lens[0], seq_lens[0] + seq_lens[1]],
                              dtype=torch.int32, device=DEVICE)

    # token_to_seq: which sequence each output token belongs to
    token_to_seq = torch.tensor([0] * seq_lens[0] + [1] * seq_lens[1],
                                  dtype=torch.int32, device=DEVICE)

    k_scale_out = torch.zeros(total_tokens, dtype=torch.float32, device=DEVICE)

    grid = (total_tokens,)

    with open(log_path, "w") as fp:
        log = logger(fp)
        log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        log(f"Kernel: {KERNEL_NAME}")
        log("\nInputs:")
        log(f"  batch_size={batch_size}, seq_lens={seq_lens}, total_tokens={total_tokens}")
        log(f"  HEAD_DIM={HEAD_DIM}, block_size={block_size}")
        log(f"  block_table: {block_table.cpu().tolist()}")
        log(f"  cu_seqlen: {cu_seqlen.cpu().tolist()}")
        try:
            fp8_dtype = torch.float8_e4m3fn

            # Build kv_cache (NHD layout): [num_blocks, block_size, HEAD_DIM] fp8 + scale
            kv_cache_value = torch.randint(-127, 127,
                                           (num_blocks, block_size, HEAD_DIM),
                                           dtype=torch.int8, device=DEVICE).view(fp8_dtype)
            kv_cache_scale = (torch.rand(num_blocks, block_size, dtype=torch.float32, device=DEVICE)
                              * 0.1 + 0.01)

            # Output buffer
            k_fp8_out = torch.zeros(total_tokens, HEAD_DIM, dtype=fp8_dtype, device=DEVICE)

            log(f"  kv_cache_value: {tuple(kv_cache_value.shape)} [blocks,block_size,head_dim] fp8, device={kv_cache_value.device}")
            log(f"  kv_cache_scale: {tuple(kv_cache_scale.shape)}")

            _cp_gather_indexer_quant_cache_kernel[grid](
                kv_cache_value,
                kv_cache_scale,
                k_fp8_out,
                k_scale_out,
                block_table,
                cu_seqlen,
                token_to_seq,
                block_size,
                block_table.stride(0),
                kv_cache_value.stride(0),
                kv_cache_scale.stride(0),
                LAYOUT="NHD",
                HEAD_DIM=HEAD_DIM,
                BLOCK_TILE_SIZE=BLOCK_TILE_SIZE,
                HEAD_TILE_SIZE=HEAD_TILE_SIZE,
                num_warps=1,
                num_stages=1,
            )

            ko = k_fp8_out.cpu()
            so = k_scale_out.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: {grid}  (one program per output token)")
            log(f"  HEAD_DIM={HEAD_DIM} (constexpr), BLOCK_TILE_SIZE={BLOCK_TILE_SIZE} (constexpr)")
            log("\nOutput:")
            log(f"  k_fp8_out: shape={tuple(ko.shape)}, dtype={ko.dtype}")
            log(f"  k_scale_out: shape={tuple(so.shape)}, mean={so.mean().item():.6f}")

            # reference: gather k values and scales using block_table
            kc_cpu = kv_cache_value.cpu()
            sc_cpu = kv_cache_scale.cpu()
            bt_cpu = block_table.cpu()
            ref_fp8 = torch.zeros_like(ko)
            ref_sc = torch.zeros_like(so)
            for tok in range(total_tokens):
                b = token_to_seq[tok].item()
                bs = cu_seqlen[b].item()
                batch_off = tok - bs
                blk = batch_off // block_size
                slot = batch_off % block_size
                phys_blk = bt_cpu[b, blk].item()
                ref_fp8[tok] = kc_cpu[phys_blk, slot]
                ref_sc[tok] = sc_cpu[phys_blk, slot]

            diff_sc = (so - ref_sc).abs().max().item()
            diff_fp8 = (ko.view(torch.uint8).float() - ref_fp8.view(torch.uint8).float()).abs().max().item()
            log("\nValidation (vs manual gather reference):")
            log(f"  k_fp8 max abs diff (raw bytes): {diff_fp8:.1f}")
            log(f"  k_scale max abs diff: {diff_sc:.8f}")
            assert diff_sc < 1e-6, f"scale diff={diff_sc}"
            assert diff_fp8 == 0, f"fp8 diff={diff_fp8}"
            log("  assert: PASSED")
            log("\nSummary:")
            log(f"  Kernel execution: SUCCESS (grid {grid})")
            log("  Validation vs reference: PASSED")
        except RuntimeError as e:
            if "Unsupported datatype" in str(e) or "Float8" in str(e):
                log(f"\nStatus: SKIPPED")
                log(f"\nReason: QAIC hardware limitation — {e}")
                log("\nSummary:\n  Kernel execution: SKIPPED (hardware does not support Float8_e4m3fn)")
            else:
                log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
                log("\nSummary:\n  Kernel execution: FAILED")
                fp.write("\n" + "-" * 36 + "\n")
                sys.exit(1)
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Kernel execution: FAILED")
            fp.write("\n" + "-" * 36 + "\n")
            sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "cp_gather_indexer_quant_cache")
