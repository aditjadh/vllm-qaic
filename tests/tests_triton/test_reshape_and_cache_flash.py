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
KERNEL_NAME = "reshape_and_cache_kernel_flash"


@triton.jit
def reshape_and_cache_kernel_flash(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    k_scale,
    v_scale,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    x: tl.constexpr,
    USE_HEAD_MAJOR_LAYOUT: tl.constexpr,
    FP8_KV_CACHE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(axis=0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        return
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    tile_i = tl.program_id(axis=1)
    tile_offs = tl.arange(0, TILE_SIZE)
    tile_pos = tile_i * TILE_SIZE + tile_offs
    # Compute strides as constexpr to avoid QAIC int64 parameter issues
    _key_stride = num_heads * head_size
    _block_stride = block_size * num_heads * head_size
    _page_stride = num_heads * head_size
    src_key_idx = token_idx * _key_stride
    src_value_idx = token_idx * _key_stride
    if USE_HEAD_MAJOR_LAYOUT:
        cur_head = tile_pos // head_size
        cur_dim = tile_pos % head_size
        tgt_idx_v = (block_idx * _block_stride + cur_head * _page_stride
                     + cur_dim + block_offset * 1)
        tgt_idx_k = (block_idx * _block_stride + cur_head * _page_stride
                     + (cur_dim // x) * block_size * x + block_offset * x
                     + (cur_dim % x))
    else:
        tgt_base = block_idx * _block_stride + block_offset * _page_stride
        tgt_idx_k = tgt_base + tile_pos
        tgt_idx_v = tgt_base + tile_pos
    key_load = tl.load(key_ptr + src_key_idx + tile_pos,
                       mask=tile_pos < (num_heads * head_size))
    if FP8_KV_CACHE:
        key_tile = key_load if key_load.dtype.is_fp8() else key_load / tl.load(k_scale)
    else:
        key_tile = key_load
    value_load = tl.load(value_ptr + src_value_idx + tile_pos,
                         mask=tile_pos < (num_heads * head_size))
    if FP8_KV_CACHE:
        if value_load.dtype.is_fp8():
            value_tile = value_load
        else:
            value_tile = value_load / tl.load(v_scale)
    else:
        value_tile = value_load
    tl.store(key_cache_ptr + tgt_idx_k, key_tile,
             mask=tile_pos < (num_heads * head_size))
    tl.store(value_cache_ptr + tgt_idx_v, value_tile,
             mask=tile_pos < (num_heads * head_size))
    return


def main(log_path):
    torch.manual_seed(42)
    num_tokens = 4
    num_heads = 2
    head_size = 8
    block_size = 4
    num_blocks = 8
    TILE_SIZE = 16  # must be power-of-2, >= num_heads*head_size

    key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float32, device=DEVICE)
    value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float32, device=DEVICE)
    # non-head-major layout: [num_blocks, block_size, num_heads, head_size]
    key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size,
                            dtype=torch.float32, device=DEVICE)
    value_cache = torch.zeros_like(key_cache)
    # slot_mapping: one slot per token
    slot_mapping = torch.tensor([0, 1, 4, 5], dtype=torch.int32, device=DEVICE)

    key_stride = key.stride(0)
    value_stride = value.stride(0)
    block_stride = key_cache.stride(0)
    page_stride = key_cache.stride(1)
    n = num_heads * head_size

    grid = (num_tokens, triton.cdiv(n, TILE_SIZE))

    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  key: {tuple(key.shape)}, value: {tuple(value.shape)}, device={key.device}")
            log(f"  key_cache: {tuple(key_cache.shape)}, value_cache: {tuple(value_cache.shape)}")
            log(f"  slot_mapping: {slot_mapping.cpu().tolist()}")
            log(f"  num_tokens={num_tokens}, num_heads={num_heads}, head_size={head_size}")
            log(f"  block_size={block_size}, TILE_SIZE={TILE_SIZE}")

            reshape_and_cache_kernel_flash[grid](
                key_ptr=key,
                value_ptr=value,
                key_cache_ptr=key_cache,
                value_cache_ptr=value_cache,
                slot_mapping_ptr=slot_mapping,
                k_scale=torch.tensor(1.0, device=DEVICE),
                v_scale=torch.tensor(1.0, device=DEVICE),
                num_heads=num_heads,
                head_size=head_size,
                block_size=block_size,
                x=1,
                USE_HEAD_MAJOR_LAYOUT=False,
                FP8_KV_CACHE=False,
                TILE_SIZE=TILE_SIZE,
                num_warps=1,
                num_stages=1,
            )

            kc = key_cache.cpu()
            vc = value_cache.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: {grid}  (num_tokens, cdiv(n={n}, TILE_SIZE={TILE_SIZE}))")
            log("\nOutput:")
            log(f"  key_cache: shape={tuple(kc.shape)}, mean={kc.mean().item():.4f}")
            log(f"  value_cache: shape={tuple(vc.shape)}, mean={vc.mean().item():.4f}")

            # reference: manually write key/value into cache at slot positions
            ref_kc = torch.zeros_like(key_cache.cpu())
            ref_vc = torch.zeros_like(value_cache.cpu())
            for t in range(num_tokens):
                slot = slot_mapping[t].item()
                bi = slot // block_size
                bo = slot % block_size
                ref_kc[bi, bo] = key[t].cpu()
                ref_vc[bi, bo] = value[t].cpu()
            max_k = (kc - ref_kc).abs().max().item()
            max_v = (vc - ref_vc).abs().max().item()
            log("\nValidation (vs manual reference):")
            log(f"  key_cache max abs diff: {max_k:.6f}")
            log(f"  value_cache max abs diff: {max_v:.6f}")
            assert max_k < 1e-5 and max_v < 1e-5, f"diff too large: k={max_k}, v={max_v}"
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
    run_test(main, LOG_DIR, "reshape_and_cache_flash")
