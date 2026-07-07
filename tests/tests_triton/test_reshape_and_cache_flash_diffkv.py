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
KERNEL_NAME = "reshape_and_cache_kernel_flash_diffkv"


@triton.jit
def reshape_and_cache_kernel_flash_diffkv(
    key_ptr,
    value_ptr,
    kv_cache_ptr,
    slot_mapping_ptr,
    k_scale,
    v_scale,
    num_heads: tl.constexpr,
    head_size_k: tl.constexpr,
    head_size_v: tl.constexpr,
    block_size: tl.constexpr,
    FP8_KV_CACHE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(axis=0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        return
    tile_i = tl.program_id(axis=1)
    tile_offs = tl.arange(0, TILE_SIZE)
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    # Compute strides as constexpr to avoid QAIC int64 parameter issues
    _key_stride = num_heads * head_size_k
    _value_stride = num_heads * head_size_v
    _block_stride = block_size * num_heads * (head_size_k + head_size_v)
    _page_stride = num_heads * (head_size_k + head_size_v)
    src_key_idx = token_idx * _key_stride + tile_i * head_size_k
    src_value_idx = token_idx * _value_stride + tile_i * head_size_v
    tgt_idx = (block_idx * _block_stride + block_offset * _page_stride
               + tile_i * (head_size_k + head_size_v))
    key_load = tl.load(key_ptr + src_key_idx + tile_offs,
                       mask=tile_offs < head_size_k)
    if FP8_KV_CACHE:
        key_tile = key_load if key_load.dtype.is_fp8() else key_load / tl.load(k_scale)
    else:
        key_tile = key_load
    value_load = tl.load(value_ptr + src_value_idx + tile_offs,
                         mask=tile_offs < head_size_v)
    if FP8_KV_CACHE:
        if value_load.dtype.is_fp8():
            value_tile = value_load
        else:
            value_tile = value_load / tl.load(v_scale)
    else:
        value_tile = value_load
    tl.store(kv_cache_ptr + tgt_idx + tile_offs, key_tile,
             mask=tile_offs < head_size_k)
    tl.store(kv_cache_ptr + tgt_idx + head_size_k + tile_offs, value_tile,
             mask=tile_offs < head_size_v)
    return


def main(log_path):
    torch.manual_seed(42)
    num_tokens = 4
    num_heads = 2
    head_size_k = 8
    head_size_v = 8
    block_size = 4
    num_blocks = 8
    # TILE_SIZE must be >= max(head_size_k, head_size_v), power of 2
    TILE_SIZE = 8

    key = torch.randn(num_tokens, num_heads, head_size_k, dtype=torch.float32, device=DEVICE)
    value = torch.randn(num_tokens, num_heads, head_size_v, dtype=torch.float32, device=DEVICE)
    # kv_cache: [num_blocks, block_size, num_heads, head_size_k + head_size_v]
    kv_cache = torch.zeros(num_blocks, block_size, num_heads, head_size_k + head_size_v,
                           dtype=torch.float32, device=DEVICE)
    slot_mapping = torch.tensor([0, 1, 4, 5], dtype=torch.int32, device=DEVICE)

    k_stride = key.stride(0)
    v_stride = value.stride(0)
    block_stride = kv_cache.stride(0)
    page_stride = kv_cache.stride(1)

    grid = (num_tokens, num_heads)

    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  key: {tuple(key.shape)}, value: {tuple(value.shape)}, device={key.device}")
            log(f"  kv_cache: {tuple(kv_cache.shape)}")
            log(f"  slot_mapping: {slot_mapping.cpu().tolist()}")
            log(f"  num_heads={num_heads}, head_size_k={head_size_k}, head_size_v={head_size_v}")
            log(f"  block_size={block_size}, TILE_SIZE={TILE_SIZE}")

            reshape_and_cache_kernel_flash_diffkv[grid](
                key_ptr=key,
                value_ptr=value,
                kv_cache_ptr=kv_cache,
                slot_mapping_ptr=slot_mapping,
                k_scale=torch.tensor(1.0, device=DEVICE),
                v_scale=torch.tensor(1.0, device=DEVICE),
                num_heads=num_heads,
                head_size_k=head_size_k,
                head_size_v=head_size_v,
                block_size=block_size,
                FP8_KV_CACHE=False,
                TILE_SIZE=TILE_SIZE,
                num_warps=1,
                num_stages=1,
            )

            kvc = kv_cache.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: {grid}  (num_tokens, num_heads)")
            log("\nOutput:")
            log(f"  kv_cache: shape={tuple(kvc.shape)}, mean={kvc.mean().item():.4f}")

            # reference: write key and value per-head into kv_cache
            ref = torch.zeros_like(kv_cache.cpu())
            for t in range(num_tokens):
                slot = slot_mapping[t].item()
                bi = slot // block_size
                bo = slot % block_size
                for h in range(num_heads):
                    ref[bi, bo, h, :head_size_k] = key[t, h].cpu()
                    ref[bi, bo, h, head_size_k:] = value[t, h].cpu()
            max_diff = (kvc - ref).abs().max().item()
            log("\nValidation (vs manual reference):")
            log(f"  max abs diff: {max_diff:.6f}")
            assert max_diff < 1e-5, f"diff too large: {max_diff}"
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
    run_test(main, LOG_DIR, "reshape_and_cache_flash_diffkv")
