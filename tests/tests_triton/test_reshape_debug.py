# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Debug script: test if using constexpr strides fixes reshape_and_cache_flash on QAIC."""
import os
import sys
import traceback
from datetime import datetime

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vllm"))
from vllm.triton_utils import tl, triton

DEVICE = "qaic"

@triton.jit
def reshape_cache_constexpr_strides(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(axis=0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        return
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    tile_offs = tl.arange(0, TILE_SIZE)
    # Use constexpr arithmetic for strides (avoids runtime int64 param issues on QAIC)
    key_stride = num_heads * head_size
    block_stride = block_size * num_heads * head_size
    page_stride = num_heads * head_size

    src_key_idx = token_idx * key_stride
    tgt_base = block_idx * block_stride + block_offset * page_stride

    key_load = tl.load(key_ptr + src_key_idx + tile_offs,
                       mask=tile_offs < key_stride)
    tl.store(key_cache_ptr + tgt_base + tile_offs, key_load,
             mask=tile_offs < key_stride)

    src_val_idx = token_idx * key_stride
    value_load = tl.load(value_ptr + src_val_idx + tile_offs,
                         mask=tile_offs < key_stride)
    tl.store(value_cache_ptr + tgt_base + tile_offs, value_load,
             mask=tile_offs < key_stride)


def main():
    torch.manual_seed(42)
    num_tokens = 4
    num_heads = 2
    head_size = 8
    block_size = 4
    num_blocks = 8
    TILE_SIZE = 16  # must be >= num_heads*head_size and power-of-2

    key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float32, device=DEVICE)
    value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float32, device=DEVICE)
    key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.float32, device=DEVICE)
    value_cache = torch.zeros_like(key_cache)
    slot_mapping = torch.tensor([0, 1, 4, 5], dtype=torch.int32, device=DEVICE)

    grid = (num_tokens,)

    reshape_cache_constexpr_strides[grid](
        key, value, key_cache, value_cache, slot_mapping,
        num_heads=num_heads, head_size=head_size, block_size=block_size,
        TILE_SIZE=TILE_SIZE, num_warps=1, num_stages=1,
    )

    kc = key_cache.cpu()
    vc = value_cache.cpu()
    k_cpu = key.cpu()
    v_cpu = value.cpu()

    ref_kc = torch.zeros_like(kc)
    ref_vc = torch.zeros_like(vc)
    slots = slot_mapping.cpu().tolist()
    for t in range(num_tokens):
        slot = slots[t]
        bi = slot // block_size
        bo = slot % block_size
        ref_kc[bi, bo] = k_cpu[t]
        ref_vc[bi, bo] = v_cpu[t]

    max_k = (kc - ref_kc).abs().max().item()
    max_v = (vc - ref_vc).abs().max().item()
    print(f"max_k diff: {max_k}")
    print(f"max_v diff: {max_v}")
    print(f"Validation: {'PASSED' if max_k < 1e-5 and max_v < 1e-5 else 'FAILED'}")

    # Also print token 0 values to compare
    print(f"\nkey[0] flat: {k_cpu[0].flatten().tolist()}")
    print(f"kc[0,0] flat: {kc[0,0].flatten().tolist()}")
    print(f"ref_kc[0,0] flat: {ref_kc[0,0].flatten().tolist()}")


if __name__ == "__main__":
    main()
