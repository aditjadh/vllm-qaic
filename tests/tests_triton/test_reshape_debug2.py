# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Debug: print exact values written by reshape_and_cache_kernel_flash."""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vllm"))
from vllm.triton_utils import tl, triton

DEVICE = "qaic"
torch.manual_seed(42)

num_tokens = 4
num_heads = 2
head_size = 8
block_size = 4
num_blocks = 8
TILE_SIZE = 16

key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float32, device=DEVICE)
value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float32, device=DEVICE)
key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.float32, device=DEVICE)
value_cache = torch.zeros_like(key_cache)
slot_mapping = torch.tensor([0, 1, 4, 5], dtype=torch.int32, device=DEVICE)

@triton.jit
def reshape_and_cache_kernel_flash(
    key_ptr, value_ptr, key_cache_ptr, value_cache_ptr, slot_mapping_ptr,
    k_scale, v_scale,
    key_stride: tl.int64, value_stride: tl.int64,
    block_stride: tl.int64, head_stride: tl.int64,
    dim_stride_k: tl.int64, dim_stride_v: tl.int64, page_stride: tl.int64,
    num_heads: tl.constexpr, head_size: tl.constexpr, block_size: tl.constexpr,
    x: tl.constexpr, USE_HEAD_MAJOR_LAYOUT: tl.constexpr,
    FP8_KV_CACHE: tl.constexpr, TILE_SIZE: tl.constexpr,
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
    _key_stride = num_heads * head_size
    _block_stride = block_size * num_heads * head_size
    _page_stride = num_heads * head_size
    src_key_idx = token_idx * _key_stride
    src_value_idx = token_idx * _key_stride
    tgt_base = block_idx * _block_stride + block_offset * _page_stride
    tgt_idx_k = tgt_base + tile_pos
    tgt_idx_v = tgt_base + tile_pos
    key_load = tl.load(key_ptr + src_key_idx + tile_pos,
                       mask=tile_pos < (num_heads * head_size))
    tl.store(key_cache_ptr + tgt_idx_k, key_load,
             mask=tile_pos < (num_heads * head_size))
    value_load = tl.load(value_ptr + src_value_idx + tile_pos,
                         mask=tile_pos < (num_heads * head_size))
    tl.store(value_cache_ptr + tgt_idx_v, value_load,
             mask=tile_pos < (num_heads * head_size))
    return

n = num_heads * head_size
grid = (num_tokens, triton.cdiv(n, TILE_SIZE))
print(f"grid={grid}, n={n}, TILE_SIZE={TILE_SIZE}")

reshape_and_cache_kernel_flash[grid](
    key, value, key_cache, value_cache, slot_mapping,
    torch.tensor(1.0, device=DEVICE), torch.tensor(1.0, device=DEVICE),
    key.stride(0), value.stride(0),
    key_cache.stride(0), 0, 0, 0, key_cache.stride(1),
    num_heads, head_size, block_size, 1, False, False, TILE_SIZE,
    num_warps=1, num_stages=1,
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
print(f"max_k={max_k}")
print(f"key[0] flat: {k_cpu[0].flatten().tolist()}")
print(f"kc[0,0] flat: {kc[0,0].flatten().tolist()}")
print(f"ref_kc[0,0] flat: {ref_kc[0,0].flatten().tolist()}")

# Check if strides differ
print(f"\nkey_cache strides: {key_cache.stride()}")
print(f"stride(0)={key_cache.stride(0)}, stride(1)={key_cache.stride(1)}")
print(f"constexpr stride(0)={block_size*num_heads*head_size}, stride(1)={num_heads*head_size}")
