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
KERNEL_NAME = "_indexer_k_quant_and_cache_kernel"


@triton.jit
def _indexer_k_quant_and_cache_kernel(
    k_ptr,
    kv_cache_ptr,
    kv_cache_scale_ptr,
    slot_mapping_ptr,
    kv_cache_scale_stride,
    kv_cache_value_stride,
    block_size,
    num_tokens,
    head_dim: tl.constexpr,
    LAYOUT: tl.constexpr,
    BLOCK_TILE_SIZE: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr,
    IS_FNUZ: tl.constexpr,
    USE_UE8M0: tl.constexpr,
):
    tid = tl.program_id(0)
    offset = tl.arange(0, head_dim)
    if LAYOUT == "SHUFFLE":
        tile_offset = (offset // HEAD_TILE_SIZE * BLOCK_TILE_SIZE * HEAD_TILE_SIZE
                       + offset % HEAD_TILE_SIZE)
    else:
        tile_offset = offset
    tile_store_offset = tile_offset
    src_ptr = k_ptr + tid * head_dim
    slot_id = tl.load(slot_mapping_ptr + tid)
    if slot_id < 0:
        return
    block_id = slot_id // block_size
    block_offset = slot_id % block_size
    tile_block_id = block_offset // BLOCK_TILE_SIZE
    tile_block_offset = block_offset % BLOCK_TILE_SIZE
    val = tl.load(src_ptr + offset)
    amax = tl.max(val.abs(), axis=-1).to(tl.float32)
    if IS_FNUZ:
        scale = tl.maximum(1e-4, amax) / 224.0
    else:
        scale = tl.maximum(1e-4, amax) / 448.0
    if USE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    fp8_val = (val.to(tl.float32) / scale).to(kv_cache_ptr.type.element_ty)
    if LAYOUT == "SHUFFLE":
        dst_ptr = (kv_cache_ptr + block_id * kv_cache_value_stride
                   + tile_block_id * BLOCK_TILE_SIZE * head_dim
                   + tile_block_offset * HEAD_TILE_SIZE)
    else:
        dst_ptr = (kv_cache_ptr + block_id * kv_cache_value_stride
                   + block_offset * head_dim)
    tl.store(dst_ptr + tile_store_offset, fp8_val)
    dst_scale_ptr = (kv_cache_scale_ptr + block_id * kv_cache_scale_stride
                     + block_offset)
    tl.store(dst_scale_ptr, scale)


def main(log_path):
    torch.manual_seed(42)
    num_tokens = 4
    head_dim = 8   # must be power-of-2 for tl.arange
    block_size = 4
    num_blocks = 8
    BLOCK_TILE_SIZE = 4
    HEAD_TILE_SIZE = 8  # after element_size division this is used as constexpr

    # k: [num_tokens, head_dim]
    k = torch.randn(num_tokens, head_dim, dtype=torch.float32, device=DEVICE)

    slot_mapping = torch.tensor([0, 1, 4, 5], dtype=torch.int32, device=DEVICE)

    grid = (num_tokens,)

    with open(log_path, "w") as fp:
        log = logger(fp)
        log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        log(f"Kernel: {KERNEL_NAME}")
        log("\nInputs:")
        log(f"  k: {tuple(k.shape)} [num_tokens,head_dim], device={k.device}")
        log(f"  head_dim={head_dim}, block_size={block_size}, BLOCK_TILE_SIZE={BLOCK_TILE_SIZE}")
        log(f"  slot_mapping: {slot_mapping.cpu().tolist()}")
        log(f"  LAYOUT='NHD', IS_FNUZ=False, USE_UE8M0=False")
        try:
            # FP8 may not be supported on QAIC
            fp8_dtype = torch.float8_e4m3fn
            kv_cache_value = torch.zeros(num_blocks * block_size * head_dim,
                                          dtype=fp8_dtype, device=DEVICE)
            # kv_cache scale portion: [num_blocks, block_size] (float32)
            kv_cache_scale = torch.zeros(num_blocks, block_size,
                                          dtype=torch.float32, device=DEVICE)

            _indexer_k_quant_and_cache_kernel[grid](
                k,
                kv_cache_value,
                kv_cache_scale,
                slot_mapping,
                kv_cache_scale.stride(0),
                num_blocks * block_size * head_dim // num_blocks,  # value stride per block
                block_size,
                num_tokens,
                head_dim=head_dim,
                LAYOUT="NHD",
                BLOCK_TILE_SIZE=BLOCK_TILE_SIZE,
                HEAD_TILE_SIZE=HEAD_TILE_SIZE,
                IS_FNUZ=False,
                USE_UE8M0=False,
                num_warps=1,
                num_stages=1,
            )

            kc = kv_cache_value.cpu().view(torch.float8_e4m3fn)
            sc = kv_cache_scale.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: {grid}  (one program per token)")
            log(f"  head_dim={head_dim} (constexpr), BLOCK_TILE_SIZE={BLOCK_TILE_SIZE} (constexpr)")
            log("\nOutput:")
            log(f"  kv_cache_value (fp8): {kc.shape}, non-zero={kc.view(torch.uint8).ne(0).sum().item()}")
            log(f"  kv_cache_scale: {sc.shape}, written slots: {sc.ne(0.0).sum().item()}")

            # reference check: dequantize back and compare
            k_cpu = k.cpu()
            sc_flat = sc.view(-1)
            kc_flat = kc.view(-1).to(torch.float32)
            errors = []
            for t in range(num_tokens):
                slot = slot_mapping[t].item()
                bi = slot // block_size
                bo = slot % block_size
                scale_val = sc[bi, bo].item()
                kc_deq = kc_flat[bi * block_size * head_dim + bo * head_dim:
                                  bi * block_size * head_dim + (bo + 1) * head_dim]
                k_ref = k_cpu[t]
                recon = kc_deq * scale_val
                errors.append((recon - k_ref.float()).abs().max().item())
            max_err = max(errors)
            log("\nValidation (dequant vs original k, tol=0.01):")
            log(f"  max abs err across tokens: {max_err:.6f}")
            assert max_err < 0.01, f"err={max_err}"
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
    run_test(main, LOG_DIR, "indexer_k_quant_and_cache")
