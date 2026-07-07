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

# flashinfer.py imports `flashinfer` which is not installed; load only the
# kernel and its wrapper by importing them from a minimal shim.
from vllm.triton_utils import tl, triton

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "_trtllm_prefill_attn_kvfp8_dequant"


# ---- kernel definition (copied verbatim from flashinfer.py) ----
@triton.jit
def _trtllm_prefill_attn_kvfp8_dequant(
    kv_cache_ptr,
    block_tables_prefill_ptr,
    block_table_stride,
    mock_kv_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    K_CACHE_STRIDE: tl.constexpr,
    KV_CACHE_STRIDE: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    mock_block_table_idx = tl.program_id(1).to(tl.int64)
    orig_page_num = tl.load(
        block_tables_prefill_ptr + batch_idx * block_table_stride + mock_block_table_idx
    ).to(tl.int64)
    if orig_page_num <= 0:
        return
    dequant_dtype = mock_kv_cache_ptr.dtype.element_ty

    k_scale_val = tl.load(k_scale_ptr)
    offset = orig_page_num * KV_CACHE_STRIDE + tl.arange(0, K_CACHE_STRIDE)
    fp8_vals = tl.load(kv_cache_ptr + offset)
    dequantized_vals = fp8_vals.to(tl.float32) * k_scale_val
    mock_cache_offset = (
        batch_idx * block_table_stride + mock_block_table_idx + 1
    ) * KV_CACHE_STRIDE + tl.arange(0, K_CACHE_STRIDE)
    dequantized_vals = dequantized_vals.to(dequant_dtype)
    tl.store(mock_kv_cache_ptr + mock_cache_offset, dequantized_vals)

    v_scale_val = tl.load(v_scale_ptr)
    offset = (
        orig_page_num * KV_CACHE_STRIDE + K_CACHE_STRIDE + tl.arange(0, K_CACHE_STRIDE)
    )
    fp8_vals = tl.load(kv_cache_ptr + offset)
    dequantized_vals = fp8_vals.to(tl.float32) * v_scale_val
    mock_cache_offset = (
        (batch_idx * block_table_stride + mock_block_table_idx + 1) * KV_CACHE_STRIDE
        + K_CACHE_STRIDE
        + tl.arange(0, K_CACHE_STRIDE)
    )
    dequantized_vals = dequantized_vals.to(dequant_dtype)
    tl.store(mock_kv_cache_ptr + mock_cache_offset, dequantized_vals)
# ----------------------------------------------------------------


def trtllm_prefill_attn_kvfp8_dequant(kv_cache, block_tables_prefill,
                                       k_scale, v_scale, dequant_dtype):
    batch_size, num_of_page_per_token = block_tables_prefill.shape
    s = kv_cache.shape
    k_cache_stride = s[2] * s[3] * s[4]
    kv_cache_stride = k_cache_stride * s[1]
    new_s = (batch_size * num_of_page_per_token + 1, s[1], s[2], s[3], s[4])
    mock_kv_cache = torch.empty(new_s, dtype=dequant_dtype, device=kv_cache.device)
    mock_block_table = torch.arange(
        start=1, end=batch_size * num_of_page_per_token + 1,
        dtype=torch.int32, device=block_tables_prefill.device,
    ).reshape(batch_size, num_of_page_per_token)
    grid = (batch_size, num_of_page_per_token)
    _trtllm_prefill_attn_kvfp8_dequant[grid](
        kv_cache, block_tables_prefill, num_of_page_per_token,
        mock_kv_cache, k_scale, v_scale,
        k_cache_stride, kv_cache_stride,
    )
    return mock_kv_cache, mock_block_table


def main(log_path):
    torch.manual_seed(42)
    # kv_cache: [num_total_pages, 2, num_heads, head_dim, page_size]
    # K_CACHE_STRIDE must be power of 2 for QAIC tl.arange
    batch_size = 2
    num_pages_per_token = 4
    num_heads = 4
    head_dim = 32
    page_size = 8          # K_CACHE_STRIDE = num_heads*head_dim*page_size = 4*32*8 = 1024
    num_total_pages = 16

    fp8_dtype = torch.float8_e4m3fn
    dequant_dtype = torch.float16

    # FP8 kv_cache
    kv_cache = torch.randint(-127, 127, (num_total_pages, 2, num_heads, head_dim, page_size),
                              dtype=torch.int8, device=DEVICE).view(fp8_dtype)

    # block table: [batch_size, num_pages_per_token] — logical page indices (> 0)
    block_tables_prefill = torch.randint(
        1, num_total_pages, (batch_size, num_pages_per_token), dtype=torch.int32, device=DEVICE
    )

    k_scale = torch.tensor(0.1, dtype=torch.float32, device=DEVICE)
    v_scale = torch.tensor(0.2, dtype=torch.float32, device=DEVICE)

    k_cache_stride = num_heads * head_dim * page_size

    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            log(os.path.basename(log_path)[:-4]); log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  kv_cache: {tuple(kv_cache.shape)} [pages,2,heads,head_dim,page_size], dtype={kv_cache.dtype}, device={kv_cache.device}")
            log(f"  block_tables_prefill: {tuple(block_tables_prefill.shape)} [batch,pages_per_token]")
            log(f"  k_scale={k_scale.item()}, v_scale={v_scale.item()}, dequant_dtype={dequant_dtype}")
            log(f"  K_CACHE_STRIDE={k_cache_stride}, KV_CACHE_STRIDE={k_cache_stride*2}")

            mock_kv, mock_bt = trtllm_prefill_attn_kvfp8_dequant(
                kv_cache, block_tables_prefill, k_scale, v_scale, dequant_dtype
            )
            mkv = mock_kv.cpu(); mbt = mock_bt.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: ({batch_size}, {num_pages_per_token})  (batch_size, pages_per_token)")
            log(f"  K_CACHE_STRIDE={k_cache_stride} (constexpr), KV_CACHE_STRIDE={k_cache_stride*2} (constexpr)")
            log("\nOutput:")
            log(f"  mock_kv_cache: shape={tuple(mkv.shape)}, dtype={mkv.dtype}, mean={mkv.float().mean().item():.4f}")
            log(f"  mock_block_table: shape={tuple(mbt.shape)}, values={mbt.flatten().tolist()}")

            # reference: dequantize K and V pages directly
            kv_flat = kv_cache.view(torch.float8_e4m3fn).cpu()
            ref_errors = []
            for b in range(batch_size):
                for p in range(num_pages_per_token):
                    page = int(block_tables_prefill[b, p].item())
                    if page <= 0:
                        continue
                    k_fp8 = kv_flat[page, 0].float() * k_scale.item()
                    v_fp8 = kv_flat[page, 1].float() * v_scale.item()
                    mock_idx = b * num_pages_per_token + p + 1
                    k_got = mkv[mock_idx, 0].float()
                    v_got = mkv[mock_idx, 1].float()
                    ref_errors.append((k_fp8 - k_got).abs().max().item())
                    ref_errors.append((v_fp8 - v_got).abs().max().item())
            max_diff = max(ref_errors) if ref_errors else 0.0
            log("\nValidation (vs direct FP8 dequant reference):")
            log(f"  max abs diff: {max_diff:.6f}")
            assert max_diff < 1e-2, f"max diff {max_diff} too large"
            log("  check: PASSED")
            log("\nSummary:")
            log(f"  Kernel execution: SUCCESS (grid ({batch_size},{num_pages_per_token}))")
            log("  Validation vs reference: PASSED")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Kernel execution: FAILED")
            fp.write("\n" + "-" * 36 + "\n"); sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "trtllm_prefill_attn_kvfp8_dequant")
