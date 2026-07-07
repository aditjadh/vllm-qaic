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

from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
    scaled_mm_kernel,
)
from vllm.triton_utils import tl, triton

# ── Configuration ────────────────────────────────────────────────────────────
M       = 64
N       = 64
K       = 128
DTYPE   = torch.float16          # A/B dtype
OUT_DTYPE = torch.float16
DEVICE  = "qaic"

BLOCK_SIZE_M = 64
BLOCK_SIZE_N = 64
BLOCK_SIZE_K = 64
# Per-row scale_a (M,1), per-col scale_b (N,1) -> scale blocks equal tile dims.
BLOCK_SIZE_SCALE_A = BLOCK_SIZE_M
BLOCK_SIZE_SCALE_B = BLOCK_SIZE_N

KERNEL_NAME = "scaled_mm_kernel"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_scaled_mm_kernel_{timestamp}.log")


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    torch.manual_seed(42)

    a       = torch.randn(M, K, dtype=DTYPE, device=DEVICE)
    b       = torch.randn(K, N, dtype=DTYPE, device=DEVICE)
    scale_a = torch.rand(M, 1, dtype=torch.float32, device=DEVICE) + 0.5
    scale_b = torch.rand(N, 1, dtype=torch.float32, device=DEVICE) + 0.5
    bias    = torch.randn(N, dtype=OUT_DTYPE, device=DEVICE)
    c       = torch.empty(M, N, dtype=OUT_DTYPE, device=DEVICE)

    a_ref       = a.clone()
    b_ref       = b.clone()
    scale_a_ref = scale_a.clone()
    scale_b_ref = scale_b.clone()
    bias_ref    = bias.clone()

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")

            log(f, "\nInputs:")
            log(f, f"  a       : shape={tuple(a.shape)}, dtype={a.dtype}, device={a.device}")
            log(f, f"  b       : shape={tuple(b.shape)}, dtype={b.dtype}")
            log(f, f"  scale_a : shape={tuple(scale_a.shape)}, dtype={scale_a.dtype}")
            log(f, f"  scale_b : shape={tuple(scale_b.shape)}, dtype={scale_b.dtype}")
            log(f, f"  bias    : shape={tuple(bias.shape)}, dtype={bias.dtype}")
            log(f, f"  M={M}, N={N}, K={K}")
            log(f, f"  BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_N={BLOCK_SIZE_N}, "
                   f"BLOCK_SIZE_K={BLOCK_SIZE_K}")

            scaled_mm_kernel[grid](
                a, b, scale_a, scale_b, c, bias,
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                tl.float32,                       # ACCUMULATOR_DTYPE
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                BLOCK_SIZE_SCALE_A=BLOCK_SIZE_SCALE_A,
                BLOCK_SIZE_SCALE_B=BLOCK_SIZE_SCALE_B,
            )

            # Force a device sync so any kernel compile/exec failure is raised
            # here rather than later at the first device read.
            c_host = c.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}  (cdiv(M,BM) * cdiv(N,BN))")
            log(f, f"  block: BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_N={BLOCK_SIZE_N}, "
                   f"BLOCK_SIZE_K={BLOCK_SIZE_K}")

            log(f, "\nOutput:")
            log(f, f"  c : shape={tuple(c.shape)}, min={c_host.min().item():.4f}, "
                   f"max={c_host.max().item():.4f}, mean={c_host.float().mean().item():.4f}")

            # ── Reference validation ────────────────────────────────────────
            # out = scale_a * scale_b.T * (a @ b) + bias
            ref = (a_ref.float() @ b_ref.float())
            ref = scale_a_ref * ref                     # (M,1) broadcast over cols
            ref = ref * scale_b_ref.T                    # (1,N) broadcast over rows
            ref = ref + bias_ref.float()
            diff = (c_host.float() - ref.cpu()).abs().max().item()
            log(f, "\nValidation (vs PyTorch reference scale_a * scale_b * (a@b) + bias):")
            log(f, f"  max abs diff: {diff:.6f}")

            torch.testing.assert_close(c_host.float(), ref.cpu(), atol=1e-1, rtol=1e-2)
            log(f, "  assert_close: PASSED")

            log(f, "\nSummary:")
            log(f, f"  Kernel execution: SUCCESS (grid {grid})")
            log(f, "  Validation vs PyTorch reference: PASSED")
            log(f, f"  max abs diff: {diff:.6f}")

        except Exception:
            msg = f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}"
            log(f, msg)
            log(f, "\nSummary:")
            log(f, "  Validation vs PyTorch reference: FAILED")
            sys.exit(1)

        finally:
            log(f, "\n" + "-" * 36)

    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
