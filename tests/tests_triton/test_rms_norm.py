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

from vllm.model_executor.layers.batch_invariant import rms_norm

# ── Configuration ────────────────────────────────────────────────────────────
NUM_ROWS    = 16
HIDDEN_SIZE = 1024
EPS         = 1e-6
DTYPE       = torch.float16
DEVICE      = "qaic"

KERNEL_NAME = "_rms_norm_kernel"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_rms_norm_{timestamp}.log")


def reference_rms_norm(x, weight, eps):
    """Pure-PyTorch reference: y = x / sqrt(mean(x^2) + eps) * weight."""
    x_f32 = x.float()
    mean_sq = x_f32.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(mean_sq + eps)
    out = x_f32 * inv_rms * weight.float()
    return out.to(x.dtype)


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    torch.manual_seed(42)

    x      = torch.randn(NUM_ROWS, HIDDEN_SIZE, dtype=DTYPE, device=DEVICE)
    weight = torch.randn(HIDDEN_SIZE, dtype=DTYPE, device=DEVICE)

    x_ref      = x.clone()
    weight_ref = weight.clone()

    grid       = (NUM_ROWS,)
    BLOCK_SIZE = 1024

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")

            log(f, "\nInputs:")
            log(f, f"  x      : shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}")
            log(f, f"  weight : shape={tuple(weight.shape)}, dtype={weight.dtype}")
            log(f, f"  eps    : {EPS}")

            out = rms_norm(x, weight, eps=EPS)

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}")
            log(f, f"  BLOCK_SIZE: {BLOCK_SIZE}")

            log(f, "\nOutput:")
            log(f, f"  out : shape={tuple(out.shape)}, min={out.min().item():.4f}, "
                   f"max={out.max().item():.4f}, mean={out.float().mean().item():.4f}")

            # ── Reference validation ────────────────────────────────────────
            ref_out = reference_rms_norm(x_ref, weight_ref, EPS)
            diff = (out.float() - ref_out.float()).abs().max().item()
            log(f, "\nValidation (vs PyTorch reference):")
            log(f, f"  max abs diff: {diff:.6f}")

            torch.testing.assert_close(out.float(), ref_out.float(), atol=1e-2, rtol=1.6e-2)
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
