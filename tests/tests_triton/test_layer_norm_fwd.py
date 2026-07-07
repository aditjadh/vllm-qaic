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

from vllm.model_executor.layers.fla.ops.layernorm_guard import (
    layer_norm_fwd_kernel,
    rms_norm_ref,
)
from vllm.triton_utils import triton

# ── Configuration ────────────────────────────────────────────────────────────
NUM_ROWS         = 16
HIDDEN_SIZE      = 1024
EPS              = 1e-6
IS_RMS_NORM      = False        # LayerNorm (set True for RMSNorm)
HAS_BIAS         = True
HAS_Z            = False        # gating branch
NORM_BEFORE_GATE = True
DTYPE            = torch.float16
DEVICE           = "qaic"

KERNEL_NAME = "layer_norm_fwd_kernel"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_layer_norm_fwd_{timestamp}.log")


def reference_layer_norm(x, weight, bias, z, eps, is_rms_norm):
    """Pure-PyTorch reference for (non-grouped) layer/rms norm."""
    if is_rms_norm:
        return rms_norm_ref(
            x, weight, bias, z=z, eps=eps,
            norm_before_gate=NORM_BEFORE_GATE, upcast=True,
        )
    # LayerNorm reference
    dtype = x.dtype
    x_f = x.float()
    w_f = weight.float()
    b_f = bias.float() if bias is not None else None
    z_f = z.float() if z is not None else None

    if z_f is not None and not NORM_BEFORE_GATE:
        x_f = x_f * (z_f * torch.sigmoid(z_f))

    mean = x_f.mean(dim=-1, keepdim=True)
    var = (x_f - mean).pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    x_hat = (x_f - mean) * rstd
    out = x_hat * w_f + b_f if b_f is not None else x_hat * w_f

    if z_f is not None and NORM_BEFORE_GATE:
        out = out * (z_f * torch.sigmoid(z_f))
    return out.to(dtype)


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    torch.manual_seed(42)

    x      = torch.randn(NUM_ROWS, HIDDEN_SIZE, dtype=DTYPE, device=DEVICE)
    weight = torch.randn(HIDDEN_SIZE, dtype=DTYPE, device=DEVICE)
    bias   = torch.randn(HIDDEN_SIZE, dtype=DTYPE, device=DEVICE) if HAS_BIAS else None
    z      = torch.randn(NUM_ROWS, HIDDEN_SIZE, dtype=DTYPE, device=DEVICE) if HAS_Z else None

    x_ref      = x.clone()
    weight_ref = weight.clone()
    bias_ref   = bias.clone() if bias is not None else None
    z_ref      = z.clone() if z is not None else None

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")

            log(f, "\nInputs:")
            log(f, f"  x      : shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}")
            log(f, f"  weight : shape={tuple(weight.shape)}, dtype={weight.dtype}")
            log(f, f"  bias   : {'None' if bias is None else tuple(bias.shape)}")
            log(f, f"  z      : {'None' if z is None else tuple(z.shape)}")
            log(f, f"  eps={EPS}, is_rms_norm={IS_RMS_NORM}, norm_before_gate={NORM_BEFORE_GATE}")

            # Launch the raw kernel directly. The layer_norm_fwd wrapper calls
            # torch.cuda.get_device_properties (unavailable here), so we pick
            # ROWS_PER_BLOCK manually and replicate the rest of the setup.
            M, N = x.shape
            group_size = N
            ngroups = N // group_size
            out = torch.empty_like(x)
            mean = (
                torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
                if not IS_RMS_NORM
                else None
            )
            rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)

            BLOCK_N = triton.next_power_of_2(group_size)
            ROWS_PER_BLOCK = 4
            num_warps = min(max(BLOCK_N // 256, 1), 8)
            grid = (triton.cdiv(M, ROWS_PER_BLOCK), ngroups)

            layer_norm_fwd_kernel[grid](
                x,
                out,
                weight,
                bias,
                z,
                mean,
                rstd,
                x.stride(0),
                out.stride(0),
                z.stride(0) if z is not None else 0,
                M,
                group_size,
                EPS,
                BLOCK_N=BLOCK_N,
                ROWS_PER_BLOCK=ROWS_PER_BLOCK,
                NORM_BEFORE_GATE=NORM_BEFORE_GATE,
                IS_RMS_NORM=IS_RMS_NORM,
                num_warps=num_warps,
            )

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}  (cdiv(M, ROWS_PER_BLOCK), ngroups)")
            log(f, f"  BLOCK_N: {BLOCK_N}, ROWS_PER_BLOCK: {ROWS_PER_BLOCK}, num_warps: {num_warps}")

            log(f, "\nOutput:")
            log(f, f"  out : shape={tuple(out.shape)}, min={out.min().item():.4f}, "
                   f"max={out.max().item():.4f}, mean={out.float().mean().item():.4f}")
            if mean is not None:
                log(f, f"  mean: shape={tuple(mean.shape)}, rstd: shape={tuple(rstd.shape)}")
            else:
                log(f, f"  mean: None (RMSNorm), rstd: shape={tuple(rstd.shape)}")

            # ── Reference validation ────────────────────────────────────────
            ref_out = reference_layer_norm(
                x_ref, weight_ref, bias_ref, z_ref, EPS, IS_RMS_NORM
            )
            diff = (out.float() - ref_out.float()).abs().max().item()
            log(f, "\nValidation (vs PyTorch reference):")
            log(f, f"  max abs diff: {diff:.6f}")

            torch.testing.assert_close(out.float(), ref_out.float(), atol=1e-2, rtol=1.6e-2)
            log(f, "  assert_close: PASSED")

            log(f, "\nSummary:")
            log(f, "  Kernel execution: SUCCESS")
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
