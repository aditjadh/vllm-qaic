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

from vllm.model_executor.layers.mamba.ops.layernorm_gated import (
    _layer_norm_fwd_1pass_kernel,
)
from vllm.triton_utils import triton

# ── Configuration ────────────────────────────────────────────────────────────
NUM_ROWS         = 16
HIDDEN_SIZE      = 1024
EPS              = 1e-6
IS_RMS_NORM      = True         # rms_norm_gated path (set False for LayerNorm)
HAS_BIAS         = False        # RMSNorm typically has no bias
HAS_Z            = False        # gating branch
NORM_BEFORE_GATE = True
DTYPE            = torch.float16
DEVICE           = "qaic"

KERNEL_NAME = "_layer_norm_fwd_1pass_kernel"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_layer_norm_fwd_1pass_{timestamp}.log")


def reference_norm(x, weight, bias, z, eps, is_rms_norm):
    """Pure-PyTorch reference mirroring the 1-pass kernel."""
    dtype = x.dtype
    x_f = x.float()
    w_f = weight.float()
    b_f = bias.float() if bias is not None else None
    z_f = z.float() if z is not None else None

    if z_f is not None and not NORM_BEFORE_GATE:
        x_f = x_f * (z_f * torch.sigmoid(z_f))

    if not is_rms_norm:
        mean = x_f.mean(dim=-1, keepdim=True)
        var = (x_f - mean).pow(2).mean(dim=-1, keepdim=True)
        x_hat = (x_f - mean) * torch.rsqrt(var + eps)
    else:
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_hat = x_f * torch.rsqrt(var + eps)

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

            MAX_FUSED_SIZE = 65536 // x.element_size()
            BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
            num_warps = min(max(BLOCK_N // 256, 1), 8)
            grid = (M, ngroups)

            # HAS_BIAS / HAS_Z are resolved by the kernel's @triton.heuristics
            # from the B / Z arguments, so they are not passed explicitly.
            _layer_norm_fwd_1pass_kernel[grid](
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
                NORM_BEFORE_GATE=NORM_BEFORE_GATE,
                IS_RMS_NORM=IS_RMS_NORM,
                num_warps=num_warps,
            )

            # Force a device sync so a kernel compile/exec failure is raised
            # here (and captured in the traceback) rather than surfacing later
            # at the first .item() as a misleading "synchronize stream" error.
            out = out.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}  (M, ngroups)")
            log(f, f"  BLOCK_N: {BLOCK_N}, num_warps: {num_warps}")

            log(f, "\nOutput:")
            log(f, f"  out : shape={tuple(out.shape)}, min={out.min().item():.4f}, "
                   f"max={out.max().item():.4f}, mean={out.float().mean().item():.4f}")
            log(f, f"  rstd: shape={tuple(rstd.shape)}, "
                   f"mean: {'None (RMSNorm)' if mean is None else tuple(mean.shape)}")

            # ── Reference validation ────────────────────────────────────────
            ref_out = reference_norm(
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
