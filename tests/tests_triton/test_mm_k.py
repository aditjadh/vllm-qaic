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

from vllm.lora.ops.triton_ops.kernel_utils import mm_k
from vllm.triton_utils import tl, triton

# ── Configuration ────────────────────────────────────────────────────────────
# mm_k is a @triton.jit device function (called from do_expand/do_shrink), not a
# standalone kernel. We wrap it in a launchable kernel that computes a single
# BLOCK_M x BLOCK_N tile of C = A @ B, iterating the full K dimension.
M       = 64
N       = 64
K       = 128
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32                  # K % BLOCK_K == 0  -> EVEN_K = True
SPLIT_K = 1
DTYPE   = torch.float16
DEVICE  = "qaic"

KERNEL_NAME = "mm_k (via wrapper kernel)"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_mm_k_{timestamp}.log")


@triton.jit
def mm_k_wrapper(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    a_m_stride, a_k_stride,
    b_k_stride, b_n_stride,
    c_m_stride, c_n_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    # Single program computes the whole BLOCK_M x BLOCK_N output tile.
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # A tile: [BLOCK_M, BLOCK_K], B tile: [BLOCK_K, BLOCK_N]
    a_tile = a_ptr + offs_m[:, None] * a_m_stride + offs_k[None, :] * a_k_stride
    b_tile = b_ptr + offs_k[:, None] * b_k_stride + offs_n[None, :] * b_n_stride

    accumulator = mm_k(
        a_tile,
        b_tile,
        a_k_stride,
        b_k_stride,
        offs_k,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        SPLIT_K,
        False,            # CAST_TYPE
        b_ptr.dtype.element_ty,
        False,            # USE_GDC
        base_k=0,
    )

    c = accumulator.to(c_ptr.dtype.element_ty)
    c_idx = c_ptr + offs_m[:, None] * c_m_stride + offs_n[None, :] * c_n_stride
    tl.store(c_idx, c)


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    torch.manual_seed(42)

    a = torch.randn(M, K, dtype=DTYPE, device=DEVICE)
    b = torch.randn(K, N, dtype=DTYPE, device=DEVICE)
    c = torch.empty(M, N, dtype=DTYPE, device=DEVICE)

    a_ref = a.clone()
    b_ref = b.clone()

    EVEN_K = (K % BLOCK_K == 0)
    grid = (1,)

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")

            log(f, "\nInputs:")
            log(f, f"  a : shape={tuple(a.shape)}, dtype={a.dtype}, device={a.device}")
            log(f, f"  b : shape={tuple(b.shape)}, dtype={b.dtype}, device={b.device}")
            log(f, f"  M={M}, N={N}, K={K}")
            log(f, f"  BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}, "
                   f"EVEN_K={EVEN_K}, SPLIT_K={SPLIT_K}")

            mm_k_wrapper[grid](
                a, b, c,
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_K=BLOCK_K,
                EVEN_K=EVEN_K,
                SPLIT_K=SPLIT_K,
            )

            # Force a device sync so any kernel compile/exec failure is raised
            # here rather than later at the first device read.
            c_host = c.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}")
            log(f, f"  block tile: BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}")

            log(f, "\nOutput:")
            log(f, f"  c : shape={tuple(c.shape)}, min={c_host.min().item():.4f}, "
                   f"max={c_host.max().item():.4f}, mean={c_host.float().mean().item():.4f}")

            # ── Reference validation ────────────────────────────────────────
            ref = (a_ref.float() @ b_ref.float())
            diff = (c_host.float() - ref.cpu()).abs().max().item()
            log(f, "\nValidation (vs PyTorch reference a @ b):")
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
