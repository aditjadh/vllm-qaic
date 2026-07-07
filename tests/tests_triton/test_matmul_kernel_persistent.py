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

from vllm.model_executor.layers.batch_invariant import matmul_kernel_persistent
from vllm.triton_utils import triton

# ── Configuration ────────────────────────────────────────────────────────────
M       = 128
N       = 128
K       = 128
DTYPE   = torch.float16
DEVICE  = "qaic"

BLOCK_SIZE_M = 64
BLOCK_SIZE_N = 64
BLOCK_SIZE_K = 64
GROUP_SIZE_M = 8
NUM_SMS      = 16          # picked manually (qaic has no cuda device props)

KERNEL_NAME = "matmul_kernel_persistent"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_matmul_kernel_persistent_{timestamp}.log")


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

    grid = (
        min(NUM_SMS, triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)),
    )

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")

            log(f, "\nInputs:")
            log(f, f"  a : shape={tuple(a.shape)}, dtype={a.dtype}, device={a.device}")
            log(f, f"  b : shape={tuple(b.shape)}, dtype={b.dtype}, device={b.device}")
            log(f, f"  M={M}, N={N}, K={K}")
            log(f, f"  BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_N={BLOCK_SIZE_N}, "
                   f"BLOCK_SIZE_K={BLOCK_SIZE_K}, GROUP_SIZE_M={GROUP_SIZE_M}, "
                   f"NUM_SMS={NUM_SMS}")

            matmul_kernel_persistent[grid](
                a, b, c,
                None,                # bias_ptr
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                GROUP_SIZE_M=GROUP_SIZE_M,
                NUM_SMS=NUM_SMS,
                A_LARGE=a.numel() > 2**31,
                B_LARGE=b.numel() > 2**31,
                C_LARGE=c.numel() > 2**31,
                HAS_BIAS=False,
            )

            # Force a device sync so any kernel compile/exec failure is raised
            # here rather than later at the first device read.
            c_host = c.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}  (persistent: min(NUM_SMS, num_tiles))")
            log(f, f"  block: BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_N={BLOCK_SIZE_N}, "
                   f"BLOCK_SIZE_K={BLOCK_SIZE_K}")

            log(f, "\nOutput:")
            log(f, f"  c : shape={tuple(c.shape)}, min={c_host.min().item():.4f}, "
                   f"max={c_host.max().item():.4f}, mean={c_host.float().mean().item():.4f}")

            # ── Reference validation ────────────────────────────────────────
            ref = a_ref.float() @ b_ref.float()
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
