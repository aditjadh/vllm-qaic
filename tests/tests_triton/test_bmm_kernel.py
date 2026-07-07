# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
import os
import sys
import subprocess
import traceback
from datetime import datetime

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vllm"))

from vllm.model_executor.layers.batch_invariant import bmm_kernel
from vllm.triton_utils import triton

# ── Configuration ────────────────────────────────────────────────────────────
B       = 4
M       = 64
N       = 64
K       = 128
DTYPE   = torch.float16
DEVICE  = "qaic"

BLOCK_SIZE_M = 64
BLOCK_SIZE_N = 64
BLOCK_SIZE_K = 64

KERNEL_NAME = "bmm_kernel"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
# The parent owns the log path and passes it to the child via env so that an
# uncatchable compiler SIGABRT can still be recorded by the parent.
if "BMM_LOG_PATH" in os.environ:
    log_path = os.environ["BMM_LOG_PATH"]
    timestamp = os.path.basename(log_path)[len("test_bmm_kernel_"):-len(".log")]
else:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path  = os.path.join(LOG_DIR, f"test_bmm_kernel_{timestamp}.log")


def log(f, msg):
    print(msg)
    f.write(msg + "\n")
    f.flush()


def main():
    torch.manual_seed(42)

    a = torch.randn(B, M, K, dtype=DTYPE, device=DEVICE)
    b = torch.randn(B, K, N, dtype=DTYPE, device=DEVICE)
    c = torch.empty(B, M, N, dtype=DTYPE, device=DEVICE)

    a_ref = a.clone()
    b_ref = b.clone()

    grid = (
        B,
        triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
    )

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")

            log(f, "\nInputs:")
            log(f, f"  a : shape={tuple(a.shape)}, dtype={a.dtype}, device={a.device}")
            log(f, f"  b : shape={tuple(b.shape)}, dtype={b.dtype}, device={b.device}")
            log(f, f"  B={B}, M={M}, N={N}, K={K}")
            log(f, f"  BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_N={BLOCK_SIZE_N}, "
                   f"BLOCK_SIZE_K={BLOCK_SIZE_K}")

            bmm_kernel[grid](
                a, b, c,
                B, M, N, K,
                a.stride(0), a.stride(1), a.stride(2),
                b.stride(0), b.stride(1), b.stride(2),
                c.stride(0), c.stride(1), c.stride(2),
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                A_LARGE=a.numel() > 2**31,
                B_LARGE=b.numel() > 2**31,
                C_LARGE=c.numel() > 2**31,
            )

            # Force a device sync so any kernel compile/exec failure is raised
            # here rather than later at the first device read.
            c_host = c.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}  (B, tiles_per_matrix)")
            log(f, f"  block: BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_N={BLOCK_SIZE_N}, "
                   f"BLOCK_SIZE_K={BLOCK_SIZE_K}")

            log(f, "\nOutput:")
            log(f, f"  c : shape={tuple(c.shape)}, min={c_host.min().item():.4f}, "
                   f"max={c_host.max().item():.4f}, mean={c_host.float().mean().item():.4f}")

            # ── Reference validation ────────────────────────────────────────
            ref = torch.bmm(a_ref.float(), b_ref.float())
            diff = (c_host.float() - ref.cpu()).abs().max().item()
            log(f, "\nValidation (vs PyTorch reference torch.bmm):")
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
    # Child role: actually run the kernel.
    if os.environ.get("BMM_CHILD") == "1":
        main()
        sys.exit(0)

    # Parent role: own the log file, run the kernel launch in a child process,
    # and if the child is killed by an uncatchable compiler abort (SIGABRT),
    # record a FAILURE entry that the child could not write itself.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path  = os.path.join(LOG_DIR, f"test_bmm_kernel_{timestamp}.log")

    env = dict(os.environ, BMM_CHILD="1", BMM_LOG_PATH=log_path)
    proc = subprocess.run([sys.executable, __file__], env=env,
                          capture_output=True, text=True)
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="")

    if proc.returncode != 0 and not (proc.stdout and "Status: SUCCESS" in proc.stdout):
        # Append a failure record if the child aborted before logging it.
        with open(log_path, "a") as f:
            f.write(f"\nStatus: FAILURE\n\nError:\n")
            f.write(f"Child process exited with code {proc.returncode} "
                    f"(negative => killed by signal; -6/134 => SIGABRT).\n")
            f.write("The Triton->Hexagon compiler aborted with an MLIR "
                    "assertion before the kernel could run:\n")
            tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
            f.write(tail + "\n")
            f.write("\nSummary:\n")
            f.write("  Kernel execution: FAILED (compiler SIGABRT, uncatchable)\n")
            f.write("  Validation vs PyTorch reference: NOT REACHED\n")
            f.write("\n" + "-" * 36 + "\n")
        print(f"\nLog saved to: {log_path}")
    sys.exit(proc.returncode)
