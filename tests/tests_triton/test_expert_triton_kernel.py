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

from vllm.model_executor.layers.fused_moe.fused_batched_moe import expert_triton_kernel
from vllm.triton_utils import tl

# ── Configuration ────────────────────────────────────────────────────────────
# expert_triton_kernel computes one expert's C = A @ B on the unquantized path.
M       = 16
N       = 64
K       = 128
DTYPE   = torch.float16
DEVICE  = "qaic"

BLOCK_M = 16
BLOCK_N = 64
BLOCK_K = 64

KERNEL_NAME = "expert_triton_kernel"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
if "EXPTK_LOG_PATH" in os.environ:
    log_path  = os.environ["EXPTK_LOG_PATH"]
    timestamp = os.path.basename(log_path)[len("test_expert_triton_kernel_"):-len(".log")]
else:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path  = os.path.join(LOG_DIR, f"test_expert_triton_kernel_{timestamp}.log")


def log(f, msg):
    print(msg)
    f.write(msg + "\n")
    f.flush()


def main():
    torch.manual_seed(42)

    a = torch.randn(M, K, dtype=DTYPE, device=DEVICE)
    b = torch.randn(K, N, dtype=DTYPE, device=DEVICE)
    c = torch.zeros(M, N, dtype=DTYPE, device=DEVICE)

    a_ref = a.clone()
    b_ref = b.clone()

    grid = (1,)

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")

            log(f, "\nInputs:")
            log(f, f"  a : shape={tuple(a.shape)}, dtype={a.dtype}, device={a.device}")
            log(f, f"  b : shape={tuple(b.shape)}, dtype={b.dtype}")
            log(f, f"  M={M}, N={N}, K={K}")
            log(f, f"  BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}")
            log(f, "  quant: unquantized")

            offs_bn = torch.arange(0, BLOCK_N, dtype=torch.int32, device=DEVICE) % N

            expert_triton_kernel[grid](
                a, b, c,
                0,                        # expert_id
                tl.float16,               # compute_type
                M, N, K,
                None, None, None,         # a_scale, b_scale, b_zp
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                0, 0, 0,                  # stride_ase, stride_asm, stride_ask
                0, 0, 0,                  # stride_bse, stride_bsk, stride_bsn
                offs_bn,
                0, 0,                     # group_n, group_k
                False, False, False,      # use_fp8_w8a8, use_int8_w8a16, per_act_token_quant
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_K=BLOCK_K,
            )

            c_host = c.cpu()   # force sync

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}")
            log(f, f"  block: BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}")

            log(f, "\nOutput:")
            log(f, f"  c : shape={tuple(c.shape)}, min={c_host.min().item():.4f}, "
                   f"max={c_host.max().item():.4f}, mean={c_host.float().mean().item():.4f}")

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
    if os.environ.get("EXPTK_CHILD") == "1":
        main()
        sys.exit(0)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path  = os.path.join(LOG_DIR, f"test_expert_triton_kernel_{timestamp}.log")

    env = dict(os.environ, EXPTK_CHILD="1", EXPTK_LOG_PATH=log_path)
    proc = subprocess.run([sys.executable, __file__], env=env,
                          capture_output=True, text=True)
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="")

    if proc.returncode < 0 and not (proc.stdout and "Status: SUCCESS" in proc.stdout):
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
