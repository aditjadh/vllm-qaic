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

from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
    invoke_moe_batched_triton_kernel,
)
from vllm.triton_utils import tl

# ── Configuration ────────────────────────────────────────────────────────────
# batched_triton_kernel: per-expert batched GEMM, A[E, max_tokens, K] @ B[E,N,K]
# -> C[E, max_tokens, N], unquantized path.
E              = 4
MAX_NUM_TOKENS = 16
N              = 64
K             = 128
DTYPE         = torch.float16
DEVICE        = "qaic"

BLOCK_SIZE_M = 16
BLOCK_SIZE_N = 64
BLOCK_SIZE_K = 64

KERNEL_NAME = "batched_triton_kernel"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
if "BTK_LOG_PATH" in os.environ:
    log_path  = os.environ["BTK_LOG_PATH"]
    timestamp = os.path.basename(log_path)[len("test_batched_triton_kernel_"):-len(".log")]
else:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path  = os.path.join(LOG_DIR, f"test_batched_triton_kernel_{timestamp}.log")


def log(f, msg):
    print(msg)
    f.write(msg + "\n")
    f.flush()


def main():
    torch.manual_seed(42)

    # Each expert processes a different number of valid tokens.
    expert_num_tokens = torch.tensor([16, 8, 4, 0], dtype=torch.int32, device=DEVICE)

    a = torch.randn(E, MAX_NUM_TOKENS, K, dtype=DTYPE, device=DEVICE)
    b = torch.randn(E, N, K, dtype=DTYPE, device=DEVICE)        # [E, N, K]
    c = torch.zeros(E, MAX_NUM_TOKENS, N, dtype=DTYPE, device=DEVICE)

    a_ref = a.clone()
    b_ref = b.clone()
    ent_ref = expert_num_tokens.clone()

    config = {
        "BLOCK_SIZE_M": BLOCK_SIZE_M,
        "BLOCK_SIZE_N": BLOCK_SIZE_N,
        "BLOCK_SIZE_K": BLOCK_SIZE_K,
    }

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")

            log(f, "\nInputs:")
            log(f, f"  a (E,Mtok,K) : shape={tuple(a.shape)}, dtype={a.dtype}, device={a.device}")
            log(f, f"  b (E,N,K)    : shape={tuple(b.shape)}, dtype={b.dtype}")
            log(f, f"  c (E,Mtok,N) : shape={tuple(c.shape)}")
            log(f, f"  expert_num_tokens: {expert_num_tokens.tolist()}")
            log(f, f"  E={E}, max_num_tokens={MAX_NUM_TOKENS}, N={N}, K={K}")
            log(f, f"  BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_N={BLOCK_SIZE_N}, "
                   f"BLOCK_SIZE_K={BLOCK_SIZE_K}")
            log(f, "  quant: unquantized")

            invoke_moe_batched_triton_kernel(
                A=a, B=b, C=c,
                expert_num_tokens=expert_num_tokens,
                compute_type=tl.float16,
                A_scale=None, B_scale=None, B_zp=None,
                use_fp8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                config=config,
                per_act_token_quant=False,
                block_shape=None,
            )

            c_host = c.cpu()   # force sync

            grid = (E, ((MAX_NUM_TOKENS + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M)
                       * ((N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N))
            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}  (E, m_blocks * n_blocks)")
            log(f, f"  block: BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_N={BLOCK_SIZE_N}, "
                   f"BLOCK_SIZE_K={BLOCK_SIZE_K}")

            log(f, "\nOutput:")
            log(f, f"  c : shape={tuple(c.shape)}, min={c_host.min().item():.4f}, "
                   f"max={c_host.max().item():.4f}, mean={c_host.float().mean().item():.4f}")

            # ── Reference: per expert, first n valid tokens = A @ B^T ────────
            ref = torch.zeros(E, MAX_NUM_TOKENS, N, dtype=torch.float32)
            for e in range(E):
                num = int(ent_ref[e])
                if num == 0:
                    continue
                ref[e, :num] = a_ref[e, :num].float() @ b_ref[e].float().T
            diff = (c_host.float() - ref.cpu()).abs().max().item()
            log(f, "\nValidation (vs PyTorch per-expert A @ B^T, valid tokens only):")
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
    if os.environ.get("BTK_CHILD") == "1":
        main()
        sys.exit(0)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path  = os.path.join(LOG_DIR, f"test_batched_triton_kernel_{timestamp}.log")

    env = dict(os.environ, BTK_CHILD="1", BTK_LOG_PATH=log_path)
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
