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

from vllm.lora.ops.triton_ops.kernel_utils import do_shrink_kernel
from vllm.triton_utils import tl, triton

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "do_shrink_kernel"

if "DOSHRINK_LOG_PATH" in os.environ:
    log_path = os.environ["DOSHRINK_LOG_PATH"]
else:
    log_path = os.path.join(LOG_DIR, f"test_do_shrink_kernel_{datetime.now():%Y-%m-%d_%H-%M-%S}.log")


# Wrap do_shrink_kernel with SLICE_NUM=1 (lora_ptr used directly, no int-to-ptr
# LUT). Computes out[m, :] = scaling * (input[ram[m]] @ lora[lora_index]^T).
@triton.jit
def do_shrink_wrapper(
    input_ptr, lora_ptr, out_ptr, ram_ptr,
    N, K, M_LEN,
    in_d0, in_d1, ld0, ld1, ld2, od0, od1, od2,
    scaling,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    ram = tl.load(ram_ptr + offs_m % M_LEN)
    do_shrink_kernel(
        0,          # pid_n
        0,          # pid_sk
        0,          # slice_id
        0,          # lora_index
        input_ptr, lora_ptr, out_ptr,
        N, K, M_LEN, ram,
        in_d0, in_d1,
        ld0, ld1, ld2,
        od0, od1, od2,
        scaling,
        BLOCK_M, BLOCK_N, BLOCK_K, EVEN_K,
        1,          # SPLIT_K
        1,          # SLICE_NUM
        False,      # USE_GDC
    )


def _mklog(f):
    def log(msg):
        print(msg); f.write(msg + "\n"); f.flush()
    return log


def main():
    torch.manual_seed(42)
    M, K, N = 8, 64, 16     # tokens, hidden, rank
    BLOCK_M, BLOCK_N, BLOCK_K = 16, 16, 64
    scaling = 0.5

    a = torch.randn(M, K, dtype=torch.float16, device=DEVICE)
    # lora A: [1, N, K] (num_loras=1)
    lora = torch.randn(1, N, K, dtype=torch.float16, device=DEVICE)
    # output: [num_slices=1, M, N]
    out = torch.zeros(1, M, N, dtype=torch.float16, device=DEVICE)
    ram = torch.arange(M, dtype=torch.int32, device=DEVICE)

    a_r, lora_r = a.clone(), lora.clone()
    EVEN_K = (K % BLOCK_K == 0)
    with open(log_path, "w") as f:
        log = _mklog(f)
        try:
            log(os.path.basename(log_path)[:-4]); log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  input(A): shape={tuple(a.shape)}, dtype={a.dtype}, device={a.device}")
            log(f"  lora(A): shape={tuple(lora.shape)} [num_loras, rank, hidden]")
            log(f"  M={M}, K(hidden)={K}, N(rank)={N}, scaling={scaling}, SLICE_NUM=1")

            do_shrink_wrapper[(1,)](
                a, lora, out, ram, N, K, M,
                a.stride(0), a.stride(1),
                lora.stride(0), lora.stride(1), lora.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                scaling,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, EVEN_K=EVEN_K,
            )
            o = out.cpu()[0]
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: (1,), BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}")
            log("\nOutput:")
            log(f"  out: shape={tuple(o.shape)}, mean={o.float().mean().item():.4f}")

            ref = scaling * (a_r.float() @ lora_r[0].float().T)
            diff = (o.float() - ref.cpu()).abs().max().item()
            log("\nValidation (vs scaling * A @ loraA^T):")
            log(f"  max abs diff: {diff:.4f}")
            torch.testing.assert_close(o.float(), ref.cpu(), atol=1e-1, rtol=1e-2)
            log("  assert_close: PASSED")
            log("\nSummary:")
            log("  Kernel execution: SUCCESS (grid (1,))")
            log("  Validation vs PyTorch reference: PASSED")
            log(f"  max abs diff: {diff:.4f}")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:")
            log("  Kernel execution: FAILED")
            log("  Validation vs PyTorch reference: NOT REACHED")
            sys.exit(1)
        finally:
            f.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    if os.environ.get("DOSHRINK_CHILD") == "1":
        main(); sys.exit(0)
    lp = os.path.join(LOG_DIR, f"test_do_shrink_kernel_{datetime.now():%Y-%m-%d_%H-%M-%S}.log")
    os.makedirs(LOG_DIR, exist_ok=True)
    env = dict(os.environ, DOSHRINK_CHILD="1", DOSHRINK_LOG_PATH=lp)
    proc = subprocess.run([sys.executable, sys.argv[0]], env=env, capture_output=True, text=True)
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="")
    if proc.returncode < 0 and not (proc.stdout and "Status: SUCCESS" in proc.stdout):
        with open(lp, "a") as f:
            f.write(f"\nStatus: FAILURE\n\nError:\nChild killed by signal "
                    f"(exit {proc.returncode}; -6/134 => SIGABRT) during compile.\n")
            f.write("\n".join(proc.stderr.strip().splitlines()[-6:]) + "\n")
            f.write("\nSummary:\n  Kernel execution: FAILED (compiler abort)\n")
            f.write("  Validation vs PyTorch reference: NOT REACHED\n\n" + "-" * 36 + "\n")
        print(f"\nLog saved to: {lp}")
    sys.exit(proc.returncode)
