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

from vllm.lora.ops.triton_ops.kernel_utils import do_expand_kernel
from vllm.triton_utils import tl, triton

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "do_expand_kernel"

if "DOEXPAND_LOG_PATH" in os.environ:
    log_path = os.environ["DOEXPAND_LOG_PATH"]
else:
    log_path = os.path.join(LOG_DIR, f"test_do_expand_kernel_{datetime.now():%Y-%m-%d_%H-%M-%S}.log")


# Wrap do_expand_kernel with SLICE_NUM=1, SAME_STRIDE=True (strides as ints,
# lora_ptr used directly). Computes out[m, :] = input[ram[m]] @ lora[idx].
@triton.jit
def do_expand_wrapper(
    input_ptr, lora_ptr, out_ptr, ram_ptr,
    N, K,
    in_d0, in_d1, in_d2, ld0, ld1, ld2, od0, od1,
    M_LEN,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    ram = tl.load(ram_ptr + offs_m % M_LEN)
    do_expand_kernel(
        0,          # pid_n
        0,          # lora_index
        0,          # slice_id
        input_ptr, lora_ptr, out_ptr,
        N, K, M_LEN, ram,
        0,          # slice_start_loc
        in_d0, in_d1, in_d2,
        ld0, ld1, ld2,
        od0, od1,
        BLOCK_M, BLOCK_N, BLOCK_K,
        True,       # SAME_STRIDE
        1,          # SLICE_NUM
        EVEN_K,
        False,      # CAST_TYPE
        False,      # ADD_INPUTS
        False,      # USE_GDC
    )


def _mklog(f):
    def log(msg):
        print(msg); f.write(msg + "\n"); f.flush()
    return log


def main():
    torch.manual_seed(42)
    M, K, N = 8, 16, 64     # tokens, rank(in), hidden(out)
    BLOCK_M, BLOCK_N, BLOCK_K = 16, 64, 16
    scaling = 1.0

    # input: [num_slices=1, M, K]
    a = torch.randn(1, M, K, dtype=torch.float16, device=DEVICE)
    # lora B: [num_loras=1, N, K]
    lora = torch.randn(1, N, K, dtype=torch.float16, device=DEVICE)
    out = torch.zeros(M, N, dtype=torch.float16, device=DEVICE)
    ram = torch.arange(M, dtype=torch.int32, device=DEVICE)

    a_r, lora_r = a.clone(), lora.clone()
    EVEN_K = (K % BLOCK_K == 0)
    with open(log_path, "w") as f:
        log = _mklog(f)
        try:
            log(os.path.basename(log_path)[:-4]); log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  input: shape={tuple(a.shape)} [num_slices, M, rank], dtype={a.dtype}, device={a.device}")
            log(f"  lora(B): shape={tuple(lora.shape)} [num_loras, hidden, rank]")
            log(f"  M={M}, K(rank)={K}, N(hidden)={N}, SLICE_NUM=1, SAME_STRIDE=True")

            do_expand_wrapper[(1,)](
                a, lora, out, ram, N, K,
                a.stride(0), a.stride(1), a.stride(2),
                lora.stride(0), lora.stride(1), lora.stride(2),
                out.stride(0), out.stride(1),
                M,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, EVEN_K=EVEN_K,
            )
            o = out.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: (1,), BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}")
            log("\nOutput:")
            log(f"  out: shape={tuple(o.shape)}, mean={o.float().mean().item():.4f}")

            ref = a_r[0].float() @ lora_r[0].float().T   # [M, N]
            diff = (o.float() - ref.cpu()).abs().max().item()
            log("\nValidation (vs input @ loraB^T):")
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
    if os.environ.get("DOEXPAND_CHILD") == "1":
        main(); sys.exit(0)
    lp = os.path.join(LOG_DIR, f"test_do_expand_kernel_{datetime.now():%Y-%m-%d_%H-%M-%S}.log")
    os.makedirs(LOG_DIR, exist_ok=True)
    env = dict(os.environ, DOEXPAND_CHILD="1", DOEXPAND_LOG_PATH=lp)
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
