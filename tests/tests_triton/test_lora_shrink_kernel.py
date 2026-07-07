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

import vllm.lora.ops.triton_ops.utils as lora_utils
from vllm.triton_utils import tl, triton

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "_lora_shrink_kernel"

if "LSHRINK_LOG_PATH" in os.environ:
    log_path = os.environ["LSHRINK_LOG_PATH"]
else:
    log_path = os.path.join(LOG_DIR, f"test_lora_shrink_kernel_{datetime.now():%Y-%m-%d_%H-%M-%S}.log")


def _mklog(f):
    def log(msg):
        print(msg); f.write(msg + "\n"); f.flush()
    return log


def main():
    torch.manual_seed(42)
    # QAIC rejects uint64; patch the pointer-table builder to use int64.
    _orig = lora_utils._get_lora_a_ptr

    def _patched(lora_a_weights, device):
        key = tuple(w.data_ptr() for w in lora_a_weights)
        if key in lora_utils._LORA_A_PTR_DICT:
            return lora_utils._LORA_A_PTR_DICT[key]
        ptrs = torch.tensor([w.data_ptr() for w in lora_a_weights], device=device, dtype=torch.int64)
        d0 = torch.tensor([w.stride(0) for w in lora_a_weights], device=device)
        d1 = torch.tensor([w.stride(1) for w in lora_a_weights], device=device)
        d2 = torch.tensor([w.stride(2) for w in lora_a_weights], device=device)
        res = (ptrs, d0, d1, d2)
        lora_utils._LORA_A_PTR_DICT[key] = res
        return res
    lora_utils._get_lora_a_ptr = _patched
    # ensure the op module picks up the patched symbol
    import vllm.lora.ops.triton_ops.lora_shrink_op as op
    op._get_lora_a_ptr = _patched

    M, K, N = 8, 64, 16     # tokens, hidden, rank
    num_loras = 1
    scaling = 0.5

    inputs = torch.randn(M, K, dtype=torch.float16, device=DEVICE)
    lora_a = [torch.randn(num_loras, N, K, dtype=torch.float16, device=DEVICE)]
    output = torch.zeros(1, M, N, dtype=torch.float16, device=DEVICE)

    # all tokens use lora 0
    token_lora_mapping = torch.zeros(M, dtype=torch.int32, device=DEVICE)
    token_indices_sorted = torch.arange(M, dtype=torch.int32, device=DEVICE)
    num_tokens_per_lora = torch.tensor([M, 0], dtype=torch.int32, device=DEVICE)
    lora_token_start_loc = torch.tensor([0, M, M], dtype=torch.int32, device=DEVICE)
    lora_ids = torch.tensor([0, -1], dtype=torch.int32, device=DEVICE)
    no_lora_flag = torch.tensor([False], dtype=torch.bool)  # CPU

    in_r, la_r = inputs.clone(), lora_a[0].clone()
    with open(log_path, "w") as f:
        log = _mklog(f)
        try:
            log(os.path.basename(log_path)[:-4]); log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  inputs: shape={tuple(inputs.shape)}, dtype={inputs.dtype}, device={inputs.device}")
            log(f"  lora_a: shape={tuple(lora_a[0].shape)} [num_loras, rank, hidden]")
            log(f"  M={M}, K(hidden)={K}, N(rank)={N}, num_loras={num_loras}, scaling={scaling}")
            log("  NOTE: _get_lora_a_ptr patched to build the pointer LUT as int64 (QAIC rejects uint64)")

            op._lora_shrink(
                inputs, lora_a, output, token_lora_mapping, token_indices_sorted,
                num_tokens_per_lora, lora_token_start_loc, lora_ids, no_lora_flag, scaling,
            )
            o = output.cpu()[0]
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log("  grid: (SPLIT_K * cdiv(M,BM) * cdiv(N,BN), NUM_SLICES, MAX_LORAS)")
            log("\nOutput:")
            log(f"  output: shape={tuple(o.shape)}, mean={o.float().mean().item():.4f}")

            ref = scaling * (in_r.float() @ la_r[0].float().T)
            diff = (o.float() - ref.cpu()).abs().max().item()
            log("\nValidation (vs scaling * inputs @ loraA^T):")
            log(f"  max abs diff: {diff:.4f}")
            torch.testing.assert_close(o.float(), ref.cpu(), atol=1e-1, rtol=1e-2)
            log("  assert_close: PASSED")
            log("\nSummary:")
            log("  Kernel execution: SUCCESS")
            log("  Validation vs PyTorch reference: PASSED")
            log(f"  max abs diff: {diff:.4f}")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:")
            log("  Kernel execution: FAILED")
            log("  Validation vs PyTorch reference: NOT REACHED")
            log("  Note: LoRA kernels use an int->pointer LUT (tt.int_to_ptr) the")
            log("        QAIC backend has not lowered in prior tests.")
            sys.exit(1)
        finally:
            f.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    if os.environ.get("LSHRINK_CHILD") == "1":
        main(); sys.exit(0)
    lp = os.path.join(LOG_DIR, f"test_lora_shrink_kernel_{datetime.now():%Y-%m-%d_%H-%M-%S}.log")
    os.makedirs(LOG_DIR, exist_ok=True)
    env = dict(os.environ, LSHRINK_CHILD="1", LSHRINK_LOG_PATH=lp)
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
