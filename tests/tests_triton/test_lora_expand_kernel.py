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

import vllm.lora.ops.triton_ops.lora_expand_op as op

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "_lora_expand_kernel"

if "LEXPAND_LOG_PATH" in os.environ:
    log_path = os.environ["LEXPAND_LOG_PATH"]
else:
    log_path = os.path.join(LOG_DIR, f"test_lora_expand_kernel_{datetime.now():%Y-%m-%d_%H-%M-%S}.log")


def _mklog(f):
    def log(msg):
        print(msg); f.write(msg + "\n"); f.flush()
    return log


def main():
    torch.manual_seed(42)
    # Single slice => _get_lora_b_ptr returns the weight tensor directly
    # (no uint64 pointer LUT), and do_expand_kernel uses it directly.
    M, K, N = 8, 16, 64     # tokens, rank(in), hidden(out)
    num_loras = 1
    add_inputs = False

    # inputs: [num_slices=1, num_tokens, rank]
    inputs = torch.randn(1, M, K, dtype=torch.float16, device=DEVICE)
    lora_b = [torch.randn(num_loras, N, K, dtype=torch.float16, device=DEVICE)]
    output = torch.zeros(M, N, dtype=torch.float16, device=DEVICE)

    token_lora_mapping = torch.zeros(M, dtype=torch.int32, device=DEVICE)
    token_indices_sorted = torch.arange(M, dtype=torch.int32, device=DEVICE)
    num_tokens_per_lora = torch.tensor([M, 0], dtype=torch.int32, device=DEVICE)
    lora_token_start_loc = torch.tensor([0, M, M], dtype=torch.int32, device=DEVICE)
    lora_ids = torch.tensor([0, -1], dtype=torch.int32, device=DEVICE)
    no_lora_flag = torch.tensor([False], dtype=torch.bool)

    in_r, lb_r = inputs.clone(), lora_b[0].clone()
    with open(log_path, "w") as f:
        log = _mklog(f)
        try:
            log(os.path.basename(log_path)[:-4]); log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  inputs: shape={tuple(inputs.shape)} [num_slices, num_tokens, rank], dtype={inputs.dtype}, device={inputs.device}")
            log(f"  lora_b: shape={tuple(lora_b[0].shape)} [num_loras, hidden, rank]")
            log(f"  M={M}, K(rank)={K}, N(hidden)={N}, num_loras={num_loras}, add_inputs={add_inputs}")
            log("  NOTE: single-slice path => lora_ptr is the weight tensor directly (no int->ptr LUT)")

            op._lora_expand(
                inputs, lora_b, output, token_lora_mapping, token_indices_sorted,
                num_tokens_per_lora, lora_token_start_loc, lora_ids, no_lora_flag,
                offset_start=0, add_inputs=add_inputs,
            )
            o = output.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log("  grid: (cdiv(M,BM)*cdiv(MAX_N,BN), NUM_SLICES, MAX_LORAS)")
            log("\nOutput:")
            log(f"  output: shape={tuple(o.shape)}, mean={o.float().mean().item():.4f}")

            ref = in_r[0].float() @ lb_r[0].float().T   # [M, N]
            diff = (o.float() - ref.cpu()).abs().max().item()
            log("\nValidation (vs inputs @ loraB^T):")
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
            sys.exit(1)
        finally:
            f.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    if os.environ.get("LEXPAND_CHILD") == "1":
        main(); sys.exit(0)
    lp = os.path.join(LOG_DIR, f"test_lora_expand_kernel_{datetime.now():%Y-%m-%d_%H-%M-%S}.log")
    os.makedirs(LOG_DIR, exist_ok=True)
    env = dict(os.environ, LEXPAND_CHILD="1", LEXPAND_LOG_PATH=lp)
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
