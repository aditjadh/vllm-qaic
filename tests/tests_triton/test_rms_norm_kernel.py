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
from _ktest import run_test, logger

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "_rms_norm_kernel"


def main(log_path):
    torch.manual_seed(42)
    with open(log_path, "w") as f:
        log = logger(f)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(ts)
        log(f"Kernel: {KERNEL_NAME}")
        log("")
        try:
            from vllm.model_executor.layers.batch_invariant import rms_norm

            M, N = 16, 256
            inp = torch.randn(M, N, dtype=torch.float16).to(device=DEVICE)
            weight = torch.ones(N, dtype=torch.float16).to(device=DEVICE)
            eps = 1e-6

            log("Inputs:")
            log(f"  input: {tuple(inp.shape)}, dtype={inp.dtype}, device={inp.device}")
            log(f"  weight: {tuple(weight.shape)}, dtype={weight.dtype}")
            log(f"  M={M}, N={N}, eps={eps}")
            log("")

            BLOCK_SIZE = 1024
            log("Grid Configuration:")
            log(f"  grid: ({M},)")
            log(f"  BLOCK_SIZE={BLOCK_SIZE}")
            log("")

            out = rms_norm(inp, weight, eps)

            log("Status: SUCCESS")
            log("")
            log("Output:")
            log(f"  out: {tuple(out.shape)}, dtype={out.dtype}, device={out.device}")
            log(f"  out[0, :4]: {out[0, :4].cpu().tolist()}")

        except Exception as e:
            log("Status: FAILURE")
            log("")
            log("Error:")
            log(traceback.format_exc())

        log("")
        log("-" * 36)
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "rms_norm_kernel")
