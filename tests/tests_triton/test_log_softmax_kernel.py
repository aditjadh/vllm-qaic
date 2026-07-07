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

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "verified_kernels")
DEVICE = "qaic"
KERNEL_NAME = "_log_softmax_kernel"


def main(log_path):
    torch.manual_seed(42)
    with open(log_path, "w") as f:
        log = logger(f)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(ts)
        log(f"Kernel: {KERNEL_NAME}")
        log("")
        try:
            from vllm.model_executor.layers.batch_invariant import (
                _log_softmax_kernel,
            )

            n_rows, n_cols = 4, 512
            BLOCK_SIZE = 1024

            x = torch.randn(n_rows, n_cols, dtype=torch.float32).to(device=DEVICE)
            output = torch.empty_like(x)

            log("Inputs:")
            log(f"  x: {tuple(x.shape)}, dtype={x.dtype}, device={x.device}")
            log(f"  n_rows={n_rows}, n_cols={n_cols}")
            log("")

            grid = (n_rows,)
            log("Grid Configuration:")
            log(f"  grid: {grid}")
            log(f"  BLOCK_SIZE: {BLOCK_SIZE}")
            log("")

            _log_softmax_kernel[grid](
                x,
                output,
                x.stride(0),
                output.stride(0),
                n_cols,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            ref = torch.nn.functional.log_softmax(x.cpu(), dim=-1).to(device=DEVICE)
            max_diff = (output - ref).abs().max().item()

            log("Status: SUCCESS")
            log("")
            log("Output:")
            log(f"  output shape: {list(output.shape)}")
            log(f"  output[0, :8]: {output[0, :8].tolist()}")
            log(f"  max diff vs log_softmax reference: {max_diff:.6e}")
            log("")
            log("------------------------------------")

        except Exception:
            log("Status: FAILURE")
            log("")
            log("Error:")
            log(traceback.format_exc())
            log("------------------------------------")
            raise


if __name__ == "__main__":
    run_test(main, LOG_DIR, "log_softmax_kernel")
