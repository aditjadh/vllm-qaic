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
KERNEL_NAME = "triton_scale_swizzle"


def main(log_path):
    torch.manual_seed(42)
    with open(log_path, "w") as f:
        log = logger(f)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(ts)
        log(f"Kernel: {KERNEL_NAME}")
        log("")
        try:
            from vllm.model_executor.layers.quantization.qutlass_utils import (
                triton_mx_block_rearrange,
            )

            # triton_scale_swizzle is the @triton.jit kernel called by triton_mx_block_rearrange.
            # Dimensions must be multiples of 128 (rows) and 4 (cols).
            rows, cols = 128, 4

            # uint8 / int8 element-size tensors
            scale_tensor = torch.randint(
                0, 255, (rows, cols), dtype=torch.uint8, device=DEVICE
            ).view(torch.int8)

            log("Inputs:")
            log(f"  scale_tensor: {tuple(scale_tensor.shape)}, dtype={scale_tensor.dtype}, device={scale_tensor.device}")
            log(f"  rows={rows}, cols={cols}")
            log("")

            BLOCK_ROWS, BLOCK_COLS = 128, 4
            n_row_blocks = (rows + BLOCK_ROWS - 1) // BLOCK_ROWS
            n_col_blocks = (cols + BLOCK_COLS - 1) // BLOCK_COLS
            log("Grid Configuration:")
            log(f"  grid: ({n_row_blocks}, {n_col_blocks})")
            log(f"  BLOCK_ROWS={BLOCK_ROWS}, BLOCK_COLS={BLOCK_COLS}")
            log("")

            out = triton_mx_block_rearrange(scale_tensor)

            log("Status: SUCCESS")
            log("")
            log("Output:")
            log(f"  out: {tuple(out.shape)}, dtype={out.dtype}, device={out.device}")
            log(f"  out.flatten()[:8]: {out.flatten()[:8].tolist()}")

        except Exception as e:
            log("Status: FAILURE")
            log("")
            log("Error:")
            log(traceback.format_exc())

        log("")
        log("-" * 36)
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "triton_scale_swizzle")
