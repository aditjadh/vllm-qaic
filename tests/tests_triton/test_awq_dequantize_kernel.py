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
KERNEL_NAME = "awq_dequantize_kernel"


def main(log_path):
    torch.manual_seed(42)
    with open(log_path, "w") as f:
        log = logger(f)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(ts)
        log(f"Kernel: {KERNEL_NAME}")
        log("")
        try:
            from vllm.model_executor.layers.quantization.awq_triton import (
                awq_dequantize_triton,
            )

            # qweight: [K, M//8] int32, group_size=128
            K, M = 256, 256
            group_size = 128

            qweight = torch.randint(0, 2**31, (K, M // 8), dtype=torch.int32, device=DEVICE)
            scales = torch.randn(K // group_size, M, dtype=torch.float16, device=DEVICE)
            zeros = torch.randint(0, 2**31, (K // group_size, M // 8), dtype=torch.int32, device=DEVICE)

            block_size_x = 32
            block_size_y = 32

            log("Inputs:")
            log(f"  qweight: {tuple(qweight.shape)}, dtype={qweight.dtype}, device={qweight.device}")
            log(f"  scales: {tuple(scales.shape)}, dtype={scales.dtype}")
            log(f"  zeros: {tuple(zeros.shape)}, dtype={zeros.dtype}")
            log(f"  K={K}, M={M}, group_size={group_size}")
            log("")

            grid_x = (M // 8 + block_size_x - 1) // block_size_x
            grid_y = (K + block_size_y - 1) // block_size_y
            log("Grid Configuration:")
            log(f"  grid: ({grid_x}, {grid_y})")
            log(f"  BLOCK_SIZE_X={block_size_x}, BLOCK_SIZE_Y={block_size_y}")
            log("")

            result = awq_dequantize_triton(
                qweight=qweight,
                scales=scales,
                zeros=zeros,
                block_size_x=block_size_x,
                block_size_y=block_size_y,
            )

            log("Status: SUCCESS")
            log("")
            log("Output:")
            log(f"  result: {tuple(result.shape)}, dtype={result.dtype}, device={result.device}")
            log(f"  result[0, :4]: {result[0, :4].tolist()}")

        except Exception as e:
            log("Status: FAILURE")
            log("")
            log("Error:")
            log(traceback.format_exc())

        log("")
        log("-" * 36)
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "awq_dequantize_kernel")
