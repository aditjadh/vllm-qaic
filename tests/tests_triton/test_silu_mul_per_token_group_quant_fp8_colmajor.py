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
KERNEL_NAME = "_silu_mul_per_token_group_quant_fp8_colmajor"


def main(log_path):
    torch.manual_seed(42)
    with open(log_path, "w") as f:
        log = logger(f)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(ts)
        log(f"Kernel: {KERNEL_NAME}")
        log("")
        try:
            from vllm.model_executor.layers.quantization.utils.fp8_utils import (
                silu_mul_per_token_group_quant_fp8_colmajor,
            )

            # M must be divisible by BLOCK_M=8 AND GROUP_SIZE=128
            # N must be divisible by 256 (N_2=N//2 must be divisible by GROUP_SIZE=128)
            M, N = 128, 512

            inp = torch.randn(M, N, dtype=torch.bfloat16).to(device=DEVICE)

            log("Inputs:")
            log(f"  input: {tuple(inp.shape)}, dtype={inp.dtype}, device={inp.device}")
            log(f"  M={M}, N={N}")
            log(f"  GROUP_SIZE=128, BLOCK_M=8, BLOCK_N=128")
            log(f"  (M must be divisible by GROUP_SIZE=128 and BLOCK_M=8)")
            log("")

            N_2 = N // 2
            BLOCK_M, BLOCK_N = 8, 128
            grid = (M // BLOCK_M, N_2 // BLOCK_N)
            log("Grid Configuration:")
            log(f"  grid: {grid}")
            log(f"  BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}")
            log("")

            x_q, x_s = silu_mul_per_token_group_quant_fp8_colmajor(inp)

            log("Status: SUCCESS")
            log("")
            log("Output:")
            log(f"  x_q: {tuple(x_q.shape)}, dtype={x_q.dtype}, device={x_q.device}")
            log(f"  x_s: {tuple(x_s.shape)}, dtype={x_s.dtype}")
            log(f"  x_s[0, :4]: {x_s[0, :4].tolist()}")

        except Exception as e:
            log("Status: FAILURE")
            log("")
            log("Error:")
            log(traceback.format_exc())

        log("")
        log("-" * 36)
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "silu_mul_per_token_group_quant_fp8_colmajor")
