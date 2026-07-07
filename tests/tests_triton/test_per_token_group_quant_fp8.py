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
KERNEL_NAME = "_per_token_group_quant_fp8"


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
                per_token_group_quant_fp8,
            )

            M, N = 8, 256
            group_size = 128

            x = torch.randn(M, N, dtype=torch.bfloat16).to(device=DEVICE)

            log("Inputs:")
            log(f"  x: {tuple(x.shape)}, dtype={x.dtype}, device={x.device}")
            log(f"  M={M}, N={N}, group_size={group_size}")
            log(f"  column_major_scales=False, use_ue8m0=False")
            log("")

            num_groups = (M * N) // group_size
            BLOCK = 128
            log("Grid Configuration:")
            log(f"  grid: ({num_groups},)")
            log(f"  BLOCK={BLOCK}")
            log("")

            x_q, x_s = per_token_group_quant_fp8(
                x, group_size, column_major_scales=False, use_ue8m0=False
            )

            log("Status: SUCCESS")
            log("")
            log("Output:")
            log(f"  x_q: {tuple(x_q.shape)}, dtype={x_q.dtype}, device={x_q.device}")
            log(f"  x_s: {tuple(x_s.shape)}, dtype={x_s.dtype}")
            log(f"  x_s[0, :4]: {x_s[0, :4].cpu().tolist()}")

        except Exception as e:
            log("Status: FAILURE")
            log("")
            log("Error:")
            log(traceback.format_exc())

        log("")
        log("-" * 36)
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "per_token_group_quant_fp8")
