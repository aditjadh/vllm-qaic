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
KERNEL_NAME = "round_int8"


def main(log_path):
    torch.manual_seed(42)
    with open(log_path, "w") as f:
        log = logger(f)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(ts)
        log(f"Kernel: {KERNEL_NAME}")
        log("")
        try:
            # round_int8 is a helper called inside _per_token_quant_int8.
            # We exercise it by running per_token_quant_int8 which invokes it.
            from vllm.model_executor.layers.quantization.utils.int8_utils import (
                per_token_quant_int8,
            )

            M, N = 4, 128

            x = torch.randn(M, N, dtype=torch.float32, device=DEVICE)

            log("Inputs:")
            log(f"  x: {tuple(x.shape)}, dtype={x.dtype}, device={x.device}")
            log(f"  M={M}, N={N}")
            log(f"  (round_int8 is exercised as part of _per_token_quant_int8)")
            log("")

            BLOCK = 128
            log("Grid Configuration:")
            log(f"  grid: ({M},)")
            log(f"  BLOCK={BLOCK}")
            log("")

            x_q, scales = per_token_quant_int8(x)

            # Spot-check: all outputs must be in int8 range
            assert x_q.dtype == torch.int8
            assert x_q.cpu().abs().max().item() <= 127

            log("Status: SUCCESS")
            log("")
            log("Output:")
            log(f"  x_q: {tuple(x_q.shape)}, dtype={x_q.dtype}, device={x_q.device}")
            log(f"  scales: {tuple(scales.shape)}, dtype={scales.dtype}")
            log(f"  x_q[0, :8]: {x_q[0, :8].cpu().tolist()}")
            log(f"  scales[:4, 0]: {scales[:4, 0].cpu().tolist()}")

        except Exception as e:
            log("Status: FAILURE")
            log("")
            log("Error:")
            log(traceback.format_exc())

        log("")
        log("-" * 36)
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "round_int8")
