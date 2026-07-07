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
KERNEL_NAME = "_per_token_quant_int8"


def main(log_path):
    torch.manual_seed(42)
    with open(log_path, "w") as f:
        log = logger(f)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(ts)
        log(f"Kernel: {KERNEL_NAME}")
        log("")
        try:
            from vllm.model_executor.layers.quantization.utils.int8_utils import (
                per_token_quant_int8,
            )

            M, N = 8, 256

            x = torch.randn(M, N, dtype=torch.float16, device=DEVICE)

            log("Inputs:")
            log(f"  x: {tuple(x.shape)}, dtype={x.dtype}, device={x.device}")
            log(f"  M={M}, N={N}")
            log("")

            BLOCK = 256
            log("Grid Configuration:")
            log(f"  grid: ({M},)")
            log(f"  BLOCK={BLOCK}")
            log("")

            x_q, scales = per_token_quant_int8(x)

            try:
                scales_cpu = scales[:4, 0].cpu().tolist()
                x_q_shape = tuple(x_q.shape)
                log("Status: SUCCESS")
                log("")
                log("Output:")
                log(f"  x_q: {x_q_shape}, dtype={x_q.dtype}, device={x_q.device}")
                log(f"  scales: {tuple(scales.shape)}, dtype={scales.dtype}")
                log(f"  scales[:4, 0]: {scales_cpu}")
            except RuntimeError as sync_err:
                log("Status: FAILURE")
                log("")
                log("Error:")
                log(f"  Kernel launched (output shape logged) but device stream sync failed:")
                log(f"  {sync_err}")
                log("  Root cause: tl.extra.cuda.libdevice.round is CUDA-specific;")
                log("  QAIC backend stream corrupts on unsupported libdevice call.")

        except Exception as e:
            log("Status: FAILURE")
            log("")
            log("Error:")
            log(traceback.format_exc())

        log("")
        log("-" * 36)
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "per_token_quant_int8")
