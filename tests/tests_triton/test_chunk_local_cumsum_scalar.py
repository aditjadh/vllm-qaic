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
KERNEL_NAME = "chunk_local_cumsum_scalar_kernel"


def main(log_path):
    torch.manual_seed(42)
    with open(log_path, "w") as f:
        log = logger(f)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(ts)
        log(f"Kernel: {KERNEL_NAME}")
        log("")
        try:
            from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum_scalar

            B, T, H = 2, 128, 4
            BT = 64

            g = torch.randn(B, T, H, device=DEVICE, dtype=torch.float32)

            log("Inputs:")
            log(f"  g: {tuple(g.shape)}, dtype={g.dtype}, device={g.device}")
            log(f"  B={B}, T={T}, H={H}, chunk_size={BT}")
            log("")

            NT = T // BT
            grid = (NT, B * H)
            log("Grid Configuration:")
            log(f"  grid: {grid}")
            log(f"  BT={BT}")
            log("")

            o = chunk_local_cumsum_scalar(g, chunk_size=BT)

            log("Status: SUCCESS")
            log("")
            log("Output:")
            log(f"  o: {tuple(o.shape)}, dtype={o.dtype}, device={o.device}")
            log(f"  o[0,0,0]: {o[0,0,0].item():.6f}")
            log(f"  o[0,:4,0]: {o[0,:4,0].tolist()}")

        except Exception as e:
            log("Status: FAILURE")
            log("")
            log("Error:")
            log(traceback.format_exc())

        log("")
        log("-" * 36)
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "chunk_local_cumsum_scalar")
