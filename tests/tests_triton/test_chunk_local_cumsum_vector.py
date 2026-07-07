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
KERNEL_NAME = "chunk_local_cumsum_vector_kernel"


def main(log_path):
    torch.manual_seed(42)
    with open(log_path, "w") as f:
        log = logger(f)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(ts)
        log(f"Kernel: {KERNEL_NAME}")
        log("")
        try:
            from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum_vector

            B, T, H, S = 2, 128, 4, 64
            BT = 64

            g = torch.randn(B, T, H, S, device=DEVICE, dtype=torch.float32)

            log("Inputs:")
            log(f"  g: {tuple(g.shape)}, dtype={g.dtype}, device={g.device}")
            log(f"  B={B}, T={T}, H={H}, S={S}, chunk_size={BT}")
            log("")

            NT = T // BT
            BS_approx = 64
            grid = (S // BS_approx, NT, B * H)
            log("Grid Configuration:")
            log(f"  grid: {grid}  [approx, BS=64]")
            log(f"  BT={BT}")
            log("")

            o = chunk_local_cumsum_vector(g, chunk_size=BT)

            log("Status: SUCCESS")
            log("")
            log("Output:")
            log(f"  o: {tuple(o.shape)}, dtype={o.dtype}, device={o.device}")
            log(f"  o[0,0,0,:4]: {o[0,0,0,:4].tolist()}")

        except Exception as e:
            log("Status: FAILURE")
            log("")
            log("Error:")
            log(traceback.format_exc())

        log("")
        log("-" * 36)
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "chunk_local_cumsum_vector")
