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
KERNEL_NAME = "chunk_scaled_dot_kkt_fwd_kernel"


def main(log_path):
    torch.manual_seed(42)
    with open(log_path, "w") as f:
        log = logger(f)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(ts)
        log(f"Kernel: {KERNEL_NAME}")
        log("")
        try:
            from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import (
                chunk_scaled_dot_kkt_fwd,
            )

            B, T, H, K = 2, 128, 4, 64
            BT = 64
            NT = T // BT

            k = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.float16)
            beta = torch.sigmoid(torch.randn(B, T, H, device=DEVICE, dtype=torch.float32))
            g = torch.randn(B, T, H, device=DEVICE, dtype=torch.float32)

            log("Inputs:")
            log(f"  k: {tuple(k.shape)}, dtype={k.dtype}, device={k.device}")
            log(f"  beta: {tuple(beta.shape)}, dtype={beta.dtype}")
            log(f"  g: {tuple(g.shape)}, dtype={g.dtype}")
            log(f"  B={B}, T={T}, H={H}, K={K}, BT={BT}")
            log("")

            grid = (NT, B * H)
            log("Grid Configuration:")
            log(f"  grid: {grid}")
            log(f"  BT={BT}")
            log("")

            A = chunk_scaled_dot_kkt_fwd(k=k, g=g, beta=beta, chunk_size=BT)

            log("Status: SUCCESS")
            log("")
            log("Output:")
            log(f"  A: {tuple(A.shape)}, dtype={A.dtype}, device={A.device}")
            log(f"  A[0,0,0,:4]: {A[0,0,0,:4].tolist()}")

        except Exception as e:
            log("Status: FAILURE")
            log("")
            log("Error:")
            log(traceback.format_exc())

        log("")
        log("-" * 36)
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "chunk_scaled_dot_kkt_fwd")
