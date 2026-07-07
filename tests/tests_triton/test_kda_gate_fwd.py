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
KERNEL_NAME = "kda_gate_fwd_kernel"


def main(log_path):
    torch.manual_seed(42)
    with open(log_path, "w") as f:
        log = logger(f)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(ts)
        log(f"Kernel: {KERNEL_NAME}")
        log("")
        try:
            from vllm.model_executor.layers.fla.ops.kda import fused_kda_gate

            T, H, D = 128, 4, 64
            beta_sp = 1.0
            threshold = 20.0

            g = torch.randn(T, H * D, device=DEVICE, dtype=torch.float32)
            A = torch.randn(H, device=DEVICE, dtype=torch.float32)

            log("Inputs:")
            log(f"  g: {tuple(g.shape)}, dtype={g.dtype}, device={g.device}")
            log(f"  A: {tuple(A.shape)}, dtype={A.dtype}")
            log(f"  T={T}, H={H}, D={D}")
            log(f"  beta={beta_sp}, threshold={threshold}")
            log("")

            BT_approx = 64
            grid = ((T + BT_approx - 1) // BT_approx, H)
            log("Grid Configuration:")
            log(f"  grid: {grid}  [approx, BT=64]")
            log("")

            y = fused_kda_gate(g=g, A=A, head_k_dim=D, beta=beta_sp, threshold=threshold)

            log("Status: SUCCESS")
            log("")
            log("Output:")
            log(f"  y: {tuple(y.shape)}, dtype={y.dtype}, device={y.device}")
            log(f"  y[0,0,:4]: {y[0,0,:4].tolist()}")

        except Exception as e:
            log("Status: FAILURE")
            log("")
            log("Error:")
            log(traceback.format_exc())

        log("")
        log("-" * 36)
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "kda_gate_fwd")
