# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
import os
import sys
import traceback
from datetime import datetime

import contextlib
import torch

torch.cuda.device = lambda idx: contextlib.nullcontext()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vllm"))
from _ktest import run_test, logger

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "merge_16x16_to_64x64_inverse_kernel"


def main(log_path):
    torch.manual_seed(42)
    with open(log_path, "w") as f:
        log = logger(f)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(ts)
        log(f"Kernel: {KERNEL_NAME}")
        log("")
        try:
            from vllm.model_executor.layers.fla.ops.solve_tril import solve_tril

            B, T, H = 2, 128, 4
            BT = 64  # triggers merge_16x16_to_64x64_inverse_kernel

            t_idx = (torch.arange(T, device=DEVICE) % BT)
            j_idx = torch.arange(BT, device=DEVICE)
            mask = (t_idx[:, None] > j_idx[None, :])
            A_raw = torch.randn(B, T, H, BT, device=DEVICE, dtype=torch.float32)
            A = A_raw * mask[None, :, None, :]

            log("Inputs:")
            log(f"  A: {tuple(A.shape)}, dtype={A.dtype}, device={A.device}")
            log(f"  B={B}, T={T}, H={H}, BT={BT}")
            log(f"  A strictly lower-triangular (per chunk)")
            log("")

            NT = T // BT
            grid = (NT, B * H)
            log("Grid Configuration:")
            log(f"  grid: {grid}")
            log(f"  BT={BT}")
            log("")

            Ai = solve_tril(A=A, output_dtype=torch.float32)

            log("Status: SUCCESS")
            log("")
            log("Output:")
            log(f"  Ai: {tuple(Ai.shape)}, dtype={Ai.dtype}, device={Ai.device}")
            log(f"  Ai[0,0,0,:4]: {Ai[0,0,0,:4].tolist()}")

        except Exception as e:
            log("Status: FAILURE")
            log("")
            log("Error:")
            log(traceback.format_exc())

        log("")
        log("-" * 36)
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "solve_tril_64x64")
