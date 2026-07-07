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

# input_guard uses torch.cuda.device which doesn't exist on QAIC
torch.cuda.device = lambda idx: contextlib.nullcontext()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vllm"))
from _ktest import run_test, logger

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "solve_tril_16x16_kernel"


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

            B, T, H = 2, 64, 4
            BT = 16  # 16x16 kernel

            # strictly lower triangular A
            A_raw = torch.randn(B, T, H, BT, device=DEVICE, dtype=torch.float32)
            A = torch.zeros_like(A_raw)
            for i in range(BT):
                for j in range(i):
                    A[:, :, :, j] = A_raw[:, :, :, j]  # below-diagonal only
            # zero out upper tri properly: each row i has entries 0..i-1
            mask = torch.tril(torch.ones(BT, BT), diagonal=-1)  # [BT,BT]
            A_2d = A_raw.reshape(B * T * H, BT)  # reuse shape
            # Proper strictly lower triangular per-chunk-row
            A = A_raw.clone()
            # zero columns >= row_index for each time step (row within BT)
            for row_i in range(BT):
                A[:, :, :, row_i] = A_raw[:, :, :, row_i]
                # keep only entries < row_i for the j-th column
            # simpler: store only strict-lower-tri of the BT-wide row vector
            # A[b, t, h, j] represents A[t, j] for j < t%BT
            # For solve_tril usage: each row t stores the t-th row of the chunk matrix
            # The kernel expects A[b,t,h,:] = row t of the BT×BT chunk matrix
            # We zero upper-triangle entries per row
            t_idx = (torch.arange(T, device=DEVICE) % BT)  # row index within chunk
            j_idx = torch.arange(BT, device=DEVICE)
            mask = (t_idx[:, None] > j_idx[None, :])  # [T, BT]
            A = A_raw * mask[None, :, None, :]  # broadcast over B, H

            log("Inputs:")
            log(f"  A: {tuple(A.shape)}, dtype={A.dtype}, device={A.device}")
            log(f"  B={B}, T={T}, H={H}, BT={BT}")
            log(f"  A strictly lower-triangular (per chunk)")
            log("")

            NT = T // BT
            grid = (NT * (BT // 16), B * H)
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
    run_test(main, LOG_DIR, "solve_tril_16x16")
