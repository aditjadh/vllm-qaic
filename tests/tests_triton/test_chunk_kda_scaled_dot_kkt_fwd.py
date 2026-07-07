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
KERNEL_NAME = "chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter"


def main(log_path):
    torch.manual_seed(42)
    with open(log_path, "w") as f:
        log = logger(f)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(ts)
        log(f"Kernel: {KERNEL_NAME}")
        log("")
        try:
            from vllm.model_executor.layers.fla.ops.kda import chunk_kda_scaled_dot_kkt_fwd

            B, T, H, K = 2, 128, 4, 64
            BT = 64
            NT = T // BT
            BC = min(16, BT)
            NC = (BT + BC - 1) // BC

            q = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.float16)
            k = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.float16)
            gk = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.float32)
            beta = torch.sigmoid(torch.randn(B, T, H, device=DEVICE, dtype=torch.float32))

            log("Inputs:")
            log(f"  q: {tuple(q.shape)}, dtype={q.dtype}, device={q.device}")
            log(f"  k: {tuple(k.shape)}, dtype={k.dtype}")
            log(f"  gk: {tuple(gk.shape)}, dtype={gk.dtype}")
            log(f"  beta: {tuple(beta.shape)}, dtype={beta.dtype}")
            log(f"  B={B}, T={T}, H={H}, K={K}, BT={BT}, BC={BC}, NC={NC}")
            log("")

            grid_inter = (NT, NC * NC, B * H)
            grid_intra = (NT, NC, B * H)
            log("Grid Configuration:")
            log(f"  inter grid: {grid_inter}")
            log(f"  intra grid: {grid_intra}")
            log(f"  BC={BC}, NC={NC}")
            log("")

            A, Aqk = chunk_kda_scaled_dot_kkt_fwd(
                q=q, k=k, gk=gk, beta=beta, chunk_size=BT
            )

            log("Status: SUCCESS")
            log("")
            log("Output:")
            log(f"  A: {tuple(A.shape)}, dtype={A.dtype}, device={A.device}")
            log(f"  Aqk: {tuple(Aqk.shape)}, dtype={Aqk.dtype}")
            log(f"  A[0,0,0,:4]: {A[0,0,0,:4].tolist()}")
            log(f"  Aqk[0,0,0,:4]: {Aqk[0,0,0,:4].tolist()}")

        except Exception as e:
            log("Status: FAILURE")
            log("")
            log("Error:")
            log(traceback.format_exc())

        log("")
        log("-" * 36)
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "chunk_kda_scaled_dot_kkt_fwd")
