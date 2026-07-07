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
KERNEL_NAME = "recompute_w_u_fwd_kernel"


def main(log_path):
    torch.manual_seed(42)
    with open(log_path, "w") as f:
        log = logger(f)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(ts)
        log(f"Kernel: {KERNEL_NAME}")
        log("")
        try:
            from vllm.model_executor.layers.fla.ops.kda import recompute_w_u_fwd

            B, T, H, K, V = 2, 128, 4, 64, 64
            BT = 64
            NT = T // BT

            k = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.float16)
            v = torch.randn(B, T, H, V, device=DEVICE, dtype=torch.float16)
            beta = torch.sigmoid(torch.randn(B, T, H, device=DEVICE, dtype=torch.float32))
            A = torch.randn(B, T, H, BT, device=DEVICE, dtype=torch.float16)
            gk = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.float32)

            log("Inputs:")
            log(f"  k: {tuple(k.shape)}, dtype={k.dtype}, device={k.device}")
            log(f"  v: {tuple(v.shape)}, dtype={v.dtype}")
            log(f"  beta: {tuple(beta.shape)}, dtype={beta.dtype}")
            log(f"  A: {tuple(A.shape)}, dtype={A.dtype}")
            log(f"  gk: {tuple(gk.shape)}, dtype={gk.dtype}")
            log(f"  B={B}, T={T}, H={H}, K={K}, V={V}, BT={BT}")
            log("")

            grid = (NT, B * H)
            log("Grid Configuration:")
            log(f"  grid: {grid}")
            log(f"  BK=64, BV=64")
            log("")

            w, u, _, kg = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, gk=gk)

            log("Status: SUCCESS")
            log("")
            log("Output:")
            log(f"  w: {tuple(w.shape)}, dtype={w.dtype}, device={w.device}")
            log(f"  u: {tuple(u.shape)}, dtype={u.dtype}")
            log(f"  kg: {tuple(kg.shape) if kg is not None else None}, dtype={kg.dtype if kg is not None else None}")
            log(f"  w[0,0,0,:4]: {w[0,0,0,:4].tolist()}")

        except Exception as e:
            log("Status: FAILURE")
            log("")
            log("Error:")
            log(traceback.format_exc())

        log("")
        log("-" * 36)
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "recompute_w_u_fwd")
