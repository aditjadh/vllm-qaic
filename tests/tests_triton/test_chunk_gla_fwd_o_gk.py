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
KERNEL_NAME = "chunk_gla_fwd_kernel_o"


def main(log_path):
    torch.manual_seed(42)
    with open(log_path, "w") as f:
        log = logger(f)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(ts)
        log(f"Kernel: {KERNEL_NAME}")
        log("")
        try:
            from vllm.model_executor.layers.fla.ops.kda import chunk_gla_fwd_o_gk

            B, T, H, K, V = 2, 128, 4, 64, 64
            BT = 64
            NT = T // BT

            q = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.float16)
            v = torch.randn(B, T, H, V, device=DEVICE, dtype=torch.float16)
            g = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.float32)
            A = torch.zeros(B, T, H, BT, device=DEVICE, dtype=torch.float32)
            h = torch.randn(B, NT, H, K, V, device=DEVICE, dtype=torch.float16)
            o = torch.empty(B, T, H, V, device=DEVICE, dtype=torch.float16)
            scale = K ** -0.5

            log("Inputs:")
            log(f"  q: {tuple(q.shape)}, dtype={q.dtype}, device={q.device}")
            log(f"  v: {tuple(v.shape)}, dtype={v.dtype}")
            log(f"  g: {tuple(g.shape)}, dtype={g.dtype}")
            log(f"  A: {tuple(A.shape)}, dtype={A.dtype}")
            log(f"  h: {tuple(h.shape)}, dtype={h.dtype}")
            log(f"  B={B}, T={T}, H={H}, K={K}, V={V}, BT={BT}, scale={scale:.6f}")
            log("")

            BV_approx = 64
            grid = (V // BV_approx, NT, B * H)
            log("Grid Configuration:")
            log(f"  grid: {grid}  [approx, BV=64]")
            log("")

            o_out = chunk_gla_fwd_o_gk(
                q=q, v=v, g=g, A=A, h=h, o=o, scale=scale, chunk_size=BT
            )

            log("Status: SUCCESS")
            log("")
            log("Output:")
            log(f"  o: {tuple(o_out.shape)}, dtype={o_out.dtype}, device={o_out.device}")
            log(f"  o[0,0,0,:4]: {o_out[0,0,0,:4].tolist()}")

        except Exception as e:
            log("Status: FAILURE")
            log("")
            log("Error:")
            log(traceback.format_exc())

        log("")
        log("-" * 36)
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "chunk_gla_fwd_o_gk")
