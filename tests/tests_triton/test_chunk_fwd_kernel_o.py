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
KERNEL_NAME = "chunk_fwd_kernel_o"


def main(log_path):
    torch.manual_seed(42)
    with open(log_path, "w") as f:
        log = logger(f)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(ts)
        log(f"Kernel: {KERNEL_NAME}")
        log("")
        try:
            from vllm.model_executor.layers.fla.ops.chunk_o import chunk_fwd_o

            B, T, Hg, K, H, V = 2, 128, 4, 64, 4, 64
            BT = 64
            NT = T // BT

            q = torch.randn(B, T, Hg, K, device=DEVICE, dtype=torch.float16)
            k = torch.randn(B, T, Hg, K, device=DEVICE, dtype=torch.float16)
            v = torch.randn(B, T, H, V, device=DEVICE, dtype=torch.float16)
            h = torch.randn(B, NT, H, K, V, device=DEVICE, dtype=torch.float16)
            g = torch.randn(B, T, H, device=DEVICE, dtype=torch.float32)

            log("Inputs:")
            log(f"  q: {tuple(q.shape)}, dtype={q.dtype}, device={q.device}")
            log(f"  k: {tuple(k.shape)}, dtype={k.dtype}")
            log(f"  v: {tuple(v.shape)}, dtype={v.dtype}")
            log(f"  h: {tuple(h.shape)}, dtype={h.dtype}")
            log(f"  g: {tuple(g.shape)}, dtype={g.dtype}")
            log(f"  B={B}, T={T}, Hg={Hg}, H={H}, K={K}, V={V}, BT={BT}")
            log("")

            BV_approx = 64
            grid = (V // BV_approx, NT, B * H)
            log("Grid Configuration:")
            log(f"  grid: {grid}  [approx, BV=64]")
            log("")

            o = chunk_fwd_o(q=q, k=k, v=v, h=h, g=g, chunk_size=BT)

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
    run_test(main, LOG_DIR, "chunk_fwd_kernel_o")
