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
KERNEL_NAME = "fused_recurrent_gated_delta_rule_fwd_kernel"


def main(log_path):
    torch.manual_seed(42)
    with open(log_path, "w") as f:
        log = logger(f)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(ts)
        log(f"Kernel: {KERNEL_NAME}")
        log("")
        try:
            from vllm.model_executor.layers.fla.ops.fused_recurrent import (
                fused_recurrent_gated_delta_rule,
            )

            B, T, H, HV, K, V = 2, 16, 4, 4, 64, 64

            q = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.float16)
            k = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.float16)
            k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
            v = torch.randn(B, T, HV, V, device=DEVICE, dtype=torch.float16)
            g = torch.nn.functional.logsigmoid(
                torch.rand(B, T, HV, device=DEVICE, dtype=torch.float32)
            )
            beta = torch.sigmoid(torch.rand(B, T, HV, device=DEVICE, dtype=torch.float16))
            h0 = torch.zeros(B, HV, K, V, device=DEVICE, dtype=torch.float32)

            log("Inputs:")
            log(f"  q: {tuple(q.shape)}, dtype={q.dtype}, device={q.device}")
            log(f"  k: {tuple(k.shape)}, dtype={k.dtype}")
            log(f"  v: {tuple(v.shape)}, dtype={v.dtype}")
            log(f"  g: {tuple(g.shape)}, dtype={g.dtype}")
            log(f"  beta: {tuple(beta.shape)}, dtype={beta.dtype}")
            log(f"  h0: {tuple(h0.shape)}, dtype={h0.dtype}")
            log(f"  B={B}, T={T}, H={H}, HV={HV}, K={K}, V={V}")
            log("")

            BK = min(64, K)
            BV = min(32, V)
            NK = (K + BK - 1) // BK
            NV = (V + BV - 1) // BV
            grid = (NK, NV, B * HV)
            log("Grid Configuration:")
            log(f"  grid: {grid}")
            log(f"  BK={BK}, BV={BV}")
            log("")

            o, final_state = fused_recurrent_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta, initial_state=h0, inplace_final_state=False
            )

            log("Status: SUCCESS")
            log("")
            log("Output:")
            log(f"  o: {tuple(o.shape)}, dtype={o.dtype}, device={o.device}")
            log(f"  final_state: {tuple(final_state.shape)}, dtype={final_state.dtype}")
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
    run_test(main, LOG_DIR, "fused_recurrent_gated_delta_rule")
