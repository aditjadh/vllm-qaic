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
KERNEL_NAME = "chunk_kda_fwd (full pipeline)"


def main(log_path):
    torch.manual_seed(42)
    with open(log_path, "w") as f:
        log = logger(f)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(ts)
        log(f"Kernel: {KERNEL_NAME}")
        log("")
        try:
            from vllm.model_executor.layers.fla.ops.kda import chunk_kda

            B, T, H, K, V = 2, 128, 4, 64, 64

            q = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.float16)
            k = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.float16)
            k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
            v = torch.randn(B, T, H, V, device=DEVICE, dtype=torch.float16)
            g = -torch.nn.functional.softplus(
                torch.randn(B, T, H, device=DEVICE, dtype=torch.float32)
            )
            beta = torch.sigmoid(torch.randn(B, T, H, device=DEVICE, dtype=torch.float32))
            h0 = torch.zeros(B, H, K, V, device=DEVICE, dtype=torch.float32)

            log("Inputs:")
            log(f"  q: {tuple(q.shape)}, dtype={q.dtype}, device={q.device}")
            log(f"  k: {tuple(k.shape)}, dtype={k.dtype}")
            log(f"  v: {tuple(v.shape)}, dtype={v.dtype}")
            log(f"  g: {tuple(g.shape)}, dtype={g.dtype}")
            log(f"  beta: {tuple(beta.shape)}, dtype={beta.dtype}")
            log(f"  h0: {tuple(h0.shape)}, dtype={h0.dtype}")
            log(f"  B={B}, T={T}, H={H}, K={K}, V={V}")
            log("")

            log("Grid Configuration:")
            log("  Multiple kernels (pipeline): chunk_local_cumsum, chunk_kda_scaled_dot_kkt,")
            log("  solve_tril, recompute_w_u, chunk_gated_delta_rule_fwd_h, chunk_gla_fwd_o_gk")
            log("")

            o, final_state = chunk_kda(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=h0, output_final_state=True,
                use_qk_l2norm_in_kernel=False,
            )

            log("Status: SUCCESS")
            log("")
            log("Output:")
            log(f"  o: {tuple(o.shape)}, dtype={o.dtype}, device={o.device}")
            log(f"  final_state: {tuple(final_state.shape) if final_state is not None else None}")
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
    run_test(main, LOG_DIR, "chunk_kda_fwd")
