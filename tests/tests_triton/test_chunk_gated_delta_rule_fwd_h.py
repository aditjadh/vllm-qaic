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
KERNEL_NAME = "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"


def main(log_path):
    torch.manual_seed(42)
    with open(log_path, "w") as f:
        log = logger(f)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(ts)
        log(f"Kernel: {KERNEL_NAME}")
        log("")
        try:
            from vllm.model_executor.layers.fla.ops.chunk_delta_h import (
                chunk_gated_delta_rule_fwd_h,
            )

            B, T, Hg, K, V = 2, 128, 4, 64, 64
            H = Hg
            BT = 64

            k = torch.randn(B, T, Hg, K, device=DEVICE, dtype=torch.float16)
            w = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.float16)
            u = torch.randn(B, T, H, V, device=DEVICE, dtype=torch.float16)
            g = torch.randn(B, T, H, device=DEVICE, dtype=torch.float32)
            h0 = torch.zeros(B, H, K, V, device=DEVICE, dtype=torch.float32)

            log("Inputs:")
            log(f"  k: {tuple(k.shape)}, dtype={k.dtype}, device={k.device}")
            log(f"  w: {tuple(w.shape)}, dtype={w.dtype}, device={w.device}")
            log(f"  u: {tuple(u.shape)}, dtype={u.dtype}, device={u.device}")
            log(f"  g: {tuple(g.shape)}, dtype={g.dtype}, device={g.device}")
            log(f"  h0: {tuple(h0.shape)}, dtype={h0.dtype}, device={h0.device}")
            log(f"  B={B}, T={T}, Hg={Hg}, H={H}, K={K}, V={V}, BT={BT}")
            log("")

            NT = T // BT
            grid = (V // 32, B * H)
            log("Grid Configuration:")
            log(f"  grid: {grid}  [approx, BV=32]")
            log(f"  NT (chunk steps): {NT}")
            log("")

            h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
                k=k,
                w=w,
                u=u,
                g=g,
                initial_state=h0,
                output_final_state=True,
                chunk_size=BT,
                save_new_value=True,
            )

            log("Status: SUCCESS")
            log("")
            log("Output:")
            log(f"  h: {tuple(h.shape)}, dtype={h.dtype}, device={h.device}")
            log(f"  v_new: {tuple(v_new.shape)}, dtype={v_new.dtype}, device={v_new.device}")
            log(f"  final_state: {tuple(final_state.shape)}, dtype={final_state.dtype}")
            log(f"  h[:2,0,0,0,:4]: {h[:2,0,0,0,:4].tolist()}")

        except Exception as e:
            log("Status: FAILURE")
            log("")
            log("Error:")
            log(traceback.format_exc())

        log("")
        log("-" * 36)
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "chunk_gated_delta_rule_fwd_h")
