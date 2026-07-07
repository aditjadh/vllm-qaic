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
KERNEL_NAME = "fused_gdn_gating_kernel"


def main(log_path):
    torch.manual_seed(42)
    with open(log_path, "w") as f:
        log = logger(f)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(ts)
        log(f"Kernel: {KERNEL_NAME}")
        log("")
        try:
            # Import the kernel wrapper from qwen3_next
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vllm"))
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "qwen3_next",
                os.path.join(os.path.dirname(__file__), "..",
                             "vllm", "vllm", "model_executor", "models", "qwen3_next.py"),
            )
            # We directly import the fused_gdn_gating function via the ops
            from vllm.triton_utils import tl, triton

            # Re-implement minimal wrapper to avoid full model import
            # The kernel signature: (g, beta_output, A_log, a, b, dt_bias, seq_len, NUM_HEADS, beta, threshold, BLK_HEADS)
            import math

            num_heads = 8
            batch = 4  # sequences
            seq_len = 1

            A_log = torch.randn(num_heads, device=DEVICE, dtype=torch.float32)
            a = torch.randn(batch, num_heads, device=DEVICE, dtype=torch.float32)
            b = torch.randn(batch, num_heads, device=DEVICE, dtype=torch.float32)
            dt_bias = torch.randn(num_heads, device=DEVICE, dtype=torch.float32)

            log("Inputs:")
            log(f"  A_log: {tuple(A_log.shape)}, dtype={A_log.dtype}, device={A_log.device}")
            log(f"  a: {tuple(a.shape)}, dtype={a.dtype}")
            log(f"  b: {tuple(b.shape)}, dtype={b.dtype}")
            log(f"  dt_bias: {tuple(dt_bias.shape)}, dtype={dt_bias.dtype}")
            log(f"  batch={batch}, num_heads={num_heads}, seq_len={seq_len}")
            log("")

            BLK_HEADS = 8
            grid = (batch, seq_len, (num_heads + BLK_HEADS - 1) // BLK_HEADS)
            log("Grid Configuration:")
            log(f"  grid: {grid}")
            log(f"  BLK_HEADS={BLK_HEADS}")
            log("")

            # Import the actual kernel function to test
            import importlib
            import importlib.util as ilu

            # Load gdn_gating directly from triton kernel in qwen3_next
            from vllm.model_executor.models.qwen3_next import fused_gdn_gating

            g_out, beta_out = fused_gdn_gating(
                A_log=A_log, a=a, b=b, dt_bias=dt_bias
            )

            log("Status: SUCCESS")
            log("")
            log("Output:")
            log(f"  g: {tuple(g_out.shape)}, dtype={g_out.dtype}, device={g_out.device}")
            log(f"  beta: {tuple(beta_out.shape)}, dtype={beta_out.dtype}")
            log(f"  g[0,0,:4]: {g_out[0,0,:4].tolist()}")
            log(f"  beta[0,0,:4]: {beta_out[0,0,:4].tolist()}")

        except Exception as e:
            log("Status: FAILURE")
            log("")
            log("Error:")
            log(traceback.format_exc())

        log("")
        log("-" * 36)
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "fused_gdn_gating")
