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
KERNEL_NAME = "compute_identity_kernel"


def main(log_path):
    torch.manual_seed(42)
    with open(log_path, "w") as f:
        log = logger(f)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(ts)
        log(f"Kernel: {KERNEL_NAME}")
        log("")
        try:
            from vllm.model_executor.layers.fused_moe.fused_moe import (
                zero_experts_compute_triton,
            )

            # compute_identity_kernel is called inside zero_experts_compute_triton
            # when zero_expert_type == "identity"
            num_tokens = 4
            hidden_dim = 256
            num_experts = 2
            top_k = 2

            hidden_states = torch.randn(num_tokens, hidden_dim, dtype=torch.float16, device=DEVICE)
            # expert_indices: some indices valid (< num_experts), some out-of-range
            expert_indices = torch.tensor(
                [[0, 1], [0, 2], [1, 3], [2, 3]], dtype=torch.int64, device=DEVICE
            )
            expert_scales = torch.ones(num_tokens, top_k, dtype=torch.float16, device=DEVICE)

            log("Inputs:")
            log(f"  hidden_states: {tuple(hidden_states.shape)}, dtype={hidden_states.dtype}, device={hidden_states.device}")
            log(f"  expert_indices: {tuple(expert_indices.shape)}, dtype={expert_indices.dtype}")
            log(f"  expert_scales: {tuple(expert_scales.shape)}, dtype={expert_scales.dtype}")
            log(f"  num_experts={num_experts}, top_k={top_k}")
            log("")

            BLOCK_SIZE = 256
            grid_size = num_tokens * (hidden_dim // BLOCK_SIZE)
            log("Grid Configuration:")
            log(f"  grid: ({grid_size},)")
            log(f"  BLOCK_SIZE={BLOCK_SIZE}")
            log("")

            output = zero_experts_compute_triton(
                expert_indices=expert_indices,
                expert_scales=expert_scales,
                num_experts=num_experts,
                zero_expert_type="identity",
                hidden_states=hidden_states,
            )

            log("Status: SUCCESS")
            log("")
            log("Output:")
            log(f"  output: {tuple(output.shape)}, dtype={output.dtype}, device={output.device}")
            log(f"  output[0, :4]: {output[0, :4].tolist()}")

        except Exception as e:
            log("Status: FAILURE")
            log("")
            log("Error:")
            log(traceback.format_exc())

        log("")
        log("-" * 36)
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "compute_identity_kernel")
