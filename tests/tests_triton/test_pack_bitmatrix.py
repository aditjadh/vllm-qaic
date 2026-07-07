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
KERNEL_NAME = "pack_bitmatrix"


def main(log_path):
    torch.manual_seed(42)
    with open(log_path, "w") as f:
        log = logger(f)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(ts)
        log(f"Kernel: {KERNEL_NAME}")
        log("")
        try:
            from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (
                make_routing_data,
            )

            # pack_bitmatrix is called inside make_routing_data
            n_rows = 8       # batch tokens
            num_topk = 2
            num_local_experts = 4

            topk_ids = torch.randint(0, num_local_experts, (n_rows, num_topk), dtype=torch.int64, device=DEVICE)
            topk_weights = torch.rand(n_rows, num_topk, dtype=torch.float32, device=DEVICE)
            # Normalize weights per row
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

            log("Inputs:")
            log(f"  topk_ids: {tuple(topk_ids.shape)}, dtype={topk_ids.dtype}, device={topk_ids.device}")
            log(f"  topk_weights: {tuple(topk_weights.shape)}, dtype={topk_weights.dtype}")
            log(f"  n_rows={n_rows}, num_topk={num_topk}, num_local_experts={num_local_experts}")
            log("")

            import triton
            BLOCK_SIZE_M = 512
            BLOCK_SIZE_K = 32
            bm_cols = triton.cdiv(num_local_experts, BLOCK_SIZE_K)
            grid = (triton.cdiv(n_rows, BLOCK_SIZE_M),)
            log("Grid Configuration:")
            log(f"  grid: {grid}")
            log(f"  BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_K={BLOCK_SIZE_K}")
            log(f"  bm_cols={bm_cols}")
            log("")

            routing_data, gather_indx, scatter_indx = make_routing_data(
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                num_local_experts=num_local_experts,
            )

            log("Status: SUCCESS")
            log("")
            log("Output:")
            log(f"  routing_data: {type(routing_data).__name__}")
            log(f"  gather_indx: {type(gather_indx).__name__}")
            log(f"  scatter_indx: {type(scatter_indx).__name__}")

        except Exception as e:
            log("Status: FAILURE")
            log("")
            log("Error:")
            log(traceback.format_exc())

        log("")
        log("-" * 36)
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "pack_bitmatrix")
