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

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "_causal_conv1d_update_kernel"


def main(log_path):
    torch.manual_seed(42)
    batch = 2
    dim = 16
    width = 4
    seqlen = 1   # single-token update (decode step)
    state_len = width - 1  # = 3

    # x: (batch, dim, seqlen)
    x = torch.randn(batch, dim, seqlen, dtype=torch.float32, device=DEVICE)
    # x must have stride(1)==1:  create with dim as innermost, then permute
    x = torch.randn(batch, seqlen, dim, dtype=torch.float32, device=DEVICE).permute(0, 2, 1)
    # shape (batch, dim, seqlen), stride(1)=1

    weight = torch.randn(dim, width, dtype=torch.float32, device=DEVICE)
    # weight.stride(1)==1 required
    weight = torch.randn(width, dim, dtype=torch.float32, device=DEVICE).permute(1, 0).contiguous()
    # shape (dim, width), stride(1)=1

    bias = torch.randn(dim, dtype=torch.float32, device=DEVICE)

    # conv_state: (num_cache_lines, dim, state_len) with stride(-2)==1
    # Use (num_cache_lines, state_len, dim).permute(0,2,1) → stride(1)=1
    num_cache_lines = batch
    conv_state = torch.zeros(
        num_cache_lines, state_len, dim, dtype=torch.float32, device=DEVICE
    ).permute(0, 2, 1)  # shape (num_cache_lines, dim, state_len), stride(1)=1

    conv_state_indices = torch.arange(batch, dtype=torch.int32, device=DEVICE)

    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  x: {tuple(x.shape)} [batch, dim, seqlen], stride={tuple(x.stride())}, device={x.device}")
            log(f"  weight: {tuple(weight.shape)}, width={width}, stride={tuple(weight.stride())}")
            log(f"  conv_state: {tuple(conv_state.shape)}, stride={tuple(conv_state.stride())}")
            log(f"  conv_state_indices: {conv_state_indices.tolist()}")

            out = causal_conv1d_update(
                x=x,
                conv_state=conv_state,
                weight=weight,
                bias=bias,
                activation="silu",
                conv_state_indices=conv_state_indices,
            )
            o = out.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: (batch={batch}, cdiv(dim={dim}, BLOCK_N=256))")
            log(f"  BLOCK_N=256, KERNEL_WIDTH={width}")
            log("\nOutput:")
            log(f"  out: shape={tuple(o.shape)}, mean={o.float().mean().item():.4f}")
            log("\nSummary:")
            log("  Kernel execution: SUCCESS")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Kernel execution: FAILED")
            fp.write("\n" + "-" * 36 + "\n")
            sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "causal_conv1d_update")
