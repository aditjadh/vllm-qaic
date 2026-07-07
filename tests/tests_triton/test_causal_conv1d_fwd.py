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

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "_causal_conv1d_fwd_kernel"


def main(log_path):
    torch.manual_seed(42)
    # Minimal varlen setup: 2 sequences, dim=16, width=4
    batch = 2
    dim = 16        # must be power-of-2 multiple of BLOCK_N=256? No — dim < BLOCK_N is fine
    width = 4       # kernel width (state_len = width - 1 = 3)
    seq_lens = [8, 6]
    total_tokens = sum(seq_lens)

    # x: (dim, total_tokens) channel-last layout required
    x = torch.randn(dim, total_tokens, dtype=torch.float32, device=DEVICE)
    x = x.contiguous()  # stride(0)=total_tokens, stride(1)=1  → channel-last

    weight = torch.randn(dim, width, dtype=torch.float32, device=DEVICE)
    weight = weight.contiguous()  # stride(1)==1 required

    bias = torch.randn(dim, dtype=torch.float32, device=DEVICE)

    # conv_states: (num_cache_lines, dim, state_len)
    state_len = width - 1
    num_cache_lines = batch
    conv_states = torch.zeros(
        num_cache_lines, dim, state_len, dtype=torch.float32, device=DEVICE
    )
    # stride(1)==1 required
    # conv_states layout: (cache_lines, dim, state_len) → stride=(dim*state_len, state_len, 1)
    # stride(1)=state_len != 1  → need permute to (cache_lines, state_len, dim)? No:
    # assert stride_istate_dim == 1  so dim must be contiguous.
    # Reshape to (num_cache_lines, dim, state_len) then permute so dim is innermost.
    conv_states = conv_states.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    # Above is a no-op if already contiguous. Let's verify: stride(1) should be 1.
    # (cache_lines, dim, state_len).contiguous() → strides (dim*state_len, state_len, 1)
    # stride(1)=state_len, not 1. Need transpose:
    conv_states = conv_states.transpose(1, 2).contiguous().transpose(1, 2)
    # Still not right. Use explicit layout: make dim the last dim.
    # conv_states must satisfy stride(1)==1 (dim stride).
    # So shape (num_cache_lines, state_len, dim) with stride(2)==1 and then view as (n, dim, state_len)?
    # Actually the assertion is on conv_states.stride(-2)==1 in validate_data.
    # Let's just create it in the right order.
    conv_states = torch.zeros(num_cache_lines, state_len, dim, dtype=torch.float32, device=DEVICE)
    # (n, state_len, dim) → conv_states[i, t, d] → stride(-2)=dim, stride(-1)=1
    # But the kernel needs stride_istate_dim=1. stride(-1)==1 means dim axis is fastest → dim axis = -1.
    # The wrapper uses conv_states.stride(1) as stride_istate_dim, so we need dim at axis 1.
    # Create (n, dim, state_len) but with Fortran layout so dim stride==1? Easier:
    # conv_states[n, dim, state_len] with dim contiguous:
    conv_states = torch.zeros(num_cache_lines, dim, state_len, dtype=torch.float32, device=DEVICE)
    # This gives strides (dim*state_len, state_len, 1). stride(1)=state_len != 1.
    # The assertion stride_istate_dim == 1 requires stride(1)==1.
    # So we need shape (n, dim, state_len) stored in column-major along dim:
    # Use: torch.zeros(n, state_len, dim).permute(0,2,1) → shape (n,dim,state_len), stride(1)=1
    conv_states = torch.zeros(
        num_cache_lines, state_len, dim, dtype=torch.float32, device=DEVICE
    ).permute(0, 2, 1)  # shape=(n,dim,state_len), strides=(dim*state_len, 1, dim)

    query_start_loc = torch.tensor([0] + [0] * batch, dtype=torch.int32, device=DEVICE)
    cumsum = 0
    vals = [0]
    for s in seq_lens:
        cumsum += s
        vals.append(cumsum)
    query_start_loc = torch.tensor(vals, dtype=torch.int32, device=DEVICE)

    cache_indices = torch.arange(batch, dtype=torch.int32, device=DEVICE)
    has_initial_state = torch.zeros(batch, dtype=torch.bool, device=DEVICE)

    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  x: {tuple(x.shape)} [dim, total_tokens], dtype={x.dtype}, device={x.device}")
            log(f"  weight: {tuple(weight.shape)} [dim, width], width={width}")
            log(f"  conv_states: {tuple(conv_states.shape)}, strides={tuple(conv_states.stride())}")
            log(f"  seq_lens={seq_lens}, total_tokens={total_tokens}")
            log(f"  query_start_loc={query_start_loc.tolist()}")
            log(f"  cache_indices={cache_indices.tolist()}")

            out = causal_conv1d_fn(
                x=x,
                weight=weight,
                bias=bias,
                conv_states=conv_states,
                query_start_loc=query_start_loc,
                cache_indices=cache_indices,
                has_initial_state=has_initial_state,
                activation="silu",
            )
            o = out.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: (num_programs(BLOCK_M), cdiv(dim, BLOCK_N)) dynamic")
            log(f"  BLOCK_M=8, BLOCK_N=256, KERNEL_WIDTH={width}")
            log("\nOutput:")
            log(f"  out: shape={tuple(o.shape)}, mean={o.mean().item():.4f}, dtype={o.dtype}")
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
    run_test(main, LOG_DIR, "causal_conv1d_fwd")
