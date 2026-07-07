# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
import os
import sys
import traceback
from datetime import datetime
from contextlib import contextmanager

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vllm"))
from _ktest import run_test, logger

# Patch torch.cuda.device to no-op for QAIC
@contextmanager
def _noop_device(*a, **kw):
    yield

import torch.cuda as _tc
_tc.device = _noop_device

from vllm.model_executor.layers.mamba.ops.ssd_chunk_state import (
    _chunk_cumsum_fwd,
    _chunk_state_fwd,
    chunk_state_varlen,
)

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "_chunk_state_varlen_kernel"


def main(log_path):
    torch.manual_seed(42)
    # 2 variable-length sequences packed into total_seqlen tokens
    nheads = 4
    ngroups = 2
    chunk_size = 16
    headdim = 16
    dstate = 8
    seq_lens = [20, 12]
    total_seqlen = sum(seq_lens)
    batch = len(seq_lens)
    nchunks = (total_seqlen + chunk_size - 1) // chunk_size  # covers full token range

    x = torch.randn(total_seqlen, nheads, headdim, dtype=torch.float32, device=DEVICE)
    B = torch.randn(total_seqlen, ngroups, dstate, dtype=torch.float32, device=DEVICE)
    dt_raw = torch.randn(total_seqlen, nheads, dtype=torch.float32, device=DEVICE) * 0.1
    A = -torch.rand(nheads, dtype=torch.float32, device=DEVICE)

    cu_chunk_seqlens = torch.arange(
        0, total_seqlen + chunk_size, chunk_size, dtype=torch.int32, device=DEVICE
    )
    # Clip to actual nchunks
    cu_chunk_seqlens = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=DEVICE),
        torch.clamp(
            torch.arange(chunk_size, (nchunks + 1) * chunk_size, chunk_size, device=DEVICE),
            max=total_seqlen,
        ).to(torch.int32),
    ])

    # cu_seqlens for varlen: cumulative over sequences
    cu_seqlens = torch.tensor([0] + [sum(seq_lens[:i+1]) for i in range(batch)],
                               dtype=torch.int32, device=DEVICE)

    dA_cumsum, dt = _chunk_cumsum_fwd(
        dt=dt_raw,
        A=A,
        chunk_size=chunk_size,
        cu_chunk_seqlens=cu_chunk_seqlens,
        dt_softplus=False,
    )

    # chunk_states: from _chunk_state_fwd
    chunk_states = _chunk_state_fwd(
        B=B,
        x=x,
        dt=dt,
        dA_cumsum=dA_cumsum,
        cu_chunk_seqlens=cu_chunk_seqlens,
    )

    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  x: {tuple(x.shape)} [total_seqlen,nheads,headdim], device={x.device}")
            log(f"  B: {tuple(B.shape)}, dt: {tuple(dt.shape)}, dA_cumsum: {tuple(dA_cumsum.shape)}")
            log(f"  chunk_states: {tuple(chunk_states.shape)} [nchunks,nheads,headdim,dstate]")
            log(f"  cu_seqlens: {cu_seqlens.tolist()}, batch={batch}")
            log(f"  nheads={nheads}, ngroups={ngroups}, headdim={headdim}, dstate={dstate}")
            log(f"  chunk_size={chunk_size}, nchunks={nchunks}")

            states = chunk_state_varlen(
                B=B,
                x=x,
                dt=dt,
                dA_cumsum=dA_cumsum,
                cu_seqlens=cu_seqlens,
                chunk_states=chunk_states,
            )
            s = states.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: (cdiv(headdim,BLOCK_M)*cdiv(dstate,BLOCK_N), batch={batch}, nheads={nheads})")
            log(f"  autotuned: hdim={headdim}, dstate={dstate}, chunk_size={chunk_size}")
            log("\nOutput:")
            log(f"  states: shape={tuple(s.shape)} [batch,nheads,headdim,dstate]")
            log(f"  mean={s.float().mean().item():.4f}, dtype={s.dtype}")
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
    run_test(main, LOG_DIR, "chunk_state_varlen")
