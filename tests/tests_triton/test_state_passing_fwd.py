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

from vllm.model_executor.layers.mamba.ops.ssd_chunk_state import _chunk_cumsum_fwd, _chunk_state_fwd
from vllm.model_executor.layers.mamba.ops.ssd_state_passing import _state_passing_fwd

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "_state_passing_fwd_kernel"


def main(log_path):
    torch.manual_seed(42)
    nheads = 4
    chunk_size = 16
    seqlen = 32
    headdim = 16
    dstate = 8
    nchunks = seqlen // chunk_size
    # states has shape (nchunks, nheads, dim) where dim = headdim * dstate flattened?
    # No: states shape per _state_passing_fwd: (nchunks, nheads, dim) where dim = headdim*dstate
    # Actually from source: nchunks, nheads, dim = states.shape — dim here is the product
    # but _chunk_state_fwd returns (nchunks, nheads, headdim, dstate).
    # _state_passing_fwd expects (nchunks, nheads, dim) where dim is flat.
    dim = headdim * dstate  # = 128

    ngroups = 2
    x = torch.randn(seqlen, nheads, headdim, dtype=torch.float32, device=DEVICE)
    B = torch.randn(seqlen, ngroups, dstate, dtype=torch.float32, device=DEVICE)
    dt_raw = torch.randn(seqlen, nheads, dtype=torch.float32, device=DEVICE) * 0.1
    A = -torch.rand(nheads, dtype=torch.float32, device=DEVICE)

    cu_chunk_seqlens = torch.arange(
        0, seqlen + chunk_size, chunk_size, dtype=torch.int32, device=DEVICE
    )
    seq_idx = torch.zeros(nchunks, dtype=torch.int32, device=DEVICE)

    dA_cumsum, dt = _chunk_cumsum_fwd(
        dt=dt_raw, A=A, chunk_size=chunk_size,
        cu_chunk_seqlens=cu_chunk_seqlens, dt_softplus=False,
    )

    chunk_states_4d = _chunk_state_fwd(
        B=B, x=x, dt=dt, dA_cumsum=dA_cumsum,
        cu_chunk_seqlens=cu_chunk_seqlens,
    )
    # Reshape to (nchunks, nheads, dim) for _state_passing_fwd
    states = chunk_states_4d.reshape(nchunks, nheads, dim).contiguous()

    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  states: {tuple(states.shape)} [nchunks,nheads,dim], device={states.device}")
            log(f"  dA_cumsum: {tuple(dA_cumsum.shape)} [nheads,nchunks,chunk_size]")
            log(f"  seq_idx: {tuple(seq_idx.shape)}")
            log(f"  nchunks={nchunks}, nheads={nheads}, dim={dim}, chunk_size={chunk_size}")

            out = _state_passing_fwd(
                states=states,
                dA_cumsum=dA_cumsum,
                cu_chunk_seqlens=cu_chunk_seqlens,
                seq_idx=seq_idx,
            )
            o = out.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: (cdiv(dim={dim}, BLOCK_SIZE), nheads={nheads})")
            log(f"  BLOCK_SIZE autotuned (options: 64,128,256,512,1024,2048)")
            log("\nOutput:")
            log(f"  out: shape={tuple(o.shape)} [nchunks,nheads,dim]")
            log(f"  mean={o.float().mean().item():.4f}, dtype={o.dtype}")
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
    run_test(main, LOG_DIR, "state_passing_fwd")
