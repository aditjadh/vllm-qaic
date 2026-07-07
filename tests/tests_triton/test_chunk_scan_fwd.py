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
from vllm.model_executor.layers.mamba.ops.ssd_chunk_scan import _chunk_scan_fwd
from vllm.model_executor.layers.mamba.ops.ssd_bmm import _bmm_chunk_fwd

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "_chunk_scan_fwd_kernel"


def main(log_path):
    torch.manual_seed(42)
    nheads = 4
    ngroups = 2
    chunk_size = 16
    seqlen = 32
    headdim = 16
    dstate = 8
    nchunks = seqlen // chunk_size

    x = torch.randn(seqlen, nheads, headdim, dtype=torch.float32, device=DEVICE)
    B = torch.randn(seqlen, ngroups, dstate, dtype=torch.float32, device=DEVICE)
    C = torch.randn(seqlen, ngroups, dstate, dtype=torch.float32, device=DEVICE)
    dt_raw = torch.randn(seqlen, nheads, dtype=torch.float32, device=DEVICE) * 0.1
    A = -torch.rand(nheads, dtype=torch.float32, device=DEVICE)
    D = torch.randn(nheads, dtype=torch.float32, device=DEVICE)

    cu_chunk_seqlens = torch.arange(
        0, seqlen + chunk_size, chunk_size, dtype=torch.int32, device=DEVICE
    )
    # seq_idx: one index per chunk (all same sequence → index 0)
    seq_idx = torch.zeros(nchunks, dtype=torch.int32, device=DEVICE)

    dA_cumsum, dt = _chunk_cumsum_fwd(
        dt=dt_raw, A=A, chunk_size=chunk_size,
        cu_chunk_seqlens=cu_chunk_seqlens, dt_softplus=False,
    )

    states = _chunk_state_fwd(
        B=B, x=x, dt=dt, dA_cumsum=dA_cumsum,
        cu_chunk_seqlens=cu_chunk_seqlens,
    )

    # cb (chunk-wise attention): (nchunks, ngroups, chunk_size, chunk_size)
    cb = _bmm_chunk_fwd(B, C, chunk_size, cu_chunk_seqlens, causal=True)

    out = torch.empty(seqlen, nheads, headdim, dtype=torch.float32, device=DEVICE)

    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  x: {tuple(x.shape)} [seqlen,nheads,headdim], device={x.device}")
            log(f"  C: {tuple(C.shape)}, cb: {tuple(cb.shape)}")
            log(f"  dt: {tuple(dt.shape)}, dA_cumsum: {tuple(dA_cumsum.shape)}")
            log(f"  states: {tuple(states.shape)}, seq_idx: {tuple(seq_idx.shape)}")
            log(f"  D: {tuple(D.shape)}")
            log(f"  chunk_size={chunk_size}, nchunks={nchunks}, nheads={nheads}")
            log(f"  ngroups={ngroups}, headdim={headdim}, dstate={dstate}")

            _chunk_scan_fwd(
                cb=cb,
                x=x,
                dt=dt,
                dA_cumsum=dA_cumsum,
                C=C,
                states=states,
                cu_chunk_seqlens=cu_chunk_seqlens,
                out=out,
                seq_idx=seq_idx,
                D=D,
            )
            o = out.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: (cdiv(chunk_size,BLOCK_M)*cdiv(headdim,BLOCK_N), nchunks={nchunks}, nheads={nheads})")
            log(f"  autotuned: chunk_size={chunk_size}, hdim={headdim}, dstate={dstate}")
            log("\nOutput:")
            log(f"  out: shape={tuple(o.shape)} [seqlen,nheads,headdim]")
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
    run_test(main, LOG_DIR, "chunk_scan_fwd")
