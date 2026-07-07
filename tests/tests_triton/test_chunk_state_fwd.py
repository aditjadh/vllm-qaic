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
)

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "_chunk_state_fwd_kernel"


def main(log_path):
    torch.manual_seed(42)
    nheads = 4
    ngroups = 2
    chunk_size = 16
    seqlen = 32
    headdim = 16
    dstate = 8
    nchunks = seqlen // chunk_size

    # x: (seqlen, nheads, headdim)
    x = torch.randn(seqlen, nheads, headdim, dtype=torch.float32, device=DEVICE)
    # B: (seqlen, ngroups, dstate)
    B = torch.randn(seqlen, ngroups, dstate, dtype=torch.float32, device=DEVICE)
    # dt raw (seqlen, nheads)
    dt_raw = torch.randn(seqlen, nheads, dtype=torch.float32, device=DEVICE) * 0.1
    A = -torch.rand(nheads, dtype=torch.float32, device=DEVICE)

    cu_chunk_seqlens = torch.arange(
        0, seqlen + chunk_size, chunk_size, dtype=torch.int32, device=DEVICE
    )

    # First compute cumsum (dt, dA_cumsum) then chunk_state_fwd
    dA_cumsum, dt = _chunk_cumsum_fwd(
        dt=dt_raw,
        A=A,
        chunk_size=chunk_size,
        cu_chunk_seqlens=cu_chunk_seqlens,
        dt_softplus=False,
    )
    # dt: (nheads, nchunks, chunk_size), dA_cumsum: same

    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  x: {tuple(x.shape)} [seqlen, nheads, headdim], device={x.device}")
            log(f"  B: {tuple(B.shape)} [seqlen, ngroups, dstate]")
            log(f"  dt: {tuple(dt.shape)} [nheads, nchunks, chunk_size]")
            log(f"  dA_cumsum: {tuple(dA_cumsum.shape)}")
            log(f"  chunk_size={chunk_size}, nchunks={nchunks}, nheads={nheads}")
            log(f"  ngroups={ngroups}, headdim={headdim}, dstate={dstate}")

            states = _chunk_state_fwd(
                B=B,
                x=x,
                dt=dt,
                dA_cumsum=dA_cumsum,
                cu_chunk_seqlens=cu_chunk_seqlens,
            )
            s = states.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: (cdiv(headdim,BLOCK_M)*cdiv(dstate,BLOCK_N), nchunks={nchunks}, nheads={nheads})")
            log(f"  autotuned: hdim={headdim}, dstate={dstate}, chunk_size={chunk_size}")
            log("\nOutput:")
            log(f"  states: shape={tuple(s.shape)} [nchunks,nheads,headdim,dstate]")
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
    run_test(main, LOG_DIR, "chunk_state_fwd")
