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

from vllm.model_executor.layers.mamba.ops.ssd_chunk_state import _chunk_cumsum_fwd

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "_chunk_cumsum_fwd_kernel"


def main(log_path):
    torch.manual_seed(42)
    nheads = 4
    chunk_size = 16
    seqlen = 32
    nchunks = seqlen // chunk_size

    # dt: (seqlen, nheads)
    dt = torch.randn(seqlen, nheads, dtype=torch.float32, device=DEVICE) * 0.1
    # A: (nheads,) — negative values typical for SSM stability
    A = -torch.rand(nheads, dtype=torch.float32, device=DEVICE)
    dt_bias = torch.randn(nheads, dtype=torch.float32, device=DEVICE) * 0.01

    # cu_chunk_seqlens: length nchunks+1
    cu_chunk_seqlens = torch.arange(
        0, seqlen + chunk_size, chunk_size, dtype=torch.int32, device=DEVICE
    )

    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  dt: {tuple(dt.shape)} [seqlen, nheads], dtype={dt.dtype}, device={dt.device}")
            log(f"  A: {tuple(A.shape)}, dt_bias: {tuple(dt_bias.shape)}")
            log(f"  chunk_size={chunk_size}, nchunks={nchunks}, nheads={nheads}, seqlen={seqlen}")

            dA_cumsum, dt_out = _chunk_cumsum_fwd(
                dt=dt,
                A=A,
                chunk_size=chunk_size,
                cu_chunk_seqlens=cu_chunk_seqlens,
                dt_bias=dt_bias,
                dt_softplus=True,
                dt_limit=(0.0, float("inf")),
            )
            dac = dA_cumsum.cpu()
            dto = dt_out.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: (nchunks={nchunks}, cdiv(nheads={nheads}, BLOCK_SIZE_H))")
            log(f"  BLOCK_SIZE_H autotuned, BLOCK_SIZE_CHUNK={chunk_size}")
            log("\nOutput:")
            log(f"  dA_cumsum: shape={tuple(dac.shape)} [nheads,nchunks,chunk_size], mean={dac.mean().item():.4f}")
            log(f"  dt_out:    shape={tuple(dto.shape)}, mean={dto.mean().item():.4f}")
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
    run_test(main, LOG_DIR, "chunk_cumsum_fwd")
