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

from vllm.model_executor.layers.mamba.ops.ssd_bmm import _bmm_chunk_fwd

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "_bmm_chunk_fwd_kernel"


def main(log_path):
    torch.manual_seed(42)
    chunk_size = 32
    ngroups = 2
    k = 16       # head dim / dstate
    seqlen = 64  # total sequence length
    nchunks = seqlen // chunk_size

    # a: (seqlen, ngroups, k)
    a = torch.randn(seqlen, ngroups, k, dtype=torch.float32, device=DEVICE)
    b = torch.randn(seqlen, ngroups, k, dtype=torch.float32, device=DEVICE)

    # cu_chunk_seqlens: [0, chunk_size, 2*chunk_size, ...] length nchunks+1
    cu_chunk_seqlens = torch.arange(
        0, seqlen + chunk_size, chunk_size, dtype=torch.int32, device=DEVICE
    )

    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  a: {tuple(a.shape)} [seqlen, ngroups, k], dtype={a.dtype}, device={a.device}")
            log(f"  b: {tuple(b.shape)}")
            log(f"  chunk_size={chunk_size}, nchunks={nchunks}, ngroups={ngroups}, k={k}")
            log(f"  cu_chunk_seqlens: {cu_chunk_seqlens.tolist()}")

            out = _bmm_chunk_fwd(a, b, chunk_size, cu_chunk_seqlens, causal=True)
            o = out.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: (cdiv(chunk_size,BLOCK_M)*cdiv(chunk_size,BLOCK_N), nchunks*ngroups)")
            log(f"  autotuned — chunk_size={chunk_size}, K={k}, IS_CAUSAL=True")
            log("\nOutput:")
            log(f"  out: shape={tuple(o.shape)} [nchunks,ngroups,chunk_size,chunk_size]")
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
    run_test(main, LOG_DIR, "bmm_chunk_fwd")
