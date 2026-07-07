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
_tc._orig_device = getattr(_tc, 'device', None)
_tc.device = _noop_device

from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    _selective_scan_update_kernel,
    selective_state_update,
)
import triton as _triton

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "_selective_scan_update_kernel"


def main(log_path):
    torch.manual_seed(42)
    batch = 2
    nheads = 4
    ngroups = 2
    dim = 16
    dstate = 8

    # state: (batch, nheads, dim, dstate)
    state = torch.randn(batch, nheads, dim, dstate, dtype=torch.float32, device=DEVICE)
    x = torch.randn(batch, nheads, dim, dtype=torch.float32, device=DEVICE)
    dt = torch.randn(batch, nheads, dim, dtype=torch.float32, device=DEVICE)
    A = -torch.rand(nheads, dim, dstate, dtype=torch.float32, device=DEVICE)
    B = torch.randn(batch, ngroups, dstate, dtype=torch.float32, device=DEVICE)
    C = torch.randn(batch, ngroups, dstate, dtype=torch.float32, device=DEVICE)
    dt_bias = torch.randn(nheads, dim, dtype=torch.float32, device=DEVICE)
    D = torch.randn(nheads, dim, dtype=torch.float32, device=DEVICE)
    z = torch.randn(batch, nheads, dim, dtype=torch.float32, device=DEVICE)
    out = torch.zeros(batch, nheads, dim, dtype=torch.float32, device=DEVICE)

    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  state: {tuple(state.shape)} [batch,nheads,dim,dstate], device={state.device}")
            log(f"  x: {tuple(x.shape)}, dt: {tuple(dt.shape)}")
            log(f"  A: {tuple(A.shape)}, B: {tuple(B.shape)}, C: {tuple(C.shape)}")
            log(f"  dt_bias: {tuple(dt_bias.shape)}")
            log(f"  D: {tuple(D.shape)}, z: {tuple(z.shape)}")
            log(f"  batch={batch}, nheads={nheads}, ngroups={ngroups}, dim={dim}, dstate={dstate}")

            selective_state_update(
                state=state,
                x=x,
                dt=dt,
                A=A,
                B=B,
                C=C,
                D=D,
                z=z,
                dt_bias=dt_bias,
                dt_softplus=True,
                out=out,
            )
            o = out.cpu()
            s = state.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: (cdiv(dim={dim}, BLOCK_SIZE_M), N={batch}, nheads={nheads})")
            log(f"  BLOCK_SIZE_M chosen by dstate={dstate}: {'32' if dstate<=16 else '16' if dstate<=32 else '8'}")
            log("\nOutput:")
            log(f"  out: shape={tuple(o.shape)}, mean={o.float().mean().item():.4f}")
            log(f"  state (updated in-place): mean={s.float().mean().item():.4f}")
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
    run_test(main, LOG_DIR, "selective_scan_update")
