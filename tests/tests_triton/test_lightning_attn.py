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

import vllm.model_executor.layers.lightning_attn as _lattn_mod

# _attention.forward calls torch.cuda.get_device_capability() which fails on
# QAIC (no CUDA). Patch it to return (8, 0) so the capability check passes.
import torch.cuda as _tc
if not hasattr(_tc, '_orig_get_device_capability'):
    _tc._orig_get_device_capability = _tc.get_device_capability
_tc.get_device_capability = lambda *a, **kw: (8, 0)

from vllm.model_executor.layers.lightning_attn import lightning_attention

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "_fwd_diag_kernel / _fwd_kv_parallel / _fwd_kv_reduce / _fwd_none_diag_kernel"


def main(log_path):
    torch.manual_seed(42)
    # lightning_attention expects [batch, heads, seq_len, dim]
    B, H, N, D = 2, 4, 64, 64
    q = torch.randn(B, H, N, D, dtype=torch.float32, device=DEVICE)
    k = torch.randn(B, H, N, D, dtype=torch.float32, device=DEVICE)
    v = torch.randn(B, H, N, D, dtype=torch.float32, device=DEVICE)
    # decay per head: positive values
    ed = torch.rand(H, dtype=torch.float32, device=DEVICE) * 0.1 + 0.01

    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            log(os.path.basename(log_path)[:-4]); log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  q: {tuple(q.shape)} [B,H,N,D], device={q.device}")
            log(f"  k: {tuple(k.shape)}")
            log(f"  v: {tuple(v.shape)}")
            log(f"  ed (decay): {tuple(ed.shape)}, range=[{ed.min().item():.4f},{ed.max().item():.4f}]")
            log(f"  block_size=64")

            out, kv = lightning_attention(q, k, v, ed, block_size=64)
            o = out.cpu(); kv_c = kv.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log("  _fwd_diag_kernel: grid=(B*H*NUM_BLOCK, NUM_CBLOCK)")
            log("  _fwd_kv_parallel: grid=(B*H, NUM_BLOCK)")
            log("  _fwd_kv_reduce:   grid=(B*H,)")
            log("  _fwd_none_diag_kernel: grid=(B*H, NUM_BLOCK*NUM_CBLOCK, NUM_E_FBLOCK)")
            log("\nOutput:")
            log(f"  out: shape={tuple(o.shape)}, mean={o.mean().item():.4f}")
            log(f"  kv_history: shape={tuple(kv_c.shape)}, mean={kv_c.mean().item():.4f}")
            log("\nSummary:")
            log(f"  Kernel execution: SUCCESS")
            log(f"  Output shape matches expected [B,H,N,D]={[B,H,N,D]}: {list(o.shape) == [B,H,N,D]}")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Kernel execution: FAILED")
            fp.write("\n" + "-" * 36 + "\n"); sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "lightning_attn")
