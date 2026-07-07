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

from vllm.model_executor.layers.lightning_attn import linear_decode_forward_triton

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "_linear_attn_decode_kernel"


def main(log_path):
    torch.manual_seed(42)
    B, H, D = 2, 4, 64
    BLOCK_SIZE = 32
    # q/k/v: [B, H, 1, D]
    q = torch.randn(B, H, 1, D, dtype=torch.float32, device=DEVICE)
    k = torch.randn(B, H, 1, D, dtype=torch.float32, device=DEVICE)
    v = torch.randn(B, H, 1, D, dtype=torch.float32, device=DEVICE)
    # kv_caches: [B, H, D, D] (slot x head x d_k x d_v)
    kv_caches = torch.zeros(B, H, D, D, dtype=torch.float32, device=DEVICE)
    slope_rate = torch.rand(H, dtype=torch.float32, device=DEVICE) * 0.1 + 0.01
    slot_idx = torch.arange(B, dtype=torch.int32, device=DEVICE)

    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            log(os.path.basename(log_path)[:-4]); log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  q/k/v: {tuple(q.shape)} [B,H,1,D], device={q.device}")
            log(f"  kv_caches: {tuple(kv_caches.shape)} [B,H,D,D]")
            log(f"  slope_rate: {tuple(slope_rate.shape)}, BLOCK_SIZE={BLOCK_SIZE}")
            log(f"  slot_idx: {slot_idx.tolist()}")

            out = linear_decode_forward_triton(q, k, v, kv_caches, slope_rate, slot_idx,
                                               BLOCK_SIZE=BLOCK_SIZE)
            o = out.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: ({B}, {H}, {D // BLOCK_SIZE})  (B, H, D//BLOCK_SIZE)")
            log("\nOutput:")
            log(f"  out: shape={tuple(o.shape)}, mean={o.mean().item():.4f}")
            log("\nSummary:")
            log(f"  Kernel execution: SUCCESS (grid ({B},{H},{D//BLOCK_SIZE}))")
            log(f"  Output shape: {list(o.shape)}")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Kernel execution: FAILED")
            fp.write("\n" + "-" * 36 + "\n"); sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "linear_attn_decode")
