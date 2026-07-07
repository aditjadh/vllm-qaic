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

from vllm.model_executor.layers.batch_invariant import mean_dim

# ── Configuration ────────────────────────────────────────────────────────────
SHAPE      = (8, 64, 16)
REDUCE_DIM = 1
DTYPE      = torch.float32
DEVICE     = "qaic"

KERNEL_NAME = "mean_kernel"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_mean_kernel_{timestamp}.log")


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    torch.manual_seed(42)

    x = torch.randn(*SHAPE, dtype=DTYPE, device=DEVICE)
    x_ref = x.clone()

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")
            log(f, "\nInputs:")
            log(f, f"  x : shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}")
            log(f, f"  reduce dim={REDUCE_DIM}")

            out = mean_dim(x, dim=REDUCE_DIM, keepdim=False)
            out_host = out.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, "  grid: (M * K,)  (flattened outer*inner of reduced view)")
            log(f, "  BLOCK_SIZE: 1024")

            log(f, "\nOutput:")
            log(f, f"  out : shape={tuple(out.shape)}, min={out_host.min().item():.4f}, "
                   f"max={out_host.max().item():.4f}, mean={out_host.mean().item():.4f}")

            ref = x_ref.float().mean(dim=REDUCE_DIM).cpu()
            diff = (out_host - ref).abs().max().item()
            log(f, "\nValidation (vs torch.mean):")
            log(f, f"  max abs diff: {diff:.6f}")

            torch.testing.assert_close(out_host, ref, atol=1e-4, rtol=1e-4)
            log(f, "  assert_close: PASSED")

            log(f, "\nSummary:")
            log(f, "  Kernel execution: SUCCESS")
            log(f, "  Validation vs PyTorch reference: PASSED")
            log(f, f"  max abs diff: {diff:.6f}")

        except Exception:
            log(f, f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log(f, "\nSummary:")
            log(f, "  Validation vs PyTorch reference: FAILED")
            sys.exit(1)
        finally:
            log(f, "\n" + "-" * 36)

    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
