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

from vllm.v1.worker.gpu.metrics.logits import get_num_nans

# ── Configuration ────────────────────────────────────────────────────────────
NUM_REQS   = 8
VOCAB_SIZE = 4096
DEVICE     = "qaic"

KERNEL_NAME = "_num_nans_kernel"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_num_nans_{timestamp}.log")


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    torch.manual_seed(42)

    logits = torch.randn(NUM_REQS, VOCAB_SIZE, dtype=torch.float32, device=DEVICE)
    # Inject a known number of NaNs into each row.
    nan_counts = [0, 1, 5, 0, 17, 3, 100, 2]
    for r, c in enumerate(nan_counts):
        if c > 0:
            logits[r, :c] = float("nan")

    logits_ref = logits.clone()

    grid = (NUM_REQS,)

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")
            log(f, "\nInputs:")
            log(f, f"  logits : shape={tuple(logits.shape)}, dtype={logits.dtype}, device={logits.device}")
            log(f, f"  injected NaN counts per row: {nan_counts}")
            log(f, f"  num_reqs={NUM_REQS}, vocab_size={VOCAB_SIZE}")

            out = get_num_nans(logits)
            out_host = out.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}  (num_reqs,)")
            log(f, "  BLOCK_SIZE: 8192")

            log(f, "\nOutput:")
            log(f, f"  num_nans per row: {out_host.tolist()}")

            ref = torch.isnan(logits_ref.cpu()).sum(dim=1).to(torch.int32)
            diff = (out_host - ref).abs().max().item()
            log(f, "\nValidation (vs PyTorch isnan count):")
            log(f, f"  reference: {ref.tolist()}")
            log(f, f"  max abs diff: {diff}")

            torch.testing.assert_close(out_host, ref, atol=0, rtol=0)
            log(f, "  assert_close: PASSED")

            log(f, "\nSummary:")
            log(f, f"  Kernel execution: SUCCESS (grid {grid})")
            log(f, "  Validation vs PyTorch reference: PASSED")
            log(f, f"  max abs diff: {diff}")

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
