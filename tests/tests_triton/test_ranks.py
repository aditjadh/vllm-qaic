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

from vllm.v1.worker.gpu.sample.logprob import _ranks_kernel

# ── Configuration ────────────────────────────────────────────────────────────
# _ranks_kernel: for each req, rank = number of logits strictly greater than the
# logit of the selected token_id.
NUM_REQS   = 8
VOCAB_SIZE = 4096
DEVICE     = "qaic"

KERNEL_NAME = "_ranks_kernel"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_ranks_{timestamp}.log")


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    torch.manual_seed(42)

    logits = torch.randn(NUM_REQS, VOCAB_SIZE, dtype=torch.float32, device=DEVICE)
    token_ids = torch.randint(0, VOCAB_SIZE, (NUM_REQS,), dtype=torch.int64, device=DEVICE)
    out = torch.empty(NUM_REQS, dtype=torch.int64, device=DEVICE)

    logits_ref = logits.clone()
    token_ids_ref = token_ids.clone()

    grid = (NUM_REQS,)
    BLOCK_SIZE = 8192

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")
            log(f, "\nInputs:")
            log(f, f"  logits    : shape={tuple(logits.shape)}, dtype={logits.dtype}, device={logits.device}")
            log(f, f"  token_ids : {token_ids.cpu().tolist()}")
            log(f, f"  num_reqs={NUM_REQS}, vocab_size={VOCAB_SIZE}")

            _ranks_kernel[grid](
                out,
                logits,
                logits.stride(0),
                token_ids,
                VOCAB_SIZE,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            out_host = out.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}  (num_reqs,)")
            log(f, f"  BLOCK_SIZE: {BLOCK_SIZE}")

            log(f, "\nOutput:")
            log(f, f"  ranks: {out_host.tolist()}")

            # ── Reference: count logits strictly greater than selected ───────
            ref = torch.empty(NUM_REQS, dtype=torch.int64)
            for r in range(NUM_REQS):
                x = logits_ref[r, token_ids_ref[r]]
                ref[r] = int((logits_ref[r] > x).sum())
            diff = (out_host - ref).abs().max().item()
            log(f, "\nValidation (vs PyTorch count of logits > selected):")
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
