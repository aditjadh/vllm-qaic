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

from vllm.v1.worker.gpu.sample.logit_bias import apply_logit_bias

# ── Configuration ────────────────────────────────────────────────────────────
# Exercises _bias_kernel via apply_logit_bias: allowed-token restriction, logit
# bias addition, and min-tokens stop-token masking.
NUM_REQS   = 4
VOCAB_SIZE = 512
DEVICE     = "qaic"

KERNEL_NAME = "_bias_kernel"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_bias_{timestamp}.log")

MAX_ALLOWED = 8
MAX_BIAS    = 8
MAX_STOP    = 8


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    torch.manual_seed(42)

    logits = torch.randn(NUM_REQS, VOCAB_SIZE, dtype=torch.float32, device=DEVICE)
    idx_mapping = torch.arange(NUM_REQS, dtype=torch.int32, device=DEVICE)
    pos = torch.zeros(NUM_REQS, dtype=torch.int32, device=DEVICE)  # current position

    # req 0: logit bias on a few tokens; req 1: min-tokens stop masking; others none
    num_allowed = torch.zeros(NUM_REQS, dtype=torch.int32, device=DEVICE)
    allowed = torch.zeros(NUM_REQS, MAX_ALLOWED, dtype=torch.int32, device=DEVICE)

    num_bias = torch.zeros(NUM_REQS, dtype=torch.int32, device=DEVICE)
    bias_token_ids = torch.zeros(NUM_REQS, MAX_BIAS, dtype=torch.int32, device=DEVICE)
    bias_vals = torch.zeros(NUM_REQS, MAX_BIAS, dtype=torch.float32, device=DEVICE)
    # req 0: add +5 to tokens 10, 20, 30
    num_bias[0] = 3
    bias_token_ids[0, :3] = torch.tensor([10, 20, 30], dtype=torch.int32, device=DEVICE)
    bias_vals[0, :3] = torch.tensor([5.0, 5.0, 5.0], dtype=torch.float32, device=DEVICE)

    min_lens = torch.zeros(NUM_REQS, dtype=torch.int32, device=DEVICE)
    num_stop = torch.zeros(NUM_REQS, dtype=torch.int32, device=DEVICE)
    stop_token_ids = torch.zeros(NUM_REQS, MAX_STOP, dtype=torch.int32, device=DEVICE)
    # req 1: pos(0) < min_len(5) so stop tokens 40,41 masked to -inf
    min_lens[1] = 5
    num_stop[1] = 2
    stop_token_ids[1, :2] = torch.tensor([40, 41], dtype=torch.int32, device=DEVICE)

    logits_ref = logits.clone()

    grid = (NUM_REQS,)

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")
            log(f, "\nInputs:")
            log(f, f"  logits : shape={tuple(logits.shape)}, dtype={logits.dtype}, device={logits.device}")
            log(f, f"  num_reqs={NUM_REQS}, vocab_size={VOCAB_SIZE}")
            log(f, f"  req0: +5.0 bias on tokens [10,20,30]")
            log(f, f"  req1: min_len=5, pos=0 -> mask stop tokens [40,41] to -inf")

            apply_logit_bias(
                logits, idx_mapping, pos,
                num_allowed, allowed,
                num_bias, bias_token_ids, bias_vals,
                min_lens, num_stop, stop_token_ids,
            )
            out_host = logits.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}  (num_reqs,)")

            # ── Reference ────────────────────────────────────────────────────
            ref = logits_ref.cpu().clone()
            ref[0, 10] += 5.0
            ref[0, 20] += 5.0
            ref[0, 30] += 5.0
            ref[1, 40] = float("-inf")
            ref[1, 41] = float("-inf")

            same_inf = (torch.isinf(out_host) == torch.isinf(ref))
            finite = ~torch.isinf(ref)
            diff = (out_host[finite] - ref[finite]).abs().max().item()
            log(f, "\nOutput:")
            log(f, f"  logits[0,10]={out_host[0,10]:.4f} (was {logits_ref[0,10].cpu():.4f})")
            log(f, f"  logits[1,40]={out_host[1,40]:.4f} (stop masked)")
            log(f, "\nValidation (vs PyTorch bias add + stop-token mask):")
            log(f, f"  -inf positions match: {bool(same_inf.all())}")
            log(f, f"  max abs diff (finite): {diff:.6f}")

            assert bool(same_inf.all()), "-inf mask mismatch"
            torch.testing.assert_close(out_host[finite], ref[finite], atol=1e-4, rtol=1e-4)
            log(f, "  assert_close: PASSED")

            log(f, "\nSummary:")
            log(f, f"  Kernel execution: SUCCESS (grid {grid})")
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
