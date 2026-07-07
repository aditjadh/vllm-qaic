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

from vllm.v1.worker.gpu.sample.min_p import apply_min_p

# ── Configuration ────────────────────────────────────────────────────────────
NUM_REQS   = 8
VOCAB_SIZE = 4096
DEVICE     = "qaic"

KERNEL_NAME = "_min_p_kernel"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_min_p_{timestamp}.log")


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    torch.manual_seed(42)

    logits = torch.randn(NUM_REQS, VOCAB_SIZE, dtype=torch.float32, device=DEVICE)
    idx_mapping = torch.arange(NUM_REQS, dtype=torch.int32, device=DEVICE)
    # min_p per request (req_state indexed). 0.0 => no-op for that request.
    min_p = torch.tensor([0.0, 0.1, 0.2, 0.5, 0.0, 0.3, 0.05, 0.8],
                         dtype=torch.float32, device=DEVICE)

    logits_ref = logits.clone()
    min_p_ref = min_p.clone()

    grid = (NUM_REQS,)

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")
            log(f, "\nInputs:")
            log(f, f"  logits : shape={tuple(logits.shape)}, dtype={logits.dtype}, device={logits.device}")
            log(f, f"  min_p  : {min_p.cpu().tolist()}")
            log(f, f"  num_reqs={NUM_REQS}, vocab_size={VOCAB_SIZE}")

            apply_min_p(logits, idx_mapping, min_p)
            out_host = logits.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}  (num_reqs,)")
            log(f, "  BLOCK_SIZE: 1024")

            # ── Reference: mask logits below max + log(min_p) to -inf ─────────
            ref = logits_ref.clone()
            for r in range(NUM_REQS):
                mp = float(min_p_ref[r])
                if mp == 0.0:
                    continue
                row = ref[r]
                thr = row.max() + torch.log(torch.tensor(mp))
                row[row < thr] = float("-inf")
            # Compare finite entries; -inf positions must match.
            got = out_host
            same_inf = (torch.isinf(got) == torch.isinf(ref.cpu()))
            finite_mask = ~torch.isinf(ref.cpu())
            diff = (got[finite_mask] - ref.cpu()[finite_mask]).abs().max().item()
            log(f, "\nOutput:")
            log(f, f"  logits after min_p: min={got[finite_mask].min().item():.4f}, "
                   f"max={got.max().item():.4f}")
            log(f, "\nValidation (vs PyTorch min_p threshold):")
            log(f, f"  -inf positions match: {bool(same_inf.all())}")
            log(f, f"  max abs diff (finite): {diff:.6f}")

            assert bool(same_inf.all()), "-inf mask mismatch"
            torch.testing.assert_close(got[finite_mask], ref.cpu()[finite_mask], atol=1e-4, rtol=1e-4)
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
