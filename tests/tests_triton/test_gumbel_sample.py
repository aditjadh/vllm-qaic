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

from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

# ── Configuration ────────────────────────────────────────────────────────────
# Exercises _gumbel_sample_kernel via gumbel_sample.
# Tests: greedy path (temp=0), stochastic path (temp>0), temperature scaling.
NUM_REQS   = 4
VOCAB_SIZE  = 2048
DEVICE      = "qaic"

KERNEL_NAME = "_gumbel_sample_kernel"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_gumbel_sample_{timestamp}.log")

BLOCK_SIZE = 1024


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    torch.manual_seed(42)

    # Logits: put a clear spike at token 7 for req 0 (greedy check)
    logits = torch.randn(NUM_REQS, VOCAB_SIZE, dtype=torch.float32, device=DEVICE)
    logits[0, 7] = 100.0

    idx_mapping = torch.arange(NUM_REQS, dtype=torch.int32, device=DEVICE)
    # req 0: temp=0 (greedy), others: temp=1.0
    temperature = torch.ones(NUM_REQS, dtype=torch.float32, device=DEVICE)
    temperature[0] = 0.0

    seeds = torch.randint(0, 2**31 - 1, (NUM_REQS,), dtype=torch.int64, device=DEVICE)
    pos   = torch.zeros(NUM_REQS, dtype=torch.int64, device=DEVICE)

    num_blocks = (VOCAB_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (NUM_REQS, num_blocks)

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")
            log(f, "\nInputs:")
            log(f, f"  logits       : shape={tuple(logits.shape)}, dtype={logits.dtype}, device={logits.device}")
            log(f, f"  idx_mapping  : {idx_mapping.tolist()}")
            log(f, f"  temperature  : {temperature.tolist()}")
            log(f, f"  seeds        : {seeds.tolist()}")
            log(f, f"  pos          : {pos.tolist()}")
            log(f, f"  num_reqs={NUM_REQS}, vocab_size={VOCAB_SIZE}")
            log(f, f"  req0: temp=0 (greedy), spike at token 7 -> expect sampled=7")
            log(f, f"  req1-3: temp=1.0 (stochastic)")

            # ── Case 1: greedy (APPLY_TEMPERATURE=False, temp=0 path) ──────────
            sampled = gumbel_sample(
                logits.clone(),
                idx_mapping,
                temperature,
                seeds,
                pos,
                apply_temperature=False,
            )
            sampled_host = sampled.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}  (num_reqs, num_blocks)")
            log(f, f"  BLOCK_SIZE: {BLOCK_SIZE}")

            log(f, "\nOutput (greedy / temp=0 for req0):")
            for i in range(NUM_REQS):
                log(f, f"  req{i}: sampled token = {sampled_host[i].item()}")

            # req0 temp=0 -> no gumbel noise -> argmax -> must be 7
            assert sampled_host[0].item() == 7, (
                f"Greedy req0 expected token 7, got {sampled_host[0].item()}"
            )
            log(f, "\nValidation:")
            log(f, f"  req0 greedy argmax == 7: PASSED")

            # ── Case 2: stochastic (temp=1) ──────────────────────────────────
            sampled2 = gumbel_sample(
                logits.clone(),
                idx_mapping,
                temperature,
                seeds,
                pos,
                apply_temperature=True,
            )
            sampled2_host = sampled2.cpu()
            log(f, "\nOutput (stochastic, APPLY_TEMPERATURE=True):")
            for i in range(NUM_REQS):
                log(f, f"  req{i}: sampled token = {sampled2_host[i].item()}")

            # req0 still temp=0 -> still greedy
            assert sampled2_host[0].item() == 7, (
                f"Greedy req0 (apply_temp=True) expected 7, got {sampled2_host[0].item()}"
            )
            log(f, f"  req0 greedy (apply_temperature=True) == 7: PASSED")

            # stochastic tokens must be in valid range
            for i in range(1, NUM_REQS):
                tok = sampled2_host[i].item()
                assert 0 <= tok < VOCAB_SIZE, f"req{i} token {tok} out of range"
            log(f, f"  req1-3 sampled tokens in [0, {VOCAB_SIZE}): PASSED")

            log(f, "\nSummary:")
            log(f, f"  Kernel execution: SUCCESS (grid {grid})")
            log(f, "  Greedy path validation: PASSED")
            log(f, "  Stochastic range validation: PASSED")

        except Exception:
            log(f, f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log(f, "\nSummary:")
            log(f, "  Kernel execution: FAILED")
            sys.exit(1)
        finally:
            log(f, "\n" + "-" * 36)

    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
