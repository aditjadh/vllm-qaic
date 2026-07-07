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

from vllm.v1.worker.gpu.sample.penalties import apply_penalties
from vllm.utils.math_utils import cdiv

# ── Configuration ────────────────────────────────────────────────────────────
# _penalties_kernel applies repetition, frequency, and presence penalties using
# a packed prompt-token bitmask and per-token output counts.
NUM_REQS   = 4
VOCAB_SIZE = 512
DEVICE     = "qaic"

KERNEL_NAME = "_penalties_kernel"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_penalties_{timestamp}.log")


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    torch.manual_seed(42)

    logits = torch.randn(NUM_REQS, VOCAB_SIZE, dtype=torch.float32, device=DEVICE)
    idx_mapping = torch.arange(NUM_REQS, dtype=torch.int32, device=DEVICE)

    rep = torch.tensor([1.0, 1.5, 1.0, 1.2], dtype=torch.float32, device=DEVICE)
    freq = torch.tensor([0.0, 0.0, 0.5, 0.1], dtype=torch.float32, device=DEVICE)
    pres = torch.tensor([0.0, 0.3, 0.0, 0.2], dtype=torch.float32, device=DEVICE)

    num_words = cdiv(VOCAB_SIZE, 32)
    prompt_bin_mask = torch.zeros(NUM_REQS, num_words, dtype=torch.int32, device=DEVICE)
    output_bin_counts = torch.zeros(NUM_REQS, VOCAB_SIZE, dtype=torch.int32, device=DEVICE)

    # Populate some prompt tokens & output counts.
    prompt_tokens = {1: [5, 6, 7], 3: [100, 101]}
    output_tokens = {1: {5: 2, 50: 1}, 2: {10: 3, 11: 1}, 3: {100: 1, 200: 4}}
    for r, toks in prompt_tokens.items():
        for t in toks:
            prompt_bin_mask[r, t // 32] |= (1 << (t % 32))
    for r, d in output_tokens.items():
        for t, c in d.items():
            output_bin_counts[r, t] = c

    logits_ref = logits.clone()
    rep_ref, freq_ref, pres_ref = rep.clone(), freq.clone(), pres.clone()
    mask_ref = prompt_bin_mask.clone()
    counts_ref = output_bin_counts.clone()

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")
            log(f, "\nInputs:")
            log(f, f"  logits : shape={tuple(logits.shape)}, dtype={logits.dtype}, device={logits.device}")
            log(f, f"  repetition_penalty: {rep.cpu().tolist()}")
            log(f, f"  frequency_penalty : {freq.cpu().tolist()}")
            log(f, f"  presence_penalty  : {pres.cpu().tolist()}")
            log(f, f"  num_reqs={NUM_REQS}, vocab_size={VOCAB_SIZE}")

            apply_penalties(
                logits, idx_mapping, rep, freq, pres,
                prompt_bin_mask, output_bin_counts,
            )
            out_host = logits.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, "  grid: (num_reqs, cdiv(vocab, 8192))")

            # ── Reference ────────────────────────────────────────────────────
            ref = logits_ref.cpu().float().clone()
            for r in range(NUM_REQS):
                rp = float(rep_ref[r]); fp = float(freq_ref[r]); pp = float(pres_ref[r])
                obc = counts_ref[r].cpu()
                out_mask = obc > 0
                # prompt bitmask unpacked
                pmask = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
                for v in range(VOCAB_SIZE):
                    if (int(mask_ref[r, v // 32]) >> (v % 32)) & 1:
                        pmask[v] = True
                if rp != 1.0:
                    scale = torch.where(pmask | out_mask, torch.tensor(rp), torch.tensor(1.0))
                    row = ref[r]
                    ref[r] = torch.where(row > 0, row / scale, row * scale)
                ref[r] -= fp * obc.float()
                ref[r] -= pp * out_mask.float()

            diff = (out_host - ref).abs().max().item()
            log(f, "\nValidation (vs PyTorch repetition+frequency+presence penalties):")
            log(f, f"  max abs diff: {diff:.5f}")

            torch.testing.assert_close(out_host, ref, atol=1e-2, rtol=1e-3)
            log(f, "  assert_close: PASSED")

            log(f, "\nSummary:")
            log(f, "  Kernel execution: SUCCESS")
            log(f, "  Validation vs PyTorch reference: PASSED")
            log(f, f"  max abs diff: {diff:.5f}")

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
