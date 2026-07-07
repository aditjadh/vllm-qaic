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

from vllm.v1.worker.gpu.sample.penalties import bincount

# ── Configuration ────────────────────────────────────────────────────────────
# bincount fills:
#   - prompt_bin_mask: a packed bitmask of which token ids appear in the prompt
#     (tokens [0, prompt_len))
#   - output_bin_counts: per-token-id counts over the output region
#     (tokens [prompt_len, prefill_len))
VOCAB_SIZE  = 256
PROMPT_LEN  = 12
PREFILL_LEN = 20            # 8 output tokens
DEVICE      = "qaic"

KERNEL_NAME = "_bincount_kernel"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_bincount_{timestamp}.log")


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    torch.manual_seed(42)

    prefill_token_ids = torch.randint(
        0, VOCAB_SIZE, (PREFILL_LEN,), dtype=torch.int32, device=DEVICE
    )
    num_words = (VOCAB_SIZE + 31) // 32
    prompt_bin_mask = torch.zeros(num_words, dtype=torch.int32, device=DEVICE)
    output_bin_counts = torch.zeros(VOCAB_SIZE, dtype=torch.int32, device=DEVICE)

    tokens_ref = prefill_token_ids.clone()

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")
            log(f, "\nInputs:")
            log(f, f"  prefill_token_ids : shape={tuple(prefill_token_ids.shape)}, dtype={prefill_token_ids.dtype}, device={prefill_token_ids.device}")
            log(f, f"  vocab_size={VOCAB_SIZE}, prompt_len={PROMPT_LEN}, prefill_len={PREFILL_LEN}")

            bincount(
                prefill_token_ids,
                PREFILL_LEN,
                PROMPT_LEN,
                prompt_bin_mask,
                output_bin_counts,
            )
            mask_host = prompt_bin_mask.cpu()
            counts_host = output_bin_counts.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, "  grid: (cdiv(prefill_len, 1024),)")
            log(f, "  BLOCK_SIZE: 1024")

            # ── Reference ────────────────────────────────────────────────────
            ref_mask = torch.zeros(num_words, dtype=torch.int32)
            for t in tokens_ref[:PROMPT_LEN].cpu().tolist():
                ref_mask[t // 32] |= (1 << (t % 32))
            ref_counts = torch.zeros(VOCAB_SIZE, dtype=torch.int32)
            for t in tokens_ref[PROMPT_LEN:PREFILL_LEN].cpu().tolist():
                ref_counts[t] += 1

            diff_mask = (mask_host - ref_mask).abs().max().item()
            diff_counts = (counts_host - ref_counts).abs().max().item()
            log(f, "\nOutput:")
            log(f, f"  prompt bits set: {int((counts_host >= 0).sum() and bin(int(mask_host.sum())) and sum(bin(int(w)).count('1') for w in mask_host.tolist()))}")
            log(f, f"  output_bin_counts nonzero: {int((counts_host > 0).sum())}")
            log(f, "\nValidation (vs PyTorch prompt-mask + output-counts):")
            log(f, f"  prompt_bin_mask max abs diff: {diff_mask}")
            log(f, f"  output_bin_counts max abs diff: {diff_counts}")

            torch.testing.assert_close(mask_host, ref_mask, atol=0, rtol=0)
            torch.testing.assert_close(counts_host, ref_counts, atol=0, rtol=0)
            log(f, "  assert_close: PASSED")

            log(f, "\nSummary:")
            log(f, "  Kernel execution: SUCCESS")
            log(f, "  Validation vs PyTorch reference: PASSED")
            log(f, f"  max abs diff: mask={diff_mask}, counts={diff_counts}")

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
