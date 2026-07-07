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

from vllm.v1.worker.gpu.structured_outputs import _apply_grammar_bitmask_kernel
from vllm.triton_utils import triton

# ── Configuration ────────────────────────────────────────────────────────────
# The kernel sets logits[token] = -inf for every token whose grammar bit is 0.
NUM_MASKS  = 4            # number of (logits row, bitmask) pairs
VOCAB_SIZE = 512
DEVICE     = "qaic"

KERNEL_NAME = "_apply_grammar_bitmask_kernel"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_apply_grammar_bitmask_{timestamp}.log")


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    torch.manual_seed(42)

    logits = torch.randn(NUM_MASKS, VOCAB_SIZE, dtype=torch.float32, device=DEVICE)
    logits_indices = torch.arange(NUM_MASKS, dtype=torch.int32, device=DEVICE)

    # bitmask: int32 packed, shape [num_masks, cdiv(vocab,32)]. bit==1 => allowed.
    num_words = (VOCAB_SIZE + 31) // 32
    # Build a random allow-mask per row, pack into int32.
    allow = (torch.rand(NUM_MASKS, VOCAB_SIZE) > 0.5)
    bitmask = torch.zeros(NUM_MASKS, num_words, dtype=torch.int32)
    for r in range(NUM_MASKS):
        for v in range(VOCAB_SIZE):
            if allow[r, v]:
                bitmask[r, v // 32] |= (1 << (v % 32))
    bitmask = bitmask.to(DEVICE)

    logits_ref = logits.clone()
    allow_ref = allow.clone()

    BLOCK_SIZE = 8192
    grid = (NUM_MASKS, triton.cdiv(VOCAB_SIZE, BLOCK_SIZE))

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")
            log(f, "\nInputs:")
            log(f, f"  logits  : shape={tuple(logits.shape)}, dtype={logits.dtype}, device={logits.device}")
            log(f, f"  bitmask : shape={tuple(bitmask.shape)}, dtype={bitmask.dtype}")
            log(f, f"  num_masks={NUM_MASKS}, vocab_size={VOCAB_SIZE}")

            _apply_grammar_bitmask_kernel[grid](
                logits,
                logits.stride(0),
                logits_indices,
                bitmask,
                bitmask.stride(0),
                VOCAB_SIZE,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            out_host = logits.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}  (num_masks, cdiv(vocab, BLOCK_SIZE))")
            log(f, f"  BLOCK_SIZE: {BLOCK_SIZE}")

            # ── Reference: disallowed tokens => -inf, allowed => unchanged ────
            errors = 0
            for r in range(NUM_MASKS):
                got = out_host[r]
                for v in range(VOCAB_SIZE):
                    if allow_ref[r, v]:
                        if got[v].item() != logits_ref[r, v].cpu().item():
                            errors += 1
                    else:
                        if not torch.isinf(got[v]) or got[v] > 0:
                            errors += 1
            log(f, "\nOutput:")
            log(f, f"  -inf count: {int(torch.isinf(out_host).sum())} / {out_host.numel()}")
            log(f, f"  allowed (expected) per row: {allow_ref.sum(dim=1).tolist()}")
            log(f, "\nValidation (disallowed -> -inf, allowed unchanged):")
            log(f, f"  errors: {errors}")

            assert errors == 0, f"{errors} mismatches"
            log(f, "  validation: PASSED")

            log(f, "\nSummary:")
            log(f, f"  Kernel execution: SUCCESS (grid {grid})")
            log(f, "  Validation vs PyTorch reference: PASSED")
            log(f, f"  errors: {errors}")

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
