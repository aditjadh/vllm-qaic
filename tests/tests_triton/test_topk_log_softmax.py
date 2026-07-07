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

from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs
from vllm.triton_utils import triton

# ── Configuration ────────────────────────────────────────────────────────────
BATCH_SIZE   = 8
VOCAB_SIZE   = 4096
TOPK         = 5            # num_logprobs (tokens per row to score)
BLOCK_SIZE   = 1024
DTYPE        = torch.float16
DEVICE       = "qaic"

KERNEL_NAME = "_topk_log_softmax_kernel"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_topk_log_softmax_{timestamp}.log")


def reference_token_logprobs(logits, token_ids):
    """Pure-PyTorch reference: log_softmax over vocab, gathered at token_ids."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    return torch.gather(logp, 1, token_ids.to(torch.int64))


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    torch.manual_seed(42)

    logits = torch.randn(BATCH_SIZE, VOCAB_SIZE, dtype=DTYPE, device=DEVICE)
    # Each row scores TOPK distinct token ids (its own argmax topk here).
    token_ids = torch.topk(logits, TOPK, dim=-1).indices.to(torch.int64)

    logits_ref    = logits.clone()
    token_ids_ref = token_ids.clone()

    grid = (BATCH_SIZE,)

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")

            log(f, "\nInputs:")
            log(f, f"  logits    : shape={tuple(logits.shape)}, dtype={logits.dtype}, device={logits.device}")
            log(f, f"  token_ids : shape={tuple(token_ids.shape)}, dtype={token_ids.dtype}")
            log(f, f"  batch_size={BATCH_SIZE}, vocab_size={VOCAB_SIZE}, topk={TOPK}")
            log(f, f"  BLOCK_SIZE={BLOCK_SIZE}, PADDED_TOPK={triton.next_power_of_2(TOPK)}")

            out = compute_token_logprobs(logits, token_ids)

            # Force a device sync so any kernel compile/exec failure is raised
            # here rather than later at the first device read.
            out_host = out.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}  (batch_size,)")
            log(f, f"  BLOCK_SIZE: {BLOCK_SIZE}, PADDED_TOPK: {triton.next_power_of_2(TOPK)}")

            log(f, "\nOutput:")
            log(f, f"  logprobs : shape={tuple(out.shape)}, min={out_host.min().item():.4f}, "
                   f"max={out_host.max().item():.4f}, mean={out_host.float().mean().item():.4f}")

            # ── Reference validation ────────────────────────────────────────
            ref = reference_token_logprobs(logits_ref, token_ids_ref)
            diff = (out_host.float() - ref.cpu()).abs().max().item()
            log(f, "\nValidation (vs PyTorch log_softmax gathered at token_ids):")
            log(f, f"  max abs diff: {diff:.6f}")

            torch.testing.assert_close(out_host.float(), ref.cpu(), atol=1e-2, rtol=1.6e-2)
            log(f, "  assert_close: PASSED")

            log(f, "\nSummary:")
            log(f, f"  Kernel execution: SUCCESS (grid {grid})")
            log(f, "  Validation vs PyTorch reference: PASSED")
            log(f, f"  max abs diff: {diff:.6f}")

        except Exception:
            msg = f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}"
            log(f, msg)
            log(f, "\nSummary:")
            log(f, "  Validation vs PyTorch reference: FAILED")
            sys.exit(1)

        finally:
            log(f, "\n" + "-" * 36)

    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
