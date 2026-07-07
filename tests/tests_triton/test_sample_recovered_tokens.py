# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vllm"))
from _ktest import run_test, logger

from vllm.v1.sample.rejection_sampler import sample_recovered_tokens_kernel
from vllm.triton_utils import triton

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "sample_recovered_tokens_kernel"


def main(log_path):
    torch.manual_seed(42)
    max_spec = 3
    num_draft = [3, 2]
    cu = torch.tensor(num_draft, dtype=torch.int32, device=DEVICE).cumsum(0).to(torch.int32)
    total = sum(num_draft)
    vocab = 16

    draft = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64, device=DEVICE)
    target_probs = torch.rand(total, vocab, dtype=torch.float32, device=DEVICE)
    draft_probs = torch.rand(total, vocab, dtype=torch.float32, device=DEVICE)
    # q ~ exponential; use fixed positive values for determinism
    q = torch.rand(2, vocab, dtype=torch.float32, device=DEVICE) + 0.1
    out = torch.empty(total, dtype=torch.int64, device=DEVICE)

    cu_r, d_r, tp_r, dp_r, q_r = (cu.clone(), draft.clone(),
        target_probs.clone(), draft_probs.clone(), q.clone())
    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(os.path.basename(log_path)[:-4]); log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  num_draft={num_draft}, cu={cu.cpu().tolist()}, vocab={vocab}")
            log(f"  draft={draft.cpu().tolist()}")
            log(f"  max_spec_len={max_spec}, NO_DRAFT_PROBS=False")

            sample_recovered_tokens_kernel[(2, max_spec)](
                out, cu, draft, draft_probs, target_probs, q,
                vocab, triton.next_power_of_2(vocab), NO_DRAFT_PROBS=False,
            )
            o = out.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: (batch=2, max_spec_len={max_spec})")
            log("\nOutput:")
            log(f"  recovered_token_ids={o.tolist()}")

            # reference: argmax over (max(target-draft,0) / q)
            ref = torch.full((total,), -1, dtype=torch.int64)
            for r in range(2):
                st = 0 if r == 0 else int(cu_r[r - 1]); en = int(cu_r[r]); nt = en - st
                for pos in range(nt):
                    prob = torch.clamp(tp_r[st + pos].cpu() - dp_r[st + pos].cpu(), min=0)
                    score = prob / q_r[r].cpu()
                    ref[st + pos] = int(torch.argmax(score))
            # compare only valid positions
            errs = 0
            for r in range(2):
                st = 0 if r == 0 else int(cu_r[r - 1]); en = int(cu_r[r])
                for idx in range(st, en):
                    if int(o[idx]) != int(ref[idx]):
                        errs += 1
            log("\nValidation:")
            log(f"  reference={ref.tolist()}")
            log(f"  mismatches: {errs}")
            assert errs == 0
            log("  validation: PASSED")
            log("\nSummary:")
            log("  Kernel execution: SUCCESS (grid (2, 3))")
            log("  Validation vs PyTorch reference: PASSED")
            log(f"  mismatches: {errs}")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Validation vs PyTorch reference: FAILED")
            fp.write("\n" + "-" * 36 + "\n"); sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "sample_recovered_tokens")
