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

from vllm.v1.sample.rejection_sampler import rejection_random_sample_kernel, PLACEHOLDER_TOKEN_ID

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "rejection_random_sample_kernel"


def main(log_path):
    max_spec = 3
    num_draft = [3, 2]
    cu = torch.tensor(num_draft, dtype=torch.int32, device=DEVICE).cumsum(0).to(torch.int32)
    total = sum(num_draft)
    vocab = 16

    draft = torch.tensor([1, 2, 3,  4, 5], dtype=torch.int64, device=DEVICE)
    # target_probs / draft_probs (NO_DRAFT_PROBS=False path)
    target_probs = torch.rand(total, vocab, dtype=torch.float32, device=DEVICE)
    target_probs = target_probs / target_probs.sum(-1, keepdim=True)
    draft_probs = torch.rand(total, vocab, dtype=torch.float32, device=DEVICE)
    draft_probs = draft_probs / draft_probs.sum(-1, keepdim=True)
    bonus = torch.tensor([777, 888], dtype=torch.int64, device=DEVICE)
    recovered = torch.tensor([9, 9, 9, 9, 9], dtype=torch.int64, device=DEVICE)
    # uniform_probs: QAIC has no float64/Double support; use float32.
    uniform = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=DEVICE)
    is_greedy = torch.tensor([0, 0], dtype=torch.int32, device=DEVICE)

    out = torch.full((2, max_spec + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32, device=DEVICE)

    cu_r, d_r, tp_r, dp_r, b_r, rec_r, u_r = (cu.clone(), draft.clone(),
        target_probs.clone(), draft_probs.clone(), bonus.clone(),
        recovered.clone(), uniform.clone())
    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(os.path.basename(log_path)[:-4]); log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  num_draft={num_draft}, cu={cu.cpu().tolist()}, vocab={vocab}")
            log(f"  draft={draft.cpu().tolist()}, recovered={recovered.cpu().tolist()}")
            log(f"  uniform_probs={uniform.cpu().tolist()} (0 => always accept when t/d>=0)")
            log(f"  bonus={bonus.cpu().tolist()}, max_spec_len={max_spec}")

            rejection_random_sample_kernel[(2,)](
                out, cu, draft, draft_probs, target_probs, bonus, recovered,
                uniform, is_greedy, max_spec, vocab, NO_DRAFT_PROBS=False,
            )
            o = out.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log("  grid: (batch=2,), NO_DRAFT_PROBS=False")
            log("\nOutput:")
            log(f"  output_token_ids=\n{o.tolist()}")

            # reference
            ref = [[PLACEHOLDER_TOKEN_ID] * (max_spec + 1) for _ in range(2)]
            for r in range(2):
                st = 0 if r == 0 else int(cu_r[r - 1]); en = int(cu_r[r]); nt = en - st
                rejected = False
                for pos in range(nt):
                    if not rejected:
                        did = int(d_r[st + pos])
                        dp = float(dp_r[st + pos, did]); tp = float(tp_r[st + pos, did])
                        up = float(u_r[st + pos])
                        if dp > 0 and tp / dp >= up:
                            tok = did
                        else:
                            rejected = True; tok = int(rec_r[st + pos])
                        ref[r][pos] = tok
                if not rejected:
                    ref[r][nt] = int(b_r[r])
            errs = sum(1 for r in range(2) for j in range(max_spec + 1) if int(o[r, j]) != ref[r][j])
            log("\nValidation:")
            log(f"  reference=\n{ref}")
            log(f"  mismatches: {errs}")
            assert errs == 0
            log("  validation: PASSED")
            log("\nSummary:")
            log("  Kernel execution: SUCCESS (grid (2,))")
            log("  Validation vs PyTorch reference: PASSED")
            log(f"  mismatches: {errs}")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Validation vs PyTorch reference: FAILED")
            fp.write("\n" + "-" * 36 + "\n"); sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "rejection_random_sample")
