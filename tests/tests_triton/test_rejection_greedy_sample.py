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

from vllm.v1.sample.rejection_sampler import rejection_greedy_sample_kernel, PLACEHOLDER_TOKEN_ID

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "rejection_greedy_sample_kernel"


def main(log_path):
    # 2 reqs, max_spec_len=3. cu_num_draft_tokens inclusive cumsum.
    max_spec = 3
    num_draft = [3, 2]
    cu = torch.tensor(num_draft, dtype=torch.int32, device=DEVICE).cumsum(0).to(torch.int32)
    total = sum(num_draft)
    # draft tokens per position
    draft = torch.tensor([10, 11, 12,  20, 21], dtype=torch.int64, device=DEVICE)
    # target argmax per position; req0 mismatches at pos1, req1 all match
    target_argmax = torch.tensor([10, 99, 12,  20, 21], dtype=torch.int64, device=DEVICE)
    bonus = torch.tensor([777, 888], dtype=torch.int64, device=DEVICE)
    is_greedy = torch.tensor([1, 1], dtype=torch.int32, device=DEVICE)

    out = torch.full((2, max_spec + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32, device=DEVICE)

    cu_r, d_r, ta_r, b_r = cu.clone(), draft.clone(), target_argmax.clone(), bonus.clone()
    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(os.path.basename(log_path)[:-4]); log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  num_draft={num_draft}, cu={cu.cpu().tolist()}")
            log(f"  draft={draft.cpu().tolist()}, target_argmax={target_argmax.cpu().tolist()}")
            log(f"  bonus={bonus.cpu().tolist()}, max_spec_len={max_spec}")

            rejection_greedy_sample_kernel[(2,)](
                out, cu, draft, target_argmax, bonus, is_greedy, max_spec,
            )
            o = out.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log("  grid: (batch=2,)")
            log("\nOutput:")
            log(f"  output_token_ids=\n{o.tolist()}")

            # reference
            ref = [[PLACEHOLDER_TOKEN_ID] * (max_spec + 1) for _ in range(2)]
            for r in range(2):
                st = 0 if r == 0 else int(cu_r[r - 1]); en = int(cu_r[r]); nt = en - st
                rejected = False
                for pos in range(nt):
                    if not rejected:
                        ref[r][pos] = int(ta_r[st + pos])
                        if int(d_r[st + pos]) != int(ta_r[st + pos]):
                            rejected = True
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
    run_test(main, LOG_DIR, "rejection_greedy_sample")
