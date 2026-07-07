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

from vllm.v1.worker.gpu.spec_decode.rejection_sample import rejection_sample

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "_rejection_sample_kernel"


def main(log_path):
    # Per req: cu_num_logits gives [start,end); the kernel accepts target tokens
    # until target != draft (draft is input_ids shifted by 1), then stops.
    num_spec = 3
    # 2 requests. logits layout: [draft0..draftN, bonus] per req.
    # req0: 3 draft + 1 bonus = 4 logits ; req1: 2 draft + 1 = 3 logits
    cu = torch.tensor([0, 4, 7], dtype=torch.int32, device=DEVICE)
    # target sampled token at each logit position
    target = torch.tensor([100, 101, 102, 103,  200, 201, 202], dtype=torch.int64, device=DEVICE)
    # input_ids (draft) — index i+1 is the draft proposed after position i
    # req0: drafts at positions 1,2,3 => [_, 101, 999, 103] (mismatch at pos1->idx2)
    input_ids = torch.tensor([0, 101, 999, 103,  0, 201, 202], dtype=torch.int64, device=DEVICE)

    cu_r, t_r, i_r = cu.clone(), target.clone(), input_ids.clone()
    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(os.path.basename(log_path)[:-4]); log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  cu_num_logits={cu.cpu().tolist()}")
            log(f"  target_sampled={target.cpu().tolist()}")
            log(f"  input_ids(draft)={input_ids.cpu().tolist()}")
            log(f"  num_speculative_steps={num_spec}")

            sampled, num_sampled = rejection_sample(target, input_ids, cu, num_spec)
            s = sampled.cpu(); ns = num_sampled.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log("  grid: (num_reqs=2,), num_warps=1")
            log("\nOutput:")
            log(f"  sampled=\n{s.tolist()}")
            log(f"  num_sampled={ns.tolist()}")

            # reference
            num_reqs = cu.shape[0] - 1
            ref_ns = []
            ref_s = [[-1] * (num_spec + 1) for _ in range(num_reqs)]
            for r in range(num_reqs):
                st = int(cu_r[r]); en = int(cu_r[r + 1]); nt = en - st
                cnt = 0; rejected = False
                for i in range(nt - 1):
                    if not rejected:
                        ts = int(t_r[st + i]); ds = int(i_r[st + i + 1])
                        ref_s[r][i] = ts; cnt += 1
                        if ts != ds:
                            rejected = True
                if not rejected:
                    ref_s[r][nt - 1] = int(t_r[st + nt - 1]); cnt += 1
                ref_ns.append(cnt)
            ok = (ns.tolist() == ref_ns)
            # compare only the meaningful prefix per row
            errs = 0
            for r in range(num_reqs):
                for j in range(ref_ns[r]):
                    if int(s[r, j]) != ref_s[r][j]:
                        errs += 1
            log("\nValidation:")
            log(f"  ref num_sampled={ref_ns}, ref sampled={ref_s}")
            log(f"  num_sampled match: {ok}, sampled-prefix errors: {errs}")
            assert ok and errs == 0, "rejection-sample mismatch"
            log("  validation: PASSED")
            log("\nSummary:")
            log("  Kernel execution: SUCCESS (grid (2,))")
            log("  Validation vs PyTorch reference: PASSED")
            log(f"  num_sampled match: {ok}, errors: {errs}")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Validation vs PyTorch reference: FAILED")
            fp.write("\n" + "-" * 36 + "\n"); sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "rejection_sample_kernel")
