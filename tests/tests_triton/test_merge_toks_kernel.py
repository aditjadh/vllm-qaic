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

from vllm.v1.spec_decode.draft_model import merge_toks_kernel

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "merge_toks_kernel"


def main(log_path):
    # Merge target tokens with a per-req "next" (bonus) token, inserting it after
    # query_end_locs and marking trailing rejected positions.
    num_reqs = 2
    # target_toks laid out per req contiguously; query_start_locs / end_locs index it.
    target = torch.tensor([10, 11, 12,  20, 21], dtype=torch.int64, device=DEVICE)
    start_locs = torch.tensor([0, 3], dtype=torch.int32, device=DEVICE)
    end_locs = torch.tensor([2, 4], dtype=torch.int32, device=DEVICE)  # last accepted idx
    next_toks = torch.tensor([99, 88], dtype=torch.int64, device=DEVICE)

    out_size = target.shape[0] + num_reqs   # merged buffer
    merged = torch.full((out_size,), -1, dtype=torch.int64, device=DEVICE)
    is_rej = torch.zeros((out_size,), dtype=torch.bool, device=DEVICE)

    t_r, s_r, e_r, n_r = (target.clone(), start_locs.clone(),
                          end_locs.clone(), next_toks.clone())
    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(os.path.basename(log_path)[:-4]); log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  target_toks={target.cpu().tolist()}")
            log(f"  query_start_locs={start_locs.cpu().tolist()}, query_end_locs={end_locs.cpu().tolist()}")
            log(f"  next_toks={next_toks.cpu().tolist()}, target_size={target.shape[0]}")

            merge_toks_kernel[(num_reqs,)](
                t_r, n_r, s_r, e_r, merged, is_rej, target.shape[0], 0,
            )
            m = merged.cpu(); ir = is_rej.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: ({num_reqs},)")
            log("\nOutput:")
            log(f"  merged_toks={m.tolist()}")
            log(f"  is_rejected={ir.tolist()}")

            # reference mirrors kernel: for pid, i in [start, next_start]:
            #   i<=end -> copy target[i]; i==end+1 -> next; else rejected fill 0
            ref_m = [-1] * out_size
            ref_r = [False] * out_size
            for pid in range(num_reqs):
                start = int(s_r[pid])
                next_start = target.shape[0] if pid == num_reqs - 1 else int(s_r[pid + 1])
                end = int(e_r[pid]); nv = int(n_r[pid])
                for i in range(start, next_start + 1):
                    o = pid + i
                    if o >= out_size:
                        continue
                    if i <= end:
                        ref_m[o] = int(t_r[i]); ref_r[o] = False
                    elif i == end + 1:
                        ref_m[o] = nv; ref_r[o] = False
                    else:
                        ref_m[o] = 0; ref_r[o] = True
            # only compare positions the kernel wrote (non -1 in either)
            errs = 0
            for o in range(out_size):
                if ref_m[o] != -1 and int(m[o]) != ref_m[o]:
                    errs += 1
            log("\nValidation:")
            log(f"  reference merged={ref_m}")
            log(f"  reference is_rejected={ref_r}")
            log(f"  written-position mismatches: {errs}")
            assert errs == 0
            log("  validation: PASSED")
            log("\nSummary:")
            log(f"  Kernel execution: SUCCESS (grid ({num_reqs},))")
            log("  Validation vs PyTorch reference: PASSED")
            log(f"  mismatches: {errs}")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Validation vs PyTorch reference: FAILED")
            fp.write("\n" + "-" * 36 + "\n"); sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "merge_toks_kernel")
