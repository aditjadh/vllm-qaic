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

from vllm.v1.sample.rejection_sampler import expand_kernel, MAX_SPEC_LEN

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "expand_kernel"


def main(log_path):
    # expand [batch] -> [num_tokens] by repeating each value cu_num_tokens times,
    # with optional replace_from/replace_to.
    batch = 4
    counts = [2, 3, 1, 4]
    cu = torch.tensor(counts, dtype=torch.int32, device=DEVICE).cumsum(0).to(torch.int32)
    total = sum(counts)
    inp = torch.tensor([10, 0, 30, 40], dtype=torch.int32, device=DEVICE)  # 0 -> replace
    out = torch.full((total,), -1, dtype=torch.int32, device=DEVICE)
    replace_from, replace_to = 0, 99

    inp_r, cu_r = inp.clone(), cu.clone()
    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(os.path.basename(log_path)[:-4]); log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  input(batch)={inp.cpu().tolist()}, counts={counts}, cu={cu.cpu().tolist()}")
            log(f"  replace_from={replace_from}, replace_to={replace_to}, total={total}")

            expand_kernel[(batch,)](out, inp, cu, replace_from, replace_to,
                                    MAX_NUM_TOKENS=MAX_SPEC_LEN)
            o = out.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: ({batch},), MAX_NUM_TOKENS={MAX_SPEC_LEN}")
            log("\nOutput:")
            log(f"  expanded={o.tolist()}")

            ref = torch.empty(total, dtype=torch.int32)
            for b in range(batch):
                st = 0 if b == 0 else int(cu_r[b - 1]); en = int(cu_r[b])
                v = int(inp_r[b]); v = replace_to if v == replace_from else v
                ref[st:en] = v
            diff = (o - ref).abs().max().item()
            log("\nValidation:")
            log(f"  reference={ref.tolist()}")
            log(f"  max abs diff: {diff}")
            assert diff == 0
            log("  validation: PASSED")
            log("\nSummary:")
            log(f"  Kernel execution: SUCCESS (grid ({batch},))")
            log("  Validation vs PyTorch reference: PASSED")
            log(f"  max abs diff: {diff}")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Validation vs PyTorch reference: FAILED")
            fp.write("\n" + "-" * 36 + "\n"); sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "expand_kernel")
