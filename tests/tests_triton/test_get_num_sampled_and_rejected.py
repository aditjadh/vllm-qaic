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

from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "_get_num_sampled_and_rejected_kernel"


def main(log_path):
    num_reqs = 3
    # req2 is chunked-prefilling (seq_len < prefill_len) -> num_sampled/rejected = 0
    num_sampled = torch.tensor([2, 1, 3], dtype=torch.int32, device=DEVICE)
    seq_lens = torch.tensor([20, 15, 4], dtype=torch.int32, device=DEVICE)
    num_logits_per = [3, 2, 5]
    cu = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    cu[1:] = torch.tensor(num_logits_per, dtype=torch.int32).cumsum(0)
    idx_mapping = torch.tensor([0, 1, 2], dtype=torch.int32, device=DEVICE)
    prefill_len = torch.tensor([10, 5, 8], dtype=torch.int32, device=DEVICE)

    ns_r, sl_r, cu_r, pl_r = (num_sampled.clone(), seq_lens.clone(),
                              cu.clone(), prefill_len.clone())
    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(os.path.basename(log_path)[:-4]); log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  num_sampled={num_sampled.cpu().tolist()}, seq_lens={seq_lens.cpu().tolist()}")
            log(f"  cu_num_logits={cu.cpu().tolist()}, prefill_len={prefill_len.cpu().tolist()}")
            log("  req2: seq_len(4) < prefill_len(8) => chunked prefill -> zeros")

            ns_out, nr_out = get_num_sampled_and_rejected(
                num_sampled, seq_lens, cu, idx_mapping, prefill_len,
            )
            ns = ns_out.cpu(); nr = nr_out.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: ({num_reqs},)")
            log("\nOutput:")
            log(f"  num_sampled(updated)={ns.tolist()}")
            log(f"  num_rejected={nr.tolist()}")

            ref_ns = []; ref_nr = []
            for b in range(num_reqs):
                chunk = int(sl_r[b]) < int(pl_r[b])
                nsamp = 0 if chunk else int(ns_r[b])
                nl = int(cu_r[b + 1]) - int(cu_r[b])
                nrej = 0 if chunk else nl - nsamp
                ref_ns.append(nsamp); ref_nr.append(nrej)
            d1 = (ns - torch.tensor(ref_ns, dtype=torch.int32)).abs().max().item()
            d2 = (nr - torch.tensor(ref_nr, dtype=torch.int32)).abs().max().item()
            log("\nValidation:")
            log(f"  ref num_sampled={ref_ns}, ref num_rejected={ref_nr}")
            log(f"  max abs diff: num_sampled={d1}, num_rejected={d2}")
            assert d1 == 0 and d2 == 0
            log("  validation: PASSED")
            log("\nSummary:")
            log(f"  Kernel execution: SUCCESS (grid ({num_reqs},))")
            log("  Validation vs PyTorch reference: PASSED")
            log(f"  max abs diff: ns={d1}, nr={d2}")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Validation vs PyTorch reference: FAILED")
            fp.write("\n" + "-" * 36 + "\n"); sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "get_num_sampled_and_rejected")
