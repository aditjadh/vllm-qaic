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

from vllm.v1.worker.gpu.input_batch import post_update

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "_post_update_kernel"


def main(log_path):
    num_reqs = 2
    max_reqs = 4
    vocab = 32
    num_spec = 3

    idx_mapping = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    num_computed = torch.tensor([10, 20, 0, 0], dtype=torch.int32, device=DEVICE)
    last_sampled = torch.full((max_reqs,), -1, dtype=torch.int64, device=DEVICE)
    output_bin_counts = torch.zeros(max_reqs, vocab, dtype=torch.int32, device=DEVICE)
    # sampled tokens per req (padded to num_spec+1)
    sampled = torch.tensor([[5, 6, 7, -1], [8, 9, -1, -1]], dtype=torch.int64, device=DEVICE)
    num_sampled = torch.tensor([3, 2], dtype=torch.int32, device=DEVICE)
    num_rejected = torch.tensor([1, 2], dtype=torch.int32, device=DEVICE)
    qsl = torch.tensor([0, 4, 8], dtype=torch.int32, device=DEVICE)

    nc_r, s_r, ns_r, nr_r, qsl_r = (num_computed.clone(), sampled.clone(),
        num_sampled.clone(), num_rejected.clone(), qsl.clone())
    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(os.path.basename(log_path)[:-4]); log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  num_computed_tokens={num_computed.cpu().tolist()}")
            log(f"  sampled_tokens={sampled.cpu().tolist()}, num_sampled={num_sampled.cpu().tolist()}")
            log(f"  num_rejected={num_rejected.cpu().tolist()}, query_start_loc={qsl.cpu().tolist()}")

            post_update(idx_mapping, num_computed, last_sampled, output_bin_counts,
                        sampled, num_sampled, num_rejected, qsl)
            ls = last_sampled.cpu(); obc = output_bin_counts.cpu(); nc = num_computed.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: ({num_reqs},), num_warps=1")
            log("\nOutput:")
            log(f"  last_sampled_tokens={ls.tolist()}")
            log(f"  num_computed_tokens={nc.tolist()}")
            nz = [(r, t, int(obc[r, t])) for r in range(max_reqs) for t in range(vocab) if obc[r, t] > 0]
            log(f"  output_bin_counts nonzero (req,tok,count): {nz}")

            # reference
            ref_ls = [-1, -1, -1, -1]
            ref_nc = nc_r.cpu().tolist()
            ref_obc = torch.zeros(max_reqs, vocab, dtype=torch.int32)
            for b in range(num_reqs):
                req = int(idx_mapping[b]); nsamp = int(ns_r[b])
                if nsamp > 0:
                    ref_ls[req] = int(s_r[b, nsamp - 1])
                for i in range(nsamp):
                    ref_obc[req, int(s_r[b, i])] += 1
                ql = int(qsl_r[b + 1]) - int(qsl_r[b])
                ref_nc[req] += ql - int(nr_r[b])
            d_ls = max(abs(int(ls[i]) - ref_ls[i]) for i in range(max_reqs))
            d_nc = max(abs(int(nc[i]) - ref_nc[i]) for i in range(max_reqs))
            d_obc = (obc - ref_obc).abs().max().item()
            log("\nValidation:")
            log(f"  ref last_sampled={ref_ls}, ref num_computed={ref_nc}")
            log(f"  max abs diff: last_sampled={d_ls}, num_computed={d_nc}, bin_counts={d_obc}")
            assert d_ls == 0 and d_nc == 0 and d_obc == 0
            log("  validation: PASSED")
            log("\nSummary:")
            log(f"  Kernel execution: SUCCESS (grid ({num_reqs},))")
            log("  Validation vs PyTorch reference: PASSED")
            log(f"  max abs diff: ls={d_ls}, nc={d_nc}, obc={d_obc}")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Validation vs PyTorch reference: FAILED")
            fp.write("\n" + "-" * 36 + "\n"); sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "post_update")
