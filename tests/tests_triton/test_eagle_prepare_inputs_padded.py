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

from vllm.v1.spec_decode.utils import eagle_prepare_inputs_padded_kernel

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "eagle_prepare_inputs_padded_kernel"


def main(log_path):
    torch.manual_seed(42)
    num_reqs = 4
    # cu_num_draft_tokens: inclusive cumsum of draft tokens per req
    num_draft = [3, 0, 2, 3]
    cu_draft = torch.tensor(num_draft, dtype=torch.int32, device=DEVICE).cumsum(0).to(torch.int32)
    # valid sampled = accepted + 1
    valid_count = torch.tensor([2, 1, 3, 4], dtype=torch.int32, device=DEVICE)
    # query_start_loc: each req's query covers its draft tokens + 1
    qlens = [d + 1 for d in num_draft]
    qsl = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    qsl[1:] = torch.tensor(qlens, dtype=torch.int32).cumsum(0)

    tok_idx = torch.full((num_reqs,), -1, dtype=torch.int64, device=DEVICE)
    num_rej = torch.full((num_reqs,), -1, dtype=torch.int32, device=DEVICE)

    cu_r, vc_r, qsl_r = cu_draft.clone(), valid_count.clone(), qsl.clone()
    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(os.path.basename(log_path)[:-4]); log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  num_draft_tokens={num_draft}, cu_num_draft={cu_draft.cpu().tolist()}")
            log(f"  valid_sampled_count={valid_count.cpu().tolist()}, query_start_loc={qsl.cpu().tolist()}")

            eagle_prepare_inputs_padded_kernel[(num_reqs,)](
                cu_draft, valid_count, qsl, tok_idx, num_rej, num_reqs,
            )
            ti = tok_idx.cpu(); nr = num_rej.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: ({num_reqs},)")
            log("\nOutput:")
            log(f"  token_indices_to_sample: {ti.tolist()}")
            log(f"  num_rejected_tokens: {nr.tolist()}")

            ref_ti = torch.empty(num_reqs, dtype=torch.int64)
            ref_nr = torch.empty(num_reqs, dtype=torch.int32)
            for r in range(num_reqs):
                ndt = int(cu_r[r]) if r == 0 else int(cu_r[r]) - int(cu_r[r - 1])
                vc = int(vc_r[r])
                nrej = ndt + 1 - vc if ndt > 0 else 0
                ref_nr[r] = nrej
                ref_ti[r] = int(qsl_r[r + 1]) - 1 - nrej
            d1 = (ti - ref_ti).abs().max().item(); d2 = (nr - ref_nr).abs().max().item()
            log("\nValidation:")
            log(f"  ref token_indices: {ref_ti.tolist()}, ref num_rejected: {ref_nr.tolist()}")
            log(f"  max abs diff: tok_idx={d1}, num_rej={d2}")
            torch.testing.assert_close(ti, ref_ti, atol=0, rtol=0)
            torch.testing.assert_close(nr, ref_nr, atol=0, rtol=0)
            log("  assert_close: PASSED")
            log("\nSummary:")
            log(f"  Kernel execution: SUCCESS (grid ({num_reqs},))")
            log("  Validation vs PyTorch reference: PASSED")
            log(f"  max abs diff: tok_idx={d1}, num_rej={d2}")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Validation vs PyTorch reference: FAILED")
            fp.write("\n" + "-" * 36 + "\n"); sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "eagle_prepare_inputs_padded")
