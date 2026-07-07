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

from vllm.v1.spec_decode.utils import eagle_prepare_next_token_padded_kernel
from vllm.triton_utils import triton

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "eagle_prepare_next_token_padded_kernel"


def main(log_path):
    torch.manual_seed(42)
    num_reqs = 4
    num_spec_per_req = 4          # num_spec_tokens + 1
    vocab_size = 1000

    # sampled_token_ids: [num_reqs, num_spec_per_req]; -1 = rejected
    sampled = torch.tensor([
        [10, 20, 30, -1],   # 3 valid
        [11, -1, -1, -1],   # 1 valid
        [12, 13, 14, 15],   # 4 valid
        [99, 98, -1, -1],   # discarded below -> backup
    ], dtype=torch.int64, device=DEVICE)
    discard = torch.tensor([0, 0, 0, 1], dtype=torch.int32, device=DEVICE)
    backup = torch.tensor([500, 501, 502, 503], dtype=torch.int64, device=DEVICE)

    next_tok = torch.full((num_reqs,), -1, dtype=torch.int64, device=DEVICE)
    valid_cnt = torch.full((num_reqs,), -1, dtype=torch.int32, device=DEVICE)

    s_r, d_r, b_r = sampled.clone(), discard.clone(), backup.clone()
    BLOCK = triton.next_power_of_2(num_spec_per_req)
    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(os.path.basename(log_path)[:-4]); log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  sampled_token_ids=\n{sampled.cpu().tolist()}")
            log(f"  discard_mask={discard.cpu().tolist()}, backup={backup.cpu().tolist()}")
            log(f"  num_reqs={num_reqs}, num_sampled_per_req={num_spec_per_req}, vocab={vocab_size}")

            eagle_prepare_next_token_padded_kernel[(num_reqs,)](
                sampled, discard, backup, next_tok, valid_cnt,
                vocab_size, num_spec_per_req, num_reqs, sampled.stride(0),
                BLOCK_SIZE_TOKENS=BLOCK,
            )
            nt = next_tok.cpu(); vc = valid_cnt.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: ({num_reqs},), BLOCK_SIZE_TOKENS={BLOCK}")
            log("\nOutput:")
            log(f"  next_token_ids: {nt.tolist()}")
            log(f"  valid_sampled_count: {vc.tolist()}")

            ref_nt = torch.empty(num_reqs, dtype=torch.int64)
            ref_vc = torch.empty(num_reqs, dtype=torch.int32)
            for r in range(num_reqs):
                if int(d_r[r]):
                    ref_nt[r] = int(b_r[r]); ref_vc[r] = 0; continue
                valids = [(j, int(s_r[r, j])) for j in range(num_spec_per_req)
                          if int(s_r[r, j]) != -1 and int(s_r[r, j]) < vocab_size]
                ref_vc[r] = len(valids)
                ref_nt[r] = valids[-1][1] if valids else int(b_r[r])
            d1 = (nt - ref_nt).abs().max().item(); d2 = (vc - ref_vc).abs().max().item()
            log("\nValidation:")
            log(f"  ref next_token={ref_nt.tolist()}, ref valid_count={ref_vc.tolist()}")
            log(f"  max abs diff: next_tok={d1}, valid_cnt={d2}")
            if d1 == 0 and d2 != 0:
                log("  NOTE: next_token_ids correct, but valid_sampled_count is wrong")
                log("        (kernel output is ~2x reference). This is a SILENT MISCOMPILE")
                log("        of the tl.sum reduction over the token mask on QAIC, matching")
                log("        the _ranks_kernel streaming-reduction miscount.")
            torch.testing.assert_close(nt, ref_nt, atol=0, rtol=0)
            torch.testing.assert_close(vc, ref_vc, atol=0, rtol=0)
            log("  assert_close: PASSED")
            log("\nSummary:")
            log(f"  Kernel execution: SUCCESS (grid ({num_reqs},))")
            log("  Validation vs PyTorch reference: PASSED")
            log(f"  max abs diff: next_tok={d1}, valid_cnt={d2}")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Validation vs PyTorch reference: FAILED")
            fp.write("\n" + "-" * 36 + "\n"); sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "eagle_prepare_next_token_padded")
