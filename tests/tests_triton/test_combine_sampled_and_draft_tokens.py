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

from vllm.v1.worker.gpu.input_batch import combine_sampled_and_draft_tokens

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "_combine_sampled_and_draft_tokens_kernel"


def main(log_path):
    num_reqs = 2
    num_spec = 2          # speculative steps -> draft_tokens [:, num_spec]
    # logits per req = 1 (sampled) + num_draft. Here both reqs decode (not prefill).
    num_logits_per = [3, 2]    # req0: 1 sampled + 2 draft ; req1: 1 + 1
    cu = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    cu[1:] = torch.tensor(num_logits_per, dtype=torch.int32).cumsum(0)
    total_logits = int(cu[-1])

    qlens = num_logits_per   # query covers the logits region here
    qsl = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    qsl[1:] = torch.tensor(qlens, dtype=torch.int32).cumsum(0)
    num_tokens = int(qsl[-1])

    idx_mapping = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    last_sampled = torch.tensor([700, 800, 0, 0], dtype=torch.int64, device=DEVICE)
    seq_lens = torch.tensor([20, 15], dtype=torch.int32, device=DEVICE)
    prefill_len = torch.tensor([10, 5, 0, 0], dtype=torch.int32, device=DEVICE)  # seq>prefill => decode
    draft = torch.tensor([[101, 102], [201, 202]], dtype=torch.int64, device=DEVICE)
    input_ids = torch.full((num_tokens,), -1, dtype=torch.int64, device=DEVICE)

    ls_r, d_r, qsl_r, cu_r = (last_sampled.clone(), draft.clone(), qsl.clone(), cu.clone())
    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(os.path.basename(log_path)[:-4]); log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  num_logits_per_req={num_logits_per}, cu_num_logits={cu.cpu().tolist()}")
            log(f"  query_start_loc={qsl.cpu().tolist()}, seq_lens={seq_lens.cpu().tolist()}")
            log(f"  prefill_len={prefill_len.cpu().tolist()}, last_sampled={last_sampled.cpu().tolist()}")
            log(f"  draft_tokens={draft.cpu().tolist()}")

            logits_indices = combine_sampled_and_draft_tokens(
                input_ids, idx_mapping, last_sampled, qsl, seq_lens,
                prefill_len, draft, cu, total_logits,
            )
            ii = input_ids.cpu(); li = logits_indices.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: ({num_reqs},), BLOCK_SIZE=next_pow2({num_spec + 1})")
            log("\nOutput:")
            log(f"  input_ids={ii.tolist()}")
            log(f"  logits_indices={li.tolist()}")

            # reference
            ref_li = [0] * total_logits
            ref_ii = [-1] * num_tokens
            for b in range(num_reqs):
                nl = int(cu_r[b + 1]) - int(cu_r[b]); ndt = nl - 1
                qe = int(qsl_r[b + 1])
                lstart = qe - nl
                for j in range(nl):
                    ref_li[int(cu_r[b]) + j] = lstart + j
                # decode (seq>prefill): write last sampled + drafts
                ref_ii[qe - nl] = int(ls_r[int(idx_mapping[b])])
                for j in range(ndt):
                    ref_ii[qe - ndt + j] = int(d_r[int(idx_mapping[b]), j])
            d_li = max(abs(int(li[i]) - ref_li[i]) for i in range(total_logits))
            errs_ii = sum(1 for i in range(num_tokens) if ref_ii[i] != -1 and int(ii[i]) != ref_ii[i])
            log("\nValidation:")
            log(f"  ref logits_indices={ref_li}")
            log(f"  ref input_ids={ref_ii}")
            log(f"  logits_indices max abs diff: {d_li}, input_ids mismatches: {errs_ii}")
            assert d_li == 0 and errs_ii == 0
            log("  validation: PASSED")
            log("\nSummary:")
            log(f"  Kernel execution: SUCCESS (grid ({num_reqs},))")
            log("  Validation vs PyTorch reference: PASSED")
            log(f"  logits_indices diff: {d_li}, input_ids mismatches: {errs_ii}")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Validation vs PyTorch reference: FAILED")
            fp.write("\n" + "-" * 36 + "\n"); sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "combine_sampled_and_draft_tokens")
