# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
import os
import sys
import traceback
from datetime import datetime

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vllm"))

from vllm.v1.worker.gpu.input_batch import prepare_prefill_inputs

# ── Configuration ────────────────────────────────────────────────────────────
MAX_MODEL_LEN = 64
MAX_REQS      = 4
DEVICE        = "qaic"

IDX_MAPPING  = [0, 1, 2]
QUERY_LENS   = [4, 3, 5]
PREFILL_LENS = [10, 5, 12, 0]      # by req_state_idx
NUM_COMPUTED = [0, 2, 1, 0]        # req1: 2+3=5 == prefill_len(5) boundary

KERNEL_NAME = "_prepare_prefill_inputs_kernel"
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = os.path.join(LOG_DIR, f"test_prepare_prefill_inputs_{timestamp}.log")


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    torch.manual_seed(42)
    num_reqs = len(IDX_MAPPING)
    num_tokens = sum(QUERY_LENS)

    qsl = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    qsl[1:] = torch.tensor(QUERY_LENS, dtype=torch.int32).cumsum(0)
    idx_mapping = torch.tensor(IDX_MAPPING, dtype=torch.int32, device=DEVICE)
    prefill_lens = torch.tensor(PREFILL_LENS, dtype=torch.int32, device=DEVICE)
    num_computed = torch.tensor(NUM_COMPUTED, dtype=torch.int32, device=DEVICE)
    prefill_token_ids = (
        torch.arange(MAX_REQS * MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE)
        .reshape(MAX_REQS, MAX_MODEL_LEN)
    )

    input_ids = torch.full((num_tokens,), -1, dtype=torch.int32, device=DEVICE)
    next_prefill = torch.full((MAX_REQS,), -1, dtype=torch.int32, device=DEVICE)

    qsl_r, idx_r, pl_r, nc_r, pti_r = (qsl.clone(), idx_mapping.clone(),
                                       prefill_lens.clone(), num_computed.clone(),
                                       prefill_token_ids.clone())
    with open(log_path, "w") as f:
        try:
            log(f, timestamp); log(f, f"Kernel: {KERNEL_NAME}")
            log(f, "\nInputs:")
            log(f, f"  prefill_token_ids: shape={tuple(prefill_token_ids.shape)}, device={prefill_token_ids.device}")
            log(f, f"  idx_mapping={IDX_MAPPING}, query_lens={QUERY_LENS}")
            log(f, f"  prefill_lens={PREFILL_LENS}, num_computed={NUM_COMPUTED}")

            prepare_prefill_inputs(input_ids, next_prefill, idx_mapping, qsl,
                                   prefill_token_ids, prefill_lens, num_computed)
            ii = input_ids.cpu()
            np_ = next_prefill.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: ({num_reqs},)  (num_reqs,), BLOCK_SIZE: 1024")
            log(f, "\nOutput:")
            log(f, f"  input_ids: {ii.tolist()}")
            log(f, f"  next_prefill_tokens: {np_.tolist()}")

            # reference
            ref_ii = torch.full((num_tokens,), -1, dtype=torch.int32)
            ref_np = torch.full((MAX_REQS,), -1, dtype=torch.int32)
            for b in range(num_reqs):
                req = int(idx_r[b]); pl = int(pl_r[req]); nc = int(nc_r[req])
                if nc >= pl:
                    continue
                qs = int(qsl_r[b]); qe = int(qsl_r[b + 1]); ql = qe - qs
                for j in range(ql):
                    ref_ii[qs + j] = int(pti_r[req, nc + j])
                if nc + ql < pl:
                    ref_np[req] = int(pti_r[req, nc + ql])
            diff_ii = (ii - ref_ii).abs().max().item()
            diff_np = (np_ - ref_np).abs().max().item()
            log(f, "\nValidation:")
            log(f, f"  input_ids max abs diff: {diff_ii}")
            log(f, f"  next_prefill max abs diff: {diff_np}")
            torch.testing.assert_close(ii, ref_ii, atol=0, rtol=0)
            torch.testing.assert_close(np_, ref_np, atol=0, rtol=0)
            log(f, "  assert_close: PASSED")
            log(f, "\nSummary:")
            log(f, f"  Kernel execution: SUCCESS (grid ({num_reqs},))")
            log(f, "  Validation vs PyTorch reference: PASSED")
            log(f, f"  max abs diff: input_ids={diff_ii}, next_prefill={diff_np}")
        except Exception:
            log(f, f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log(f, "\nSummary:\n  Validation vs PyTorch reference: FAILED")
            sys.exit(1)
        finally:
            log(f, "\n" + "-" * 36)
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
