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

from vllm.v1.worker.gpu.sample.prompt_logprob import get_prompt_logprobs_token_ids

# ── Configuration ────────────────────────────────────────────────────────────
# _prompt_logprobs_token_ids_kernel gathers, for each query position, the NEXT
# prefill token id (pos shifted by +1) into a flat per-token output buffer.
MAX_MODEL_LEN = 64
DEVICE        = "qaic"

# Per-batch request layout
IDX_MAPPING  = [0, 1, 2]
QUERY_LENS   = [4, 3, 5]            # query_start_loc = cumsum
NUM_COMPUTED = [0, 2, 1, 0]         # indexed by req_state_idx

KERNEL_NAME = "_prompt_logprobs_token_ids_kernel"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_prompt_logprobs_token_ids_{timestamp}.log")


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    torch.manual_seed(42)

    num_reqs = len(IDX_MAPPING)
    num_tokens = sum(QUERY_LENS)

    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    query_start_loc[1:] = torch.tensor(QUERY_LENS, dtype=torch.int32).cumsum(0)

    idx_mapping = torch.tensor(IDX_MAPPING, dtype=torch.int32, device=DEVICE)
    num_computed_tokens = torch.tensor(NUM_COMPUTED, dtype=torch.int32, device=DEVICE)

    # prefill_token_ids: [max_num_reqs, max_model_len]; fill with distinct values.
    max_reqs = len(NUM_COMPUTED)
    prefill_token_ids = (
        torch.arange(max_reqs * MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE)
        .reshape(max_reqs, MAX_MODEL_LEN)
    )

    qsl_ref = query_start_loc.clone()
    idx_ref = idx_mapping.clone()
    nct_ref = num_computed_tokens.clone()
    pti_ref = prefill_token_ids.clone()

    grid = (num_reqs,)

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")
            log(f, "\nInputs:")
            log(f, f"  prefill_token_ids : shape={tuple(prefill_token_ids.shape)}, dtype={prefill_token_ids.dtype}, device={prefill_token_ids.device}")
            log(f, f"  idx_mapping       : {IDX_MAPPING}")
            log(f, f"  query_lens        : {QUERY_LENS} -> query_start_loc={query_start_loc.tolist()}")
            log(f, f"  num_computed      : {NUM_COMPUTED}")
            log(f, f"  num_tokens={num_tokens}")

            out = get_prompt_logprobs_token_ids(
                num_tokens, query_start_loc, idx_mapping,
                num_computed_tokens, prefill_token_ids,
            )
            out_host = out.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}  (num_reqs,)")
            log(f, "  BLOCK_SIZE: 1024")

            log(f, "\nOutput:")
            log(f, f"  token_ids: {out_host.tolist()}")

            # ── Reference: out[query_start+b] = prefill[req, num_computed+1+b] ──
            ref = torch.empty(num_tokens, dtype=torch.int64)
            for batch_idx in range(num_reqs):
                req = int(idx_ref[batch_idx])
                qs = int(qsl_ref[batch_idx])
                qe = int(qsl_ref[batch_idx + 1])
                nc = int(nct_ref[req])
                for b in range(qe - qs):
                    ref[qs + b] = int(pti_ref[req, nc + 1 + b])
            diff = (out_host - ref).abs().max().item()
            log(f, "\nValidation (vs PyTorch shifted prefill gather):")
            log(f, f"  reference: {ref.tolist()}")
            log(f, f"  max abs diff: {diff}")

            torch.testing.assert_close(out_host, ref, atol=0, rtol=0)
            log(f, "  assert_close: PASSED")

            log(f, "\nSummary:")
            log(f, f"  Kernel execution: SUCCESS (grid {grid})")
            log(f, "  Validation vs PyTorch reference: PASSED")
            log(f, f"  max abs diff: {diff}")

        except Exception:
            log(f, f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log(f, "\nSummary:")
            log(f, "  Validation vs PyTorch reference: FAILED")
            sys.exit(1)
        finally:
            log(f, "\n" + "-" * 36)

    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
