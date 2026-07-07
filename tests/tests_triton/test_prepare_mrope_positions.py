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

from vllm.v1.worker.gpu.mm.mrope_utils import _prepare_mrope_positions_kernel

# ── Configuration ────────────────────────────────────────────────────────────
MAX_NUM_REQS   = 4
MAX_MODEL_LEN  = 32
MAX_NUM_TOKENS = 64
BLOCK_SIZE     = 1024
DEVICE         = "qaic"

# Per-batch request layout (batch_idx -> req_state_idx)
IDX_MAPPING        = [0, 1, 2]
QUERY_LENS         = [4, 2, 3]          # query_start_loc = cumsum
PREFILL_LENS       = [10, 5, 8, 0]      # indexed by req_state_idx
NUM_COMPUTED       = [0, 5, 2, 0]       # req0: prefill, req1: decode, req2: prefill
PREFILL_MROPE_DELTA = [100, 200, 300, 0]

KERNEL_NAME = "_prepare_mrope_positions_kernel"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "verified_kernels")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_prepare_mrope_positions_{timestamp}.log")


def reference_prepare(
    prefill_mrope_positions, prefill_mrope_delta,
    idx_mapping, query_start_loc, prefill_lens, num_computed_tokens,
):
    """Pure-PyTorch reference mirroring the Triton kernel."""
    out = torch.zeros(3, MAX_NUM_TOKENS + 1, dtype=torch.int64)
    num_reqs = idx_mapping.shape[0]
    for batch_idx in range(num_reqs):
        req_state_idx = int(idx_mapping[batch_idx])
        prefill_len   = int(prefill_lens[req_state_idx])
        num_computed  = int(num_computed_tokens[req_state_idx])
        is_prefill    = num_computed < prefill_len

        query_start = int(query_start_loc[batch_idx])
        query_end   = int(query_start_loc[batch_idx + 1])
        query_len   = query_end - query_start
        mrope_delta = int(prefill_mrope_delta[req_state_idx])

        for blk in range(query_len):
            orig_pos = num_computed + blk
            for j in range(3):
                if is_prefill:
                    pos = int(prefill_mrope_positions[3 * req_state_idx + j, orig_pos])
                else:
                    pos = orig_pos + mrope_delta
                out[j, query_start + blk] = pos
    return out


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    torch.manual_seed(42)

    # prefill_mrope_positions laid out as (MAX_NUM_REQS * 3, MAX_MODEL_LEN),
    # addressed as [3*req + j, pos]. Fill with deterministic known values.
    prefill_mrope_positions = (
        torch.arange(MAX_NUM_REQS * 3 * MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE)
        .reshape(MAX_NUM_REQS * 3, MAX_MODEL_LEN)
    )

    prefill_mrope_delta = torch.tensor(PREFILL_MROPE_DELTA, dtype=torch.int32, device=DEVICE)
    idx_mapping         = torch.tensor(IDX_MAPPING, dtype=torch.int32, device=DEVICE)

    query_start_loc = torch.zeros(len(QUERY_LENS) + 1, dtype=torch.int32, device=DEVICE)
    query_start_loc[1:] = torch.tensor(QUERY_LENS, dtype=torch.int32).cumsum(0)

    prefill_lens        = torch.tensor(PREFILL_LENS, dtype=torch.int32, device=DEVICE)
    num_computed_tokens = torch.tensor(NUM_COMPUTED, dtype=torch.int32, device=DEVICE)

    mrope_positions = torch.zeros(3, MAX_NUM_TOKENS + 1, dtype=torch.int64, device=DEVICE)

    num_reqs = idx_mapping.shape[0]
    grid = (num_reqs,)

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")

            log(f, "\nInputs:")
            log(f, f"  mrope_positions       : shape={tuple(mrope_positions.shape)}, dtype={mrope_positions.dtype}, device={mrope_positions.device}")
            log(f, f"  prefill_mrope_positions: shape={tuple(prefill_mrope_positions.shape)}, dtype={prefill_mrope_positions.dtype}")
            log(f, f"  prefill_mrope_delta    : {PREFILL_MROPE_DELTA}")
            log(f, f"  idx_mapping            : {IDX_MAPPING}")
            log(f, f"  query_lens             : {QUERY_LENS} -> query_start_loc={query_start_loc.tolist()}")
            log(f, f"  prefill_lens           : {PREFILL_LENS}")
            log(f, f"  num_computed_tokens    : {NUM_COMPUTED}")
            log(f, f"  BLOCK_SIZE             : {BLOCK_SIZE}")

            _prepare_mrope_positions_kernel[grid](
                mrope_positions,
                mrope_positions.stride(0),
                prefill_mrope_positions,
                3 * MAX_MODEL_LEN,
                MAX_MODEL_LEN,
                prefill_mrope_delta,
                idx_mapping,
                query_start_loc,
                prefill_lens,
                num_computed_tokens,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}")
            log(f, f"  BLOCK_SIZE: {BLOCK_SIZE}")

            total_tokens = int(query_start_loc[-1])
            log(f, "\nOutput (first {} token positions):".format(total_tokens))
            for j in range(3):
                log(f, f"  dim {j}: {mrope_positions[j, :total_tokens].tolist()}")

            # ── Reference validation ────────────────────────────────────────
            ref = reference_prepare(
                prefill_mrope_positions.cpu(), prefill_mrope_delta.cpu(),
                idx_mapping.cpu(), query_start_loc.cpu(),
                prefill_lens.cpu(), num_computed_tokens.cpu(),
            )
            diff = (mrope_positions.cpu() - ref).abs().max().item()
            log(f, "\nValidation (vs PyTorch reference):")
            log(f, f"  max abs diff: {diff}")

            torch.testing.assert_close(mrope_positions.cpu(), ref, atol=0, rtol=0)
            log(f, "  assert_close: PASSED")

            log(f, "\nSummary:")
            log(f, f"  Kernel execution: SUCCESS (grid {grid})")
            log(f, "  Validation vs PyTorch reference: PASSED")
            log(f, f"  max abs diff: {diff}")

        except Exception:
            msg = f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}"
            log(f, msg)
            log(f, "\nSummary:")
            log(f, "  Validation vs PyTorch reference: FAILED")
            sys.exit(1)

        finally:
            log(f, "\n" + "-" * 36)

    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
