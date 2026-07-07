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

from vllm.v1.attention.ops.triton_unified_attention import find_seq_idx
from vllm.triton_utils import tl, triton

# ── Configuration ────────────────────────────────────────────────────────────
# find_seq_idx is a @triton.jit device function: binary-search the segment that
# a target token index falls into, given cumulative query_start_loc.
# With use_q_block_mode=False it returns the seq index s.t.
#   query_start_loc[s] <= target < query_start_loc[s+1].
DEVICE = "qaic"
QUERY_LENS = [4, 3, 5, 2]         # cumsum -> boundaries
BLOCK_Q    = 1                     # unused when use_q_block_mode=False
NUM_TARGETS = 14                   # = sum(QUERY_LENS)

KERNEL_NAME = "find_seq_idx (via wrapper kernel)"
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = os.path.join(LOG_DIR, f"test_find_seq_idx_{timestamp}.log")


@triton.jit
def find_seq_idx_wrapper(
    qsl_ptr, out_ptr, num_seqs, n,
    BLOCK_Q: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < n:
        idx = find_seq_idx(qsl_ptr, pid, num_seqs, BLOCK_Q, False)
        tl.store(out_ptr + pid, idx)


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    num_seqs = len(QUERY_LENS)
    qsl = torch.zeros(num_seqs + 1, dtype=torch.int32, device=DEVICE)
    qsl[1:] = torch.tensor(QUERY_LENS, dtype=torch.int32).cumsum(0)
    out = torch.full((NUM_TARGETS,), -1, dtype=torch.int32, device=DEVICE)
    qsl_r = qsl.clone()

    grid = (NUM_TARGETS,)
    with open(log_path, "w") as f:
        try:
            log(f, timestamp); log(f, f"Kernel: {KERNEL_NAME}")
            log(f, "\nInputs:")
            log(f, f"  query_lens={QUERY_LENS}, query_start_loc={qsl.cpu().tolist()}")
            log(f, f"  num_seqs={num_seqs}, num_targets={NUM_TARGETS}, use_q_block_mode=False")

            find_seq_idx_wrapper[grid](qsl, out, num_seqs, NUM_TARGETS,
                                       BLOCK_Q=BLOCK_Q, BLOCK_SIZE=1)
            out_h = out.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}  (one program per target index)")
            log(f, "\nOutput:")
            log(f, f"  seq_idx per target: {out_h.tolist()}")

            # reference: bucketize
            ref = torch.empty(NUM_TARGETS, dtype=torch.int32)
            bnd = qsl_r.cpu()
            for t in range(NUM_TARGETS):
                s = 0
                for k in range(num_seqs):
                    if bnd[k] <= t:
                        s = k
                ref[t] = s
            diff = (out_h - ref).abs().max().item()
            log(f, "\nValidation (vs PyTorch segment bucketize):")
            log(f, f"  reference: {ref.tolist()}")
            log(f, f"  max abs diff: {diff}")
            torch.testing.assert_close(out_h, ref, atol=0, rtol=0)
            log(f, "  assert_close: PASSED")
            log(f, "\nSummary:")
            log(f, f"  Kernel execution: SUCCESS (grid {grid})")
            log(f, "  Validation vs PyTorch reference: PASSED")
            log(f, f"  max abs diff: {diff}")
        except Exception:
            log(f, f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log(f, "\nSummary:\n  Validation vs PyTorch reference: FAILED")
            sys.exit(1)
        finally:
            log(f, "\n" + "-" * 36)
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
