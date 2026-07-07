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

from vllm.v1.worker.gpu.input_batch import expand_idx_mapping

# ── Configuration ────────────────────────────────────────────────────────────
# expand_idx_mapping repeats each request's state index across its logit slots,
# using cu_num_logits as the segment boundaries.
DEVICE = "qaic"
IDX_MAPPING   = [5, 3, 7, 2]       # req -> req_state_idx
NUM_LOGITS    = [2, 1, 4, 3]       # logits per request

KERNEL_NAME = "_expand_idx_mapping_kernel"
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = os.path.join(LOG_DIR, f"test_expand_idx_mapping_{timestamp}.log")


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    num_reqs = len(IDX_MAPPING)
    idx_mapping = torch.tensor(IDX_MAPPING, dtype=torch.int32, device=DEVICE)
    cu = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    cu[1:] = torch.tensor(NUM_LOGITS, dtype=torch.int32).cumsum(0)
    total = sum(NUM_LOGITS)
    max_expand = max(NUM_LOGITS)

    idx_r = idx_mapping.clone()
    cu_r = cu.clone()
    with open(log_path, "w") as f:
        try:
            log(f, timestamp); log(f, f"Kernel: {KERNEL_NAME}")
            log(f, "\nInputs:")
            log(f, f"  idx_mapping={IDX_MAPPING}, num_logits_per_req={NUM_LOGITS}")
            log(f, f"  cu_num_logits={cu.cpu().tolist()}, total={total}, max_expand={max_expand}")

            out = expand_idx_mapping(idx_mapping, total, cu, max_expand)
            out_h = out.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: ({num_reqs},), BLOCK_SIZE: next_pow2({max_expand})")
            log(f, "\nOutput:")
            log(f, f"  expanded_idx_mapping: {out_h.tolist()}")

            ref = torch.empty(total, dtype=out_h.dtype)
            for r in range(num_reqs):
                s = int(cu_r[r]); e = int(cu_r[r + 1])
                ref[s:e] = int(idx_r[r])
            diff = (out_h - ref).abs().max().item()
            log(f, "\nValidation:")
            log(f, f"  reference: {ref.tolist()}")
            log(f, f"  max abs diff: {diff}")
            torch.testing.assert_close(out_h, ref, atol=0, rtol=0)
            log(f, "  assert_close: PASSED")
            log(f, "\nSummary:")
            log(f, f"  Kernel execution: SUCCESS (grid ({num_reqs},))")
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
