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

from vllm.v1.worker.gpu.input_batch import prepare_pos_seq_lens

# ── Configuration ────────────────────────────────────────────────────────────
MAX_REQS = 6                      # seq_lens buffer length (>= num_reqs for padding)
DEVICE   = "qaic"

IDX_MAPPING  = [0, 1, 2]
QUERY_LENS   = [4, 3, 5]
NUM_COMPUTED = [0, 2, 7, 0]       # by req_state_idx

KERNEL_NAME = "_prepare_pos_seq_lens_kernel"
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = os.path.join(LOG_DIR, f"test_prepare_pos_seq_lens_{timestamp}.log")


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    num_reqs = len(IDX_MAPPING)
    num_tokens = sum(QUERY_LENS)

    qsl = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    qsl[1:] = torch.tensor(QUERY_LENS, dtype=torch.int32).cumsum(0)
    idx_mapping = torch.tensor(IDX_MAPPING, dtype=torch.int32, device=DEVICE)
    num_computed = torch.tensor(NUM_COMPUTED, dtype=torch.int32, device=DEVICE)

    pos = torch.full((num_tokens,), -1, dtype=torch.int32, device=DEVICE)
    seq_lens = torch.full((MAX_REQS,), -1, dtype=torch.int32, device=DEVICE)

    qsl_r, idx_r, nc_r = qsl.clone(), idx_mapping.clone(), num_computed.clone()
    with open(log_path, "w") as f:
        try:
            log(f, timestamp); log(f, f"Kernel: {KERNEL_NAME}")
            log(f, "\nInputs:")
            log(f, f"  idx_mapping={IDX_MAPPING}, query_lens={QUERY_LENS}, num_computed={NUM_COMPUTED}")
            log(f, f"  num_tokens={num_tokens}, max_num_reqs(seq_lens len)={MAX_REQS}")

            prepare_pos_seq_lens(idx_mapping, qsl, num_computed, pos, seq_lens)
            pos_h = pos.cpu(); sl_h = seq_lens.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: ({num_reqs + 1},)  (num_reqs+1; last pads seq_lens), BLOCK_SIZE: 1024")
            log(f, "\nOutput:")
            log(f, f"  pos: {pos_h.tolist()}")
            log(f, f"  seq_lens: {sl_h.tolist()}")

            ref_pos = torch.full((num_tokens,), -1, dtype=torch.int32)
            ref_sl = torch.zeros(MAX_REQS, dtype=torch.int32)
            for b in range(num_reqs):
                req = int(idx_r[b]); nc = int(nc_r[req])
                qs = int(qsl_r[b]); qe = int(qsl_r[b + 1]); ql = qe - qs
                ref_sl[b] = nc + ql
                for j in range(ql):
                    ref_pos[qs + j] = nc + j
            diff_pos = (pos_h - ref_pos).abs().max().item()
            diff_sl = (sl_h - ref_sl).abs().max().item()
            log(f, "\nValidation:")
            log(f, f"  reference seq_lens: {ref_sl.tolist()}")
            log(f, f"  pos max abs diff: {diff_pos}")
            log(f, f"  seq_lens max abs diff: {diff_sl}")
            torch.testing.assert_close(pos_h, ref_pos, atol=0, rtol=0)
            torch.testing.assert_close(sl_h, ref_sl, atol=0, rtol=0)
            log(f, "  assert_close: PASSED")
            log(f, "\nSummary:")
            log(f, f"  Kernel execution: SUCCESS (grid ({num_reqs + 1},))")
            log(f, "  Validation vs PyTorch reference: PASSED")
            log(f, f"  max abs diff: pos={diff_pos}, seq_lens={diff_sl}")
        except Exception:
            log(f, f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log(f, "\nSummary:\n  Validation vs PyTorch reference: FAILED")
            sys.exit(1)
        finally:
            log(f, "\n" + "-" * 36)
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
