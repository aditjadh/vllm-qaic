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

from vllm.v1.worker.gpu.buffer_utils import _apply_write_kernel

# ── Configuration ────────────────────────────────────────────────────────────
# _apply_write_kernel writes N staged variable-length content blocks into rows of
# an output buffer: row write_indices[pid], starting at write_starts[pid], with
# content write_contents[cu_lens[pid-1] : cu_lens[pid]].
NUM_ROWS = 8
ROW_LEN  = 32
DEVICE   = "qaic"

KERNEL_NAME = "_apply_write_kernel"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_apply_write_kernel_{timestamp}.log")


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    torch.manual_seed(42)

    output = torch.zeros(NUM_ROWS, ROW_LEN, dtype=torch.int32, device=DEVICE)

    # 3 staged writes:
    #   write 0 -> row 2, start 0,  content [10,11,12,13]
    #   write 1 -> row 5, start 4,  content [20,21,22]
    #   write 2 -> row 0, start 10, content [30,31,32,33,34]
    writes = [
        (2, 0, [10, 11, 12, 13]),
        (5, 4, [20, 21, 22]),
        (0, 10, [30, 31, 32, 33, 34]),
    ]
    n = len(writes)

    indices = torch.tensor([w[0] for w in writes], dtype=torch.int32, device=DEVICE)
    starts = torch.tensor([w[1] for w in writes], dtype=torch.int32, device=DEVICE)
    contents_list = []
    cu_lens = []
    acc = 0
    for w in writes:
        contents_list.extend(w[2])
        acc += len(w[2])
        cu_lens.append(acc)
    contents = torch.tensor(contents_list, dtype=torch.int32, device=DEVICE)
    cu_lens_t = torch.tensor(cu_lens, dtype=torch.int32, device=DEVICE)

    grid = (n,)
    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")
            log(f, "\nInputs:")
            log(f, f"  output : shape={tuple(output.shape)}, dtype={output.dtype}, device={output.device}")
            log(f, f"  writes : {writes}")
            log(f, f"  cu_lens={cu_lens}, n={n}")

            _apply_write_kernel[grid](
                output, output.stride(0),
                indices, starts, contents, cu_lens_t,
                BLOCK_SIZE=1024,
            )
            out_host = output.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}  (num staged writes,)")
            log(f, "  BLOCK_SIZE: 1024")

            # ── Reference ────────────────────────────────────────────────────
            ref = torch.zeros(NUM_ROWS, ROW_LEN, dtype=torch.int32)
            for (row, start, content) in writes:
                ref[row, start : start + len(content)] = torch.tensor(content, dtype=torch.int32)

            diff = (out_host - ref).abs().max().item()
            log(f, "\nOutput:")
            log(f, f"  row 2: {out_host[2, :6].tolist()}")
            log(f, f"  row 5: {out_host[5, :8].tolist()}")
            log(f, f"  row 0: {out_host[0, 8:16].tolist()}")
            log(f, "\nValidation (vs PyTorch staged scatter-write):")
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
