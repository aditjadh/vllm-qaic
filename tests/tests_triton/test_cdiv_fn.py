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

from vllm.v1.attention.ops.triton_unified_attention import cdiv_fn
from vllm.triton_utils import tl, triton

# ── Configuration ────────────────────────────────────────────────────────────
# cdiv_fn is a @triton.jit device function: ceil-div (x + y - 1) // y.
NUM_VALS   = 16
BLOCK_SIZE = 16
DIVISOR    = 7
DEVICE     = "qaic"

KERNEL_NAME = "cdiv_fn (via wrapper kernel)"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_cdiv_fn_{timestamp}.log")


@triton.jit
def cdiv_wrapper(in_ptr, out_ptr, y, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(in_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + offs, cdiv_fn(x, y), mask=mask)


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    vals = torch.tensor(
        [0, 1, 6, 7, 8, 13, 14, 15, 21, 100, 49, 50, 48, 2, 70, 99],
        dtype=torch.int32, device=DEVICE,
    )
    out = torch.empty(NUM_VALS, dtype=torch.int32, device=DEVICE)
    vals_ref = vals.clone()

    grid = (1,)
    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")
            log(f, "\nInputs:")
            log(f, f"  values : {vals.cpu().tolist()}")
            log(f, f"  divisor={DIVISOR}, num_vals={NUM_VALS}, BLOCK_SIZE={BLOCK_SIZE}")

            cdiv_wrapper[grid](vals, out, DIVISOR, NUM_VALS, BLOCK_SIZE=BLOCK_SIZE)
            out_host = out.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}, BLOCK_SIZE: {BLOCK_SIZE}")
            log(f, "\nOutput:")
            log(f, f"  cdiv : {out_host.tolist()}")

            ref = ((vals_ref.cpu() + DIVISOR - 1) // DIVISOR)
            diff = (out_host - ref).abs().max().item()
            log(f, "\nValidation (vs PyTorch (x+y-1)//y):")
            log(f, f"  reference : {ref.tolist()}")
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
