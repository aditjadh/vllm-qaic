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

from vllm.model_executor.layers.fused_moe.deep_gemm_utils import round_up_128
from vllm.triton_utils import tl, triton

# ── Configuration ────────────────────────────────────────────────────────────
# round_up_128 is a @triton.jit device function: rounds x up to a multiple of
# 128 via ((x + 127) // 128) * 128. We wrap it to round a block of ints.
NUM_VALS   = 16
BLOCK_SIZE = 16
DEVICE     = "qaic"

KERNEL_NAME = "round_up_128 (via wrapper kernel)"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_round_up_128_{timestamp}.log")


@triton.jit
def round_up_128_wrapper(
    in_ptr,
    out_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(in_ptr + offs, mask=mask, other=0)
    y = round_up_128(x)
    tl.store(out_ptr + offs, y, mask=mask)


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    # Mix of exact multiples, just-over, and just-under 128 boundaries.
    vals = torch.tensor(
        [0, 1, 127, 128, 129, 255, 256, 257, 5, 10, 130, 64, 511, 512, 513, 1000],
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
            log(f, f"  num_vals={NUM_VALS}, BLOCK_SIZE={BLOCK_SIZE}")

            round_up_128_wrapper[grid](vals, out, NUM_VALS, BLOCK_SIZE=BLOCK_SIZE)

            out_host = out.cpu()   # force sync

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}")
            log(f, f"  BLOCK_SIZE: {BLOCK_SIZE}")

            log(f, "\nOutput:")
            log(f, f"  rounded : {out_host.tolist()}")

            # ── Reference validation ────────────────────────────────────────
            ref = ((vals_ref.cpu() + 127) // 128) * 128
            diff = (out_host - ref).abs().max().item()
            log(f, "\nValidation (vs PyTorch ((x+127)//128)*128):")
            log(f, f"  reference : {ref.tolist()}")
            log(f, f"  max abs diff: {diff}")

            torch.testing.assert_close(out_host, ref, atol=0, rtol=0)
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
