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

from vllm.model_executor.layers.fla.ops.op import gather
from vllm.model_executor.layers.fla.ops.utils import is_gather_supported
from vllm.triton_utils import tl, triton

# ── Configuration ────────────────────────────────────────────────────────────
# gather wraps tl.gather (or a None fallback if unsupported). We gather along
# axis 1 of a [ROWS, COLS] tile using a per-element index tile.
ROWS   = 8
COLS   = 16
DEVICE = "qaic"

KERNEL_NAME = "gather (tl.gather wrapper)"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_gather_{timestamp}.log")


@triton.jit
def gather_wrapper(
    src_ptr, idx_ptr, out_ptr,
    ROWS: tl.constexpr, COLS: tl.constexpr,
):
    r = tl.arange(0, ROWS)
    c = tl.arange(0, COLS)
    offs = r[:, None] * COLS + c[None, :]
    src = tl.load(src_ptr + offs)
    idx = tl.load(idx_ptr + offs)
    out = gather(src, idx, axis=1)
    tl.store(out_ptr + offs, out)


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    torch.manual_seed(42)

    src = torch.randn(ROWS, COLS, dtype=torch.float32, device=DEVICE)
    idx = torch.randint(0, COLS, (ROWS, COLS), dtype=torch.int32, device=DEVICE)
    out = torch.empty(ROWS, COLS, dtype=torch.float32, device=DEVICE)

    src_ref = src.clone()
    idx_ref = idx.clone()

    grid = (1,)
    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")
            log(f, "\nInputs:")
            log(f, f"  src : shape={tuple(src.shape)}, dtype={src.dtype}, device={src.device}")
            log(f, f"  idx : shape={tuple(idx.shape)}, dtype={idx.dtype}")
            log(f, f"  is_gather_supported={is_gather_supported}")
            log(f, f"  ROWS={ROWS}, COLS={COLS}, axis=1")

            if not is_gather_supported:
                log(f, "\nStatus: SKIPPED")
                log(f, "  tl.gather is NOT supported on this backend; the vLLM")
                log(f, "  fallback 'gather' returns None (a compile-time stub).")
                log(f, "  There is no functional kernel to execute.")
                log(f, "\nSummary:")
                log(f, "  Kernel execution: SKIPPED (gather unsupported, fallback is a no-op stub)")
                log(f, "\n" + "-" * 36)
                print(f"\nLog saved to: {log_path}")
                return

            gather_wrapper[grid](src, idx, out, ROWS=ROWS, COLS=COLS)
            out_host = out.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}")
            log(f, "\nOutput:")
            log(f, f"  out : shape={tuple(out.shape)}, mean={out_host.mean().item():.4f}")

            ref = torch.gather(src_ref.cpu(), 1, idx_ref.cpu().to(torch.int64))
            diff = (out_host - ref).abs().max().item()
            log(f, "\nValidation (vs torch.gather along dim 1):")
            log(f, f"  max abs diff: {diff:.6f}")

            torch.testing.assert_close(out_host, ref, atol=1e-5, rtol=1e-5)
            log(f, "  assert_close: PASSED")

            log(f, "\nSummary:")
            log(f, f"  Kernel execution: SUCCESS (grid {grid})")
            log(f, "  Validation vs PyTorch reference: PASSED")
            log(f, f"  max abs diff: {diff:.6f}")

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
