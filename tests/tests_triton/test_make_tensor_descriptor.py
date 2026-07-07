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

import vllm.model_executor.layers.fla.ops.op as fla_op
from vllm.triton_utils import tl, triton

# ── Configuration ────────────────────────────────────────────────────────────
# make_tensor_descriptor builds a TMA (Tensor Memory Accelerator) descriptor.
# It resolves to triton.language.(_experimental_)make_tensor_descriptor when TMA
# is available, else a None-returning stub. We probe whether a real TMA op is
# present and, if so, build a descriptor over a 2D tensor and load a tile.
M, N    = 64, 64
BLOCK_M = 32
BLOCK_N = 32
DEVICE  = "qaic"

KERNEL_NAME = "make_tensor_descriptor"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_make_tensor_descriptor_{timestamp}.log")


@triton.jit
def tma_load_wrapper(
    a_ptr, out_ptr, M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    desc = fla_op.make_tensor_descriptor(
        a_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N]
    )
    tile = desc.load([0, 0])
    r = tl.arange(0, BLOCK_M)
    c = tl.arange(0, BLOCK_N)
    tl.store(out_ptr + r[:, None] * BLOCK_N + c[None, :], tile)


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def _tma_available():
    tl_mod = triton.language
    return hasattr(tl_mod, "_experimental_make_tensor_descriptor") or hasattr(
        tl_mod, "make_tensor_descriptor"
    )


def main():
    torch.manual_seed(42)

    a = torch.randn(M, N, dtype=torch.float32, device=DEVICE)
    out = torch.empty(BLOCK_M, BLOCK_N, dtype=torch.float32, device=DEVICE)
    a_ref = a.clone()

    grid = (1,)
    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")
            log(f, "\nInputs:")
            log(f, f"  a : shape={tuple(a.shape)}, dtype={a.dtype}, device={a.device}")
            log(f, f"  M={M}, N={N}, BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}")
            log(f, f"  TMA available (triton.language has make_tensor_descriptor): {_tma_available()}")

            if not _tma_available():
                log(f, "\nStatus: SKIPPED")
                log(f, "  TMA is not available; vLLM's make_tensor_descriptor is a")
                log(f, "  None-returning compile stub. No functional kernel to run.")
                log(f, "\nSummary:")
                log(f, "  Kernel execution: SKIPPED (TMA unsupported, fallback is a no-op stub)")
                log(f, "\n" + "-" * 36)
                print(f"\nLog saved to: {log_path}")
                return

            tma_load_wrapper[grid](a, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
            out_host = out.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}, block: ({BLOCK_M}, {BLOCK_N})")
            log(f, "\nOutput:")
            log(f, f"  tile : shape={tuple(out.shape)}, mean={out_host.mean().item():.4f}")

            ref = a_ref[:BLOCK_M, :BLOCK_N].cpu()
            diff = (out_host - ref).abs().max().item()
            log(f, "\nValidation (vs a[:BLOCK_M, :BLOCK_N]):")
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
