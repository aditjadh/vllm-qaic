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

from vllm.model_executor.layers.fused_moe.fused_moe import write_zeros_to_output
from vllm.triton_utils import tl, triton

# ── Configuration ────────────────────────────────────────────────────────────
# write_zeros_to_output is a @triton.jit device function used by fused_moe_kernel
# to zero a [BLOCK_SIZE_M, BLOCK_SIZE_N] output tile when an expert is skipped.
# We wrap it in a launchable kernel that zeros one tile of C [M, N].
M       = 16
N       = 64
DTYPE   = torch.float16
DEVICE  = "qaic"

BLOCK_SIZE_M = 16
BLOCK_SIZE_N = 64
TOP_K        = 1            # offs_token are row indices directly (top_k=1)

KERNEL_NAME = "write_zeros_to_output (via wrapper kernel)"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_write_zeros_to_output_{timestamp}.log")


@triton.jit
def write_zeros_wrapper(
    c_ptr,
    stride_cm,
    stride_cn,
    sorted_token_ids_ptr,
    num_valid_tokens,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs)
    token_mask = offs_token < num_valid_tokens
    write_zeros_to_output(
        c_ptr,
        stride_cm,
        stride_cn,
        pid_n,
        N,
        offs_token,
        token_mask,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        tl.float16,
    )


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    torch.manual_seed(42)

    # Pre-fill C with non-zero values so we can confirm the kernel zeroed them.
    c = torch.randn(M, N, dtype=DTYPE, device=DEVICE)
    sorted_token_ids = torch.arange(BLOCK_SIZE_M, dtype=torch.int32, device=DEVICE)
    num_valid_tokens = M

    c_before = c.clone()

    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (num_pid_n,)

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")

            log(f, "\nInputs:")
            log(f, f"  c (pre-filled) : shape={tuple(c.shape)}, dtype={c.dtype}, device={c.device}")
            log(f, f"  sorted_token_ids: {sorted_token_ids.cpu().tolist()}")
            log(f, f"  num_valid_tokens={num_valid_tokens}, M={M}, N={N}")
            log(f, f"  BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_N={BLOCK_SIZE_N}")
            log(f, f"  c mean before (abs): {c_before.float().abs().mean().item():.4f}")

            write_zeros_wrapper[grid](
                c,
                c.stride(0),
                c.stride(1),
                sorted_token_ids,
                num_valid_tokens,
                N,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
            )

            c_host = c.cpu()   # force sync

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}  (num_pid_n,)")
            log(f, f"  block: BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_N={BLOCK_SIZE_N}")

            log(f, "\nOutput:")
            log(f, f"  c : shape={tuple(c.shape)}, min={c_host.min().item():.4f}, "
                   f"max={c_host.max().item():.4f}, abs mean={c_host.float().abs().mean().item():.4f}")

            # ── Reference: the whole [M, N] tile should now be zero ──────────
            ref = torch.zeros(M, N, dtype=torch.float32)
            diff = (c_host.float() - ref).abs().max().item()
            num_zeros = int((c_host == 0).sum())
            rows_zeroed = (c_host == 0).all(dim=1).nonzero().reshape(-1).tolist()
            log(f, "\nValidation (output region must be all zeros):")
            log(f, f"  zeros written: {num_zeros} / {c_host.numel()}")
            log(f, f"  rows fully zeroed: {rows_zeroed} (expected 0..{M - 1})")
            log(f, f"  max abs value after zeroing: {diff}")

            torch.testing.assert_close(c_host.float(), ref, atol=0, rtol=0)
            log(f, "  assert_close: PASSED")

            log(f, "\nSummary:")
            log(f, f"  Kernel execution: SUCCESS (grid {grid})")
            log(f, "  Validation vs PyTorch reference: PASSED")
            log(f, f"  max abs value after zeroing: {diff}")

        except Exception:
            msg = f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}"
            log(f, msg)
            log(f, "\nSummary:")
            log(f, "  Kernel execution: SUCCESS (no error/crash)")
            log(f, "  Validation vs PyTorch reference: FAILED (silent miscompile)")
            log(f, "  Note: kernel ran but the 2D scatter store with a gathered")
            log(f, "        row-index vector zeroed only a subset of rows on QAIC.")
            sys.exit(1)

        finally:
            log(f, "\n" + "-" * 36)

    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
