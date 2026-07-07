# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
import os
import sys
import traceback
from datetime import datetime

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vllm"))
from _ktest import run_test, logger

from vllm.triton_utils import tl, triton
from vllm.model_executor.layers.mamba.ops.mamba_ssm import softplus

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "softplus"


@triton.jit
def _softplus_wrapper_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = softplus(x)
    tl.store(out_ptr + offs, y, mask=mask)


def main(log_path):
    torch.manual_seed(42)
    n = 32
    x = torch.randn(n, dtype=torch.float32, device=DEVICE) * 5.0
    out = torch.empty(n, dtype=torch.float32, device=DEVICE)
    x_cpu = x.clone().cpu()

    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  x: shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}")
            log(f"  n={n}, range=[{x_cpu.min().item():.2f},{x_cpu.max().item():.2f}]")

            _softplus_wrapper_kernel[(1,)](x, out, n, BLOCK=n)

            o = out.cpu()
            ref = F.softplus(x_cpu)
            diff = (o - ref).abs().max().item()

            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: (1,), BLOCK={n}")
            log("\nOutput:")
            log(f"  out mean={o.mean().item():.4f}, ref mean={ref.mean().item():.4f}")
            log(f"\nValidation:")
            log(f"  max abs diff vs F.softplus: {diff:.6f}")
            assert diff < 1e-4, f"diff {diff} too large"
            log("  check: PASSED")
            log("\nSummary:")
            log("  Kernel execution: SUCCESS (grid (1,))")
            log("  Validation vs PyTorch F.softplus: PASSED")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Kernel execution: FAILED")
            fp.write("\n" + "-" * 36 + "\n")
            sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "softplus")
