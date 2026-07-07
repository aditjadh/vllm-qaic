# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vllm"))
from _ktest import run_test, logger

from vllm.v1.attention.ops.triton_unified_attention import apply_softcap
from vllm.triton_utils import tl, triton

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "apply_softcap (device fn)"


@triton.jit
def _tanh_approx(x):
    p = tl.exp(x); n = tl.exp(-x)
    return (p - n) / (p + n)


@triton.jit
def dev_wrapper(x_ptr, tanh_out, softcap_out, softcap, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(tanh_out + offs, _tanh_approx(x.to(tl.float32)), mask=mask)
    tl.store(softcap_out + offs, apply_softcap(x, softcap), mask=mask)


def main(log_path):
    torch.manual_seed(42)
    n = 32
    x = torch.randn(n, dtype=torch.float32, device=DEVICE) * 3.0
    tanh_o = torch.empty(n, dtype=torch.float32, device=DEVICE)
    soft_o = torch.empty(n, dtype=torch.float32, device=DEVICE)
    softcap = 5.0
    x_r = x.clone()
    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(os.path.basename(log_path)[:-4]); log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  x: shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}")
            log(f"  softcap={softcap}, n={n}")
            dev_wrapper[(1,)](x, tanh_o, soft_o, softcap, n, BLOCK_SIZE=n)
            to = tanh_o.cpu(); so = soft_o.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: (1,), BLOCK_SIZE={n}")
            log("\nOutput:")
            log(f"  tanh mean={to.mean().item():.4f}, softcap mean={so.mean().item():.4f}")
            # references
            ref_tanh = torch.tanh(x_r.cpu()).float()
            # apply_softcap(S=x, x=softcap) = softcap * tanh(x/softcap)
            ref_soft = softcap * torch.tanh(x_r.cpu() / softcap)
            d1 = (to - ref_tanh).abs().max().item()
            d2 = (so - ref_soft).abs().max().item()
            log("\nValidation:")
            log(f"  tanh (tl.math.tanh) max abs diff: {d1:.6f}")
            log(f"  apply_softcap (== softcap*tanh(x/softcap)) max abs diff: {d2:.6f}")
            torch.testing.assert_close(to, ref_tanh, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(so, ref_soft, atol=1e-3, rtol=1e-3)
            log("  assert_close: PASSED")
            log("\nSummary:")
            log("  Kernel execution: SUCCESS (grid (1,))")
            log("  Validation vs PyTorch reference: PASSED")
            log(f"  max abs diff: tanh={d1:.6f}, softcap={d2:.6f}")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Validation vs PyTorch reference: FAILED")
            fp.write("\n" + "-" * 36 + "\n"); sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "tanh_softcap")
