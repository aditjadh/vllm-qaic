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

from vllm.v1.attention.ops.common import _correct_attn_cp_out_kernel, correct_attn_out, CPTritonContext

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "_correct_attn_cp_out_kernel"


def main(log_path):
    torch.manual_seed(42)
    B, H, D = 4, 2, 32
    N = 3   # CP world size (num ranks)
    out = torch.randn(B, H, D, dtype=torch.float32, device=DEVICE)
    lses = torch.randn(N, B, H, dtype=torch.float32, device=DEVICE)
    cp_rank = 1

    out_r, lses_r = out.clone(), lses.clone()
    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(os.path.basename(log_path)[:-4]); log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  out: shape={tuple(out.shape)} [B,H,D], device={out.device}")
            log(f"  lses: shape={tuple(lses.shape)} [N,B,H], cp_rank={cp_rank}")
            ctx = CPTritonContext()
            new_out, lse = correct_attn_out(out, lses, cp_rank, ctx, is_lse_base_on_e=True)
            no = new_out.cpu(); le = lse.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: ({B}, {H}, 1)")
            log("\nOutput:")
            log(f"  corrected out: shape={tuple(no.shape)}, mean={no.mean().item():.4f}")
            log(f"  final lse: shape={tuple(le.shape)}, mean={le.mean().item():.4f}")
            # reference
            ref_out = torch.empty(B, H, D)
            ref_lse = torch.empty(B, H)
            for b in range(B):
                for h in range(H):
                    lse_vec = lses_r[:, b, h].cpu().clone()
                    lse_vec = torch.where((lse_vec != lse_vec) | torch.isinf(lse_vec),
                                          torch.tensor(-float("inf")), lse_vec)
                    lmax = lse_vec.max()
                    lmax = torch.where(lmax == -float("inf"), torch.tensor(0.0), lmax)
                    acc = torch.exp(lse_vec - lmax).sum()
                    final_lse = torch.log(acc) + lmax
                    ref_lse[b, h] = final_lse
                    factor = torch.exp(lses_r[cp_rank, b, h].cpu() - final_lse)
                    ref_out[b, h] = out_r[b, h].cpu() * factor
            d_o = (no - ref_out).abs().max().item()
            d_l = (le - ref_lse).abs().max().item()
            log("\nValidation (vs PyTorch CP lse-correction):")
            log(f"  out max abs diff: {d_o:.6f}, lse max abs diff: {d_l:.6f}")
            torch.testing.assert_close(no, ref_out, atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(le, ref_lse, atol=1e-3, rtol=1e-3)
            log("  assert_close: PASSED")
            log("\nSummary:")
            log(f"  Kernel execution: SUCCESS (grid ({B}, {H}, 1))")
            log("  Validation vs PyTorch reference: PASSED")
            log(f"  max abs diff: out={d_o:.6f}, lse={d_l:.6f}")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Validation vs PyTorch reference: FAILED")
            fp.write("\n" + "-" * 36 + "\n"); sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "correct_attn_cp_out")
