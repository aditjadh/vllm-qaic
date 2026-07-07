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

from vllm.v1.attention.ops.triton_merge_attn_states import merge_attn_states

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "merge_attn_states_kernel"


def main(log_path):
    torch.manual_seed(42)
    NUM_TOKENS, NUM_HEADS, HEAD_SIZE = 8, 4, 32
    p_out = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_SIZE, dtype=torch.float32, device=DEVICE)
    s_out = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_SIZE, dtype=torch.float32, device=DEVICE)
    p_lse = torch.randn(NUM_HEADS, NUM_TOKENS, dtype=torch.float32, device=DEVICE)
    s_lse = torch.randn(NUM_HEADS, NUM_TOKENS, dtype=torch.float32, device=DEVICE)
    out = torch.empty(NUM_TOKENS, NUM_HEADS, HEAD_SIZE, dtype=torch.float32, device=DEVICE)

    po_r, so_r, pl_r, sl_r = p_out.clone(), s_out.clone(), p_lse.clone(), s_lse.clone()
    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(os.path.basename(log_path)[:-4]); log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  prefix_output: {tuple(p_out.shape)}, suffix_output: {tuple(s_out.shape)}, device={p_out.device}")
            log(f"  prefix_lse: {tuple(p_lse.shape)}, suffix_lse: {tuple(s_lse.shape)}")
            merge_attn_states(out, p_out, p_lse, s_out, s_lse)
            o = out.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: ({NUM_TOKENS}, {NUM_HEADS})  (num_tokens, num_heads)")
            log("\nOutput:")
            log(f"  out: shape={tuple(o.shape)}, mean={o.mean().item():.4f}")
            # reference: online-softmax merge over the 2 partials
            ref = torch.empty_like(po_r.cpu())
            for t in range(NUM_TOKENS):
                for h in range(NUM_HEADS):
                    pl = float(pl_r[h, t]); sl = float(sl_r[h, t])
                    m = max(pl, sl)
                    pse = torch.exp(torch.tensor(pl - m)); sse = torch.exp(torch.tensor(sl - m))
                    denom = pse + sse
                    ref[t, h] = (po_r[t, h].cpu() * (pse / denom) + so_r[t, h].cpu() * (sse / denom))
            diff = (o - ref).abs().max().item()
            log("\nValidation (vs PyTorch online-softmax merge):")
            log(f"  max abs diff: {diff:.6f}")
            torch.testing.assert_close(o, ref, atol=1e-4, rtol=1e-4)
            log("  assert_close: PASSED")
            log("\nSummary:")
            log(f"  Kernel execution: SUCCESS (grid ({NUM_TOKENS}, {NUM_HEADS}))")
            log("  Validation vs PyTorch reference: PASSED")
            log(f"  max abs diff: {diff:.6f}")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Validation vs PyTorch reference: FAILED")
            fp.write("\n" + "-" * 36 + "\n"); sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "merge_attn_states")
