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

from vllm.v1.attention.ops.triton_unified_attention import reduce_segments
from vllm.triton_utils import triton

# ── Configuration ────────────────────────────────────────────────────────────
# reduce_segments combines per-segment flash-attention partials (segm_output,
# segm_max, segm_expsum) into a final output via online-softmax rescaling.
NUM_TOKENS       = 1
NUM_QUERY_HEADS  = 2
HEAD_SIZE        = 32
HEAD_SIZE_PADDED = 32
NUM_SEGMENTS     = 4
TILE_SIZE        = 16
BLOCK_Q          = 16
SEQ_LEN          = 64       # -> with TILE_SIZE/segments, act_num_segments = 4
DEVICE           = "qaic"

KERNEL_NAME = "reduce_segments"
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = os.path.join(LOG_DIR, f"test_reduce_segments_{timestamp}.log")


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    torch.manual_seed(42)

    # segm tensors: [num_tokens, num_query_heads, num_segments(, head)]
    segm_output = torch.randn(NUM_TOKENS, NUM_QUERY_HEADS, NUM_SEGMENTS, HEAD_SIZE_PADDED,
                              dtype=torch.float32, device=DEVICE)
    segm_max = torch.randn(NUM_TOKENS, NUM_QUERY_HEADS, NUM_SEGMENTS,
                           dtype=torch.float32, device=DEVICE)
    segm_expsum = torch.rand(NUM_TOKENS, NUM_QUERY_HEADS, NUM_SEGMENTS,
                             dtype=torch.float32, device=DEVICE) + 0.1

    output = torch.zeros(NUM_TOKENS, NUM_QUERY_HEADS, HEAD_SIZE, dtype=torch.float32, device=DEVICE)
    seq_lens = torch.tensor([SEQ_LEN], dtype=torch.int32, device=DEVICE)
    qsl = torch.tensor([0, NUM_TOKENS], dtype=torch.int32, device=DEVICE)

    so_r, sm_r, se_r = segm_output.clone(), segm_max.clone(), segm_expsum.clone()

    grid = (NUM_TOKENS, NUM_QUERY_HEADS)
    with open(log_path, "w") as f:
        try:
            log(f, timestamp); log(f, f"Kernel: {KERNEL_NAME}")
            log(f, "\nInputs:")
            log(f, f"  segm_output : shape={tuple(segm_output.shape)}, device={segm_output.device}")
            log(f, f"  segm_max    : shape={tuple(segm_max.shape)}")
            log(f, f"  segm_expsum : shape={tuple(segm_expsum.shape)}")
            log(f, f"  num_tokens={NUM_TOKENS}, num_query_heads={NUM_QUERY_HEADS}, "
                   f"head_size={HEAD_SIZE}, num_segments={NUM_SEGMENTS}, seq_len={SEQ_LEN}")

            reduce_segments[grid](
                output, segm_output, segm_max, segm_expsum,
                seq_lens, 1, NUM_QUERY_HEADS,
                1.0,                          # out_scale_inv
                output.stride(0), output.stride(1),
                0,                            # block_table_stride (unused)
                TILE_SIZE, HEAD_SIZE, HEAD_SIZE_PADDED,
                qsl, BLOCK_Q, NUM_SEGMENTS, False,   # USE_FP8
            )
            out_h = output.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}  (num_tokens, num_query_heads)")
            log(f, "\nOutput:")
            log(f, f"  output : shape={tuple(output.shape)}, mean={out_h.mean().item():.4f}")

            # ── Reference: online-softmax combine over all NUM_SEGMENTS ──────
            # With seq_len=64, tiles_per_segment=cdiv(64,4*16)=1,
            # act_num_segments=cdiv(64,1*16)=4 -> all 4 segments active.
            ref = torch.zeros(NUM_TOKENS, NUM_QUERY_HEADS, HEAD_SIZE)
            for t in range(NUM_TOKENS):
                for h in range(NUM_QUERY_HEADS):
                    sm = sm_r[t, h].cpu()
                    se = se_r[t, h].cpu()
                    so = so_r[t, h].cpu()[:, :HEAD_SIZE]
                    omax = sm.max()
                    se_r2 = se * torch.exp(sm - omax)
                    osum = se_r2.sum()
                    acc = (so * torch.exp(sm - omax)[:, None]).sum(dim=0)
                    ref[t, h] = torch.where(torch.tensor(osum == 0.0), torch.zeros(HEAD_SIZE), acc / osum)
            diff = (out_h - ref).abs().max().item()
            log(f, "\nValidation (vs PyTorch online-softmax segment combine):")
            log(f, f"  max abs diff: {diff:.6f}")
            torch.testing.assert_close(out_h, ref, atol=1e-3, rtol=1e-3)
            log(f, "  assert_close: PASSED")

            log(f, "\nSummary:")
            log(f, f"  Kernel execution: SUCCESS (grid {grid})")
            log(f, "  Validation vs PyTorch reference: PASSED")
            log(f, f"  max abs diff: {diff:.6f}")
        except Exception:
            log(f, f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log(f, "\nSummary:\n  Validation vs PyTorch reference: FAILED")
            sys.exit(1)
        finally:
            log(f, "\n" + "-" * 36)
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
