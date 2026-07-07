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

from vllm.model_executor.layers.fused_moe.utils import count_expert_num_tokens
from vllm.triton_utils import triton

# ── Configuration ────────────────────────────────────────────────────────────
# _count_expert_num_tokens counts how many (token, slot) entries in topk_ids map
# to each local expert. Tests the no-expert-map path.
NUM_TOKENS       = 64
TOP_K            = 2
NUM_LOCAL_EXPERTS = 8
DEVICE           = "qaic"

KERNEL_NAME = "_count_expert_num_tokens"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_count_expert_num_tokens_{timestamp}.log")


def reference_count(topk_ids, num_local_experts):
    """out[e] = number of entries in topk_ids equal to e."""
    out = torch.zeros(num_local_experts, dtype=torch.int32)
    flat = topk_ids.reshape(-1)
    for e in range(num_local_experts):
        out[e] = int((flat == e).sum())
    return out


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    torch.manual_seed(42)

    topk_ids = torch.randint(
        0, NUM_LOCAL_EXPERTS, (NUM_TOKENS, TOP_K), dtype=torch.int32, device=DEVICE
    )
    topk_ids_ref = topk_ids.clone()

    grid = (NUM_LOCAL_EXPERTS,)
    BLOCK_SIZE = triton.next_power_of_2(min(topk_ids.numel(), 1024))

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")

            log(f, "\nInputs:")
            log(f, f"  topk_ids : shape={tuple(topk_ids.shape)}, dtype={topk_ids.dtype}, device={topk_ids.device}")
            log(f, f"  num_tokens={NUM_TOKENS}, top_k={TOP_K}, num_local_experts={NUM_LOCAL_EXPERTS}")
            log(f, f"  expert_map: None (no EP remap)")

            out = count_expert_num_tokens(topk_ids, NUM_LOCAL_EXPERTS, expert_map=None)

            out_host = out.cpu()   # force sync

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}  (num_local_experts,)")
            log(f, f"  BLOCK_SIZE: {BLOCK_SIZE}")

            log(f, "\nOutput:")
            log(f, f"  expert_num_tokens : {out_host.tolist()}  (sum={int(out_host.sum())})")

            # ── Reference validation ────────────────────────────────────────
            ref = reference_count(topk_ids_ref, NUM_LOCAL_EXPERTS)
            diff = (out_host - ref).abs().max().item()
            log(f, "\nValidation (vs PyTorch per-expert count):")
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
