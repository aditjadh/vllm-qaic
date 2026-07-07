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

from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    _fwd_kernel_ep_scatter_1,
)
from vllm.triton_utils import triton

# ── Configuration ────────────────────────────────────────────────────────────
# scatter_1 computes expert_start_loc = exclusive cumsum of round_up_128(counts),
# and fills m_indices[start : start+count] = expert_id (rest left as -1).
NUM_EXPERTS = 4
COUNTS      = [10, 130, 5, 64]     # tokens per expert (pre-rounding)
BLOCK_E     = 128
DEVICE      = "qaic"

KERNEL_NAME = "_fwd_kernel_ep_scatter_1"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_ep_scatter_1_{timestamp}.log")


def round_up_128(x):
    return ((x + 127) // 128) * 128


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    counts = torch.tensor(COUNTS, dtype=torch.int32, device=DEVICE)
    rounded = [round_up_128(c) for c in COUNTS]
    M_sum = sum(rounded)

    expert_start_loc = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=DEVICE)
    m_indices = torch.full((M_sum,), -1, dtype=torch.int32, device=DEVICE)

    grid = (NUM_EXPERTS,)

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")

            log(f, "\nInputs:")
            log(f, f"  num_recv_tokens_per_expert: {COUNTS}")
            log(f, f"  num_experts={NUM_EXPERTS}, BLOCK_E={BLOCK_E}")
            log(f, f"  rounded(128) counts: {rounded}, M_sum={M_sum}")

            _fwd_kernel_ep_scatter_1[grid](
                counts,
                expert_start_loc,
                m_indices,
                num_experts=NUM_EXPERTS,
                BLOCK_E=BLOCK_E,
                BLOCK_EXPERT_NUM=triton.next_power_of_2(NUM_EXPERTS),
            )

            esl_host = expert_start_loc.cpu()
            mi_host = m_indices.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}  (num_experts,)")
            log(f, f"  BLOCK_E={BLOCK_E}, BLOCK_EXPERT_NUM={triton.next_power_of_2(NUM_EXPERTS)}")

            log(f, "\nOutput:")
            log(f, f"  expert_start_loc: {esl_host.tolist()}")
            log(f, f"  m_indices counts per value:")
            for e in range(-1, NUM_EXPERTS):
                log(f, f"    value {e:2d}: {int((mi_host == e).sum())}")

            # ── Reference validation ────────────────────────────────────────
            ref_start = []
            acc = 0
            for r in rounded:
                ref_start.append(acc)
                acc += r
            ref_start = torch.tensor(ref_start, dtype=torch.int32)

            ref_m = torch.full((M_sum,), -1, dtype=torch.int32)
            for e in range(NUM_EXPERTS):
                s = ref_start[e]
                ref_m[s : s + COUNTS[e]] = e

            diff_start = (esl_host - ref_start).abs().max().item()
            diff_m = (mi_host - ref_m).abs().max().item()
            log(f, "\nValidation (vs PyTorch reference):")
            log(f, f"  expert_start_loc max abs diff: {diff_start}")
            log(f, f"  m_indices max abs diff: {diff_m}")

            torch.testing.assert_close(esl_host, ref_start, atol=0, rtol=0)
            torch.testing.assert_close(mi_host, ref_m, atol=0, rtol=0)
            log(f, "  assert_close: PASSED")

            log(f, "\nSummary:")
            log(f, f"  Kernel execution: SUCCESS (grid {grid})")
            log(f, "  Validation vs PyTorch reference: PASSED")
            log(f, f"  max abs diff: start={diff_start}, m_indices={diff_m}")

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
