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

from vllm.model_executor.layers.fused_moe.deep_gemm_utils import apply_expert_map
from vllm.triton_utils import tl, triton

# ── Configuration ────────────────────────────────────────────────────────────
# apply_expert_map(expert_id, expert_map) is a @triton.jit device function:
#   returns expert_map[expert_id], or passes through -1 unchanged.
# We wrap it in a launchable kernel that maps an array of expert ids elementwise.
NUM_IDS    = 16
NUM_EXPERTS = 8
BLOCK_SIZE = 16
DEVICE     = "qaic"

KERNEL_NAME = "apply_expert_map (via wrapper kernel)"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_apply_expert_map_{timestamp}.log")


@triton.jit
def apply_expert_map_wrapper(
    ids_ptr,
    out_ptr,
    expert_map,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    # apply_expert_map uses `if expert_id != -1`, so it needs a SCALAR id (the
    # way the scatter/gather kernels call it). One program handles one id.
    pid = tl.program_id(0)
    if pid < n:
        expert_id = tl.load(ids_ptr + pid)
        mapped = apply_expert_map(expert_id, expert_map)
        tl.store(out_ptr + pid, mapped)


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    torch.manual_seed(42)

    # ids include some -1 entries (no-expert) which must pass through unchanged.
    ids = torch.randint(0, NUM_EXPERTS, (NUM_IDS,), dtype=torch.int32, device=DEVICE)
    ids[3] = -1
    ids[10] = -1
    # expert_map: permutation/remap of expert indices to local space.
    expert_map = torch.randperm(NUM_EXPERTS, device=DEVICE).to(torch.int32)
    out = torch.empty(NUM_IDS, dtype=torch.int32, device=DEVICE)

    ids_ref = ids.clone()
    expert_map_ref = expert_map.clone()

    grid = (NUM_IDS,)

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")

            log(f, "\nInputs:")
            log(f, f"  ids        : shape={tuple(ids.shape)}, dtype={ids.dtype}, device={ids.device}")
            log(f, f"  ids values : {ids.cpu().tolist()}")
            log(f, f"  expert_map : {expert_map.cpu().tolist()}")
            log(f, f"  num_ids={NUM_IDS}, num_experts={NUM_EXPERTS}, BLOCK_SIZE={BLOCK_SIZE}")

            apply_expert_map_wrapper[grid](
                ids, out, expert_map, NUM_IDS, BLOCK_SIZE=BLOCK_SIZE
            )

            out_host = out.cpu()   # force sync

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}")
            log(f, f"  BLOCK_SIZE: {BLOCK_SIZE}")

            log(f, "\nOutput:")
            log(f, f"  mapped ids : {out_host.tolist()}")

            # ── Reference: out[i] = expert_map[ids[i]] if ids[i] != -1 else -1 ──
            ref = torch.empty(NUM_IDS, dtype=torch.int32)
            for i in range(NUM_IDS):
                e = int(ids_ref[i])
                ref[i] = -1 if e == -1 else int(expert_map_ref[e])
            diff = (out_host - ref).abs().max().item()
            log(f, "\nValidation (vs PyTorch expert_map[id] with -1 passthrough):")
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
