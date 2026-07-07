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

from vllm.model_executor.layers.batch_invariant import _compute_pid
from vllm.triton_utils import tl, triton

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "_compute_pid"


@triton.jit
def compute_pid_wrapper(out_m_ptr, out_n_ptr, num_pid_in_group, num_pid_m,
                        n, GROUP_SIZE_M: tl.constexpr, NUM_SMS: tl.constexpr,
                        BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid < n:
        pm, pn = _compute_pid(pid, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        tl.store(out_m_ptr + pid, pm)
        tl.store(out_n_ptr + pid, pn)


def main(log_path):
    # Grouped-GEMM tile-id -> (pid_m, pid_n) mapping.
    num_pid_m = 4
    num_pid_n = 5
    GROUP_SIZE_M = 2
    NUM_SMS = 8
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    n = num_pid_m * num_pid_n

    out_m = torch.full((n,), -1, dtype=torch.int32, device=DEVICE)
    out_n = torch.full((n,), -1, dtype=torch.int32, device=DEVICE)

    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(os.path.basename(log_path)[:-4]); log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  num_pid_m={num_pid_m}, num_pid_n={num_pid_n}, GROUP_SIZE_M={GROUP_SIZE_M}, NUM_SMS={NUM_SMS}")
            log(f"  num_tiles={n}")

            compute_pid_wrapper[(n,)](out_m, out_n, num_pid_in_group, num_pid_m, n,
                                      GROUP_SIZE_M=GROUP_SIZE_M, NUM_SMS=NUM_SMS,
                                      BLOCK_SIZE=1)
            om = out_m.cpu(); on = out_n.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: ({n},) (one program per tile id)")
            log("\nOutput:")
            log(f"  pid_m={om.tolist()}")
            log(f"  pid_n={on.tolist()}")

            # reference (same arithmetic)
            ref_m = []; ref_n = []
            for t in range(n):
                gid = t // num_pid_in_group
                first_pid_m = gid * GROUP_SIZE_M
                gsm = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                ref_m.append(first_pid_m + (t % gsm))
                ref_n.append((t % num_pid_in_group) // gsm)
            d_m = (om - torch.tensor(ref_m, dtype=torch.int32)).abs().max().item()
            d_n = (on - torch.tensor(ref_n, dtype=torch.int32)).abs().max().item()
            log("\nValidation:")
            log(f"  ref pid_m={ref_m}")
            log(f"  ref pid_n={ref_n}")
            log(f"  max abs diff: pid_m={d_m}, pid_n={d_n}")
            assert d_m == 0 and d_n == 0
            log("  validation: PASSED")
            log("\nSummary:")
            log(f"  Kernel execution: SUCCESS (grid ({n},))")
            log("  Validation vs PyTorch reference: PASSED")
            log(f"  max abs diff: pid_m={d_m}, pid_n={d_n}")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Validation vs PyTorch reference: FAILED")
            fp.write("\n" + "-" * 36 + "\n"); sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "compute_pid")
