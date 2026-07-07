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
from _ktest import run_test, logger

from vllm.v1.attention.backends.flashinfer import _copy_page_indices_kernel

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "_copy_page_indices_kernel"

BLOCK_SIZE = 16


def main(log_path):
    torch.manual_seed(42)
    # 3 requests with different block counts: [2, 3, 1] blocks
    num_reqs = 3
    max_blocks_per_req = 4
    blocks_per_req = [2, 3, 1]
    total_blocks = sum(blocks_per_req)

    # block_table: [num_reqs, max_blocks_per_req]
    block_table = torch.zeros(num_reqs, max_blocks_per_req, dtype=torch.int32, device=DEVICE)
    for i, nb in enumerate(blocks_per_req):
        block_table[i, :nb] = torch.randint(0, 100, (nb,), dtype=torch.int32)

    # cu_num_blocks: cumulative block counts (prefix sum), length = num_reqs + 1
    cu_num_blocks = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    for i, nb in enumerate(blocks_per_req):
        cu_num_blocks[i + 1] = cu_num_blocks[i] + nb

    page_indices = torch.zeros(total_blocks, dtype=torch.int32, device=DEVICE)

    bt_r = block_table.clone()

    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            log(os.path.basename(log_path)[:-4]); log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  num_reqs={num_reqs}, blocks_per_req={blocks_per_req}, total_blocks={total_blocks}")
            log(f"  block_table: {tuple(block_table.shape)}, device={block_table.device}")
            log(f"  cu_num_blocks: {cu_num_blocks.cpu().tolist()}")
            log(f"  BLOCK_SIZE={BLOCK_SIZE}")

            grid = (num_reqs,)
            _copy_page_indices_kernel[grid](
                page_indices,
                block_table,
                block_table.stride(0),
                cu_num_blocks,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            pi = page_indices.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: ({num_reqs},)  (one program per request)")
            log("\nOutput:")
            log(f"  page_indices: {pi.tolist()}")

            # reference: flatten block_table according to blocks_per_req
            ref = []
            for i, nb in enumerate(blocks_per_req):
                ref.extend(bt_r[i, :nb].cpu().tolist())
            ref = torch.tensor(ref, dtype=torch.int32)
            diff = (pi - ref).abs().max().item()
            log("\nValidation (vs flattened block_table):")
            log(f"  expected:      {ref.tolist()}")
            log(f"  max abs diff:  {diff}")
            torch.testing.assert_close(pi, ref)
            log("  assert_close: PASSED")
            log("\nSummary:")
            log(f"  Kernel execution: SUCCESS (grid ({num_reqs},))")
            log("  Validation vs reference: PASSED")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Validation vs reference: FAILED")
            fp.write("\n" + "-" * 36 + "\n"); sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "copy_page_indices")
