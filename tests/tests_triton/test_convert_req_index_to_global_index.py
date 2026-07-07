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

from vllm.v1.attention.backends.mla.flashmla_sparse import (
    _convert_req_index_to_global_index_kernel,
    triton_convert_req_index_to_global_index,
)

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "_convert_req_index_to_global_index_kernel"


def main(log_path):
    torch.manual_seed(42)
    num_tokens = 4
    num_requests = 4
    max_blocks = 8
    BLOCK_SIZE = 4
    NUM_TOPK = 8
    BLOCK_N = 8  # must divide NUM_TOPK

    # req_id_per_token: which request each token belongs to
    req_id = torch.tensor([0, 0, 1, 2], dtype=torch.int32, device=DEVICE)

    # block_table: [num_requests, max_blocks] — physical block ids
    block_table = torch.randint(0, 20, (num_requests, max_blocks),
                                dtype=torch.int32, device=DEVICE)

    # token_indices: per-token logical token indices [num_tokens, NUM_TOPK]
    token_indices = torch.randint(0, BLOCK_SIZE * max_blocks,
                                  (num_tokens, NUM_TOPK),
                                  dtype=torch.int32, device=DEVICE)

    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  req_id: {req_id.cpu().tolist()}, device={req_id.device}")
            log(f"  block_table: {tuple(block_table.shape)} [num_requests,max_blocks]")
            log(f"  token_indices: {tuple(token_indices.shape)} [num_tokens,NUM_TOPK]")
            log(f"  BLOCK_SIZE={BLOCK_SIZE}, NUM_TOPK={NUM_TOPK}, BLOCK_N={BLOCK_N}")

            out = triton_convert_req_index_to_global_index(
                req_id=req_id,
                block_table=block_table,
                token_indices=token_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                NUM_TOPK_TOKENS=NUM_TOPK,
                BLOCK_N=BLOCK_N,
                HAS_PREFILL_WORKSPACE=False,
            )

            o = out.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: ({num_tokens}, {NUM_TOPK // BLOCK_N})  (tokens x column-tiles)")
            log(f"  BLOCK_SIZE={BLOCK_SIZE} (constexpr), BLOCK_N={BLOCK_N} (constexpr)")
            log("\nOutput:")
            log(f"  out: shape={tuple(o.shape)}, dtype={o.dtype}")
            log(f"  sample row 0: {o[0].tolist()}")

            # reference: compute expected global indices
            bt_cpu = block_table.cpu()
            ti_cpu = token_indices.cpu()
            req_cpu = req_id.cpu()
            ref = torch.zeros_like(ti_cpu)
            for t in range(num_tokens):
                r = req_cpu[t].item()
                for j in range(NUM_TOPK):
                    tok = ti_cpu[t, j].item()
                    if tok < 0:
                        ref[t, j] = -1
                        continue
                    blk = tok // BLOCK_SIZE
                    off = tok % BLOCK_SIZE
                    if blk >= max_blocks:
                        ref[t, j] = -1
                        continue
                    phys = bt_cpu[r, blk].item()
                    ref[t, j] = phys * BLOCK_SIZE + off
            max_diff = (o - ref).abs().max().item()
            log("\nValidation (vs manual reference):")
            log(f"  max abs diff: {max_diff}")
            assert max_diff == 0, f"mismatch: {max_diff}"
            log("  assert: PASSED")
            log("\nSummary:")
            log(f"  Kernel execution: SUCCESS")
            log("  Validation vs reference: PASSED")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Kernel execution: FAILED")
            fp.write("\n" + "-" * 36 + "\n")
            sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "convert_req_index_to_global_index")
