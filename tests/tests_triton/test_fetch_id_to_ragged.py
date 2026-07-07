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

from vllm.triton_utils import tl, triton

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "fetch_id_to_ragged_kernel"


@triton.jit
def fetch_id_to_ragged_kernel(
    in_tensor_ptr,
    cumsum_ptr,
    out_tensor_ptr,
    in_tensor_ptr_stride,
    TOPK: tl.constexpr,
    TOKEN_NUM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    seq_id = tl.program_id(0)
    block_id = tl.program_id(1)
    offset = tl.arange(0, BLOCK_SIZE)
    token_start = tl.load(cumsum_ptr + seq_id)
    token_end = tl.load(cumsum_ptr + seq_id + 1)
    token_num = token_end - token_start
    row_offset = block_id * BLOCK_SIZE
    if row_offset >= token_num:
        return
    in_tensor_offset = seq_id * in_tensor_ptr_stride + row_offset + offset
    in_tensor_mask = (row_offset + offset) < TOPK
    in_tensor_val = tl.load(in_tensor_ptr + in_tensor_offset, mask=in_tensor_mask)
    out_tensor_offset = token_start + row_offset + offset
    out_tensor_mask = (out_tensor_offset < token_end) & in_tensor_mask
    tl.store(out_tensor_ptr + out_tensor_offset, in_tensor_val, mask=out_tensor_mask)


def main(log_path):
    torch.manual_seed(42)
    # 3 sequences with different valid token counts, topk=8 per seq
    TOPK = 8
    BLOCK_SIZE = 8  # must be power-of-2 and equal to TOPK here for simplicity
    num_tokens = 3  # number of sequences

    # in_tensor: [num_tokens, TOPK] — each row has TOPK candidate indices
    in_tensor = torch.randint(0, 100, (num_tokens, TOPK), dtype=torch.int32, device=DEVICE)

    # cumsum: how many valid tokens each seq contributes to the ragged output
    # seq 0: 3 valid, seq 1: 5 valid, seq 2: 2 valid
    valid_counts = torch.tensor([3, 5, 2], dtype=torch.int32)
    cumsum = torch.zeros(num_tokens + 1, dtype=torch.int32, device=DEVICE)
    cumsum[1:] = torch.cumsum(valid_counts, dim=0).to(torch.int32)

    total_out = int(cumsum[-1].item())
    out_tensor = torch.zeros(total_out, dtype=torch.int32, device=DEVICE)

    num_block_per_row = triton.cdiv(TOPK, BLOCK_SIZE)
    grid = (num_tokens, num_block_per_row)

    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  in_tensor: {tuple(in_tensor.shape)} [num_seqs,TOPK], device={in_tensor.device}")
            log(f"  cumsum: {cumsum.cpu().tolist()}")
            log(f"  valid_counts: {valid_counts.tolist()}, total_out={total_out}")
            log(f"  TOPK={TOPK}, BLOCK_SIZE={BLOCK_SIZE}")

            fetch_id_to_ragged_kernel[grid](
                in_tensor,
                cumsum,
                out_tensor,
                in_tensor.stride(0),
                TOPK=TOPK,
                TOKEN_NUM=num_tokens,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=1,
                num_stages=1,
            )

            ot = out_tensor.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: {grid}  (num_seqs={num_tokens}, blocks_per_row={num_block_per_row})")
            log(f"  TOPK={TOPK} (constexpr), BLOCK_SIZE={BLOCK_SIZE} (constexpr)")
            log("\nOutput:")
            log(f"  out_tensor: {ot.tolist()}")

            # reference: copy first valid_count[i] entries from in_tensor[i]
            in_cpu = in_tensor.cpu()
            ref = []
            for i in range(num_tokens):
                cnt = valid_counts[i].item()
                ref.extend(in_cpu[i, :cnt].tolist())
            ref = torch.tensor(ref, dtype=torch.int32)
            max_diff = (ot - ref).abs().max().item()
            log("\nValidation (vs manual scatter reference):")
            log(f"  expected: {ref.tolist()}")
            log(f"  max abs diff: {max_diff}")
            assert max_diff == 0, f"mismatch: {max_diff}"
            log("  assert: PASSED")
            log("\nSummary:")
            log(f"  Kernel execution: SUCCESS (grid {grid})")
            log("  Validation vs reference: PASSED")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Kernel execution: FAILED")
            fp.write("\n" + "-" * 36 + "\n")
            sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "fetch_id_to_ragged")
