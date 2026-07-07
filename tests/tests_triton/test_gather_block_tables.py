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
KERNEL_NAME = "_gather_block_tables_kernel"


@triton.jit
def _load_ptr(ptr_to_ptr, elem_dtype):
    ptr = tl.load(ptr_to_ptr)
    ptr = tl.cast(ptr, tl.pointer_type(elem_dtype))
    return tl.multiple_of(ptr, 16)


@triton.jit
def _gather_block_tables_kernel(
    batch_idx_to_req_idx,
    src_block_table_ptrs,
    dst_block_table_ptrs,
    block_table_strides,
    num_blocks_ptr,
    num_blocks_stride,
    BLOCK_SIZE: tl.constexpr,
):
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)
    req_idx = tl.load(batch_idx_to_req_idx + batch_idx)

    group_num_blocks_ptr = num_blocks_ptr + group_id * num_blocks_stride
    num_blocks = tl.load(group_num_blocks_ptr + req_idx)

    stride = tl.load(block_table_strides + group_id)
    src_block_table_ptr = _load_ptr(src_block_table_ptrs + group_id, tl.int32)
    src_row_ptr = src_block_table_ptr + req_idx * stride
    dst_block_table_ptr = _load_ptr(dst_block_table_ptrs + group_id, tl.int32)
    dst_row_ptr = dst_block_table_ptr + batch_idx * stride

    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        block_ids = tl.load(src_row_ptr + offset, mask=offset < num_blocks)
        tl.store(dst_row_ptr + offset, block_ids, mask=offset < num_blocks)


def main(log_path):
    torch.manual_seed(42)
    num_kv_cache_groups = 1
    max_num_reqs = 4
    max_num_blocks = 8
    batch_size = 3

    # src_block_table: [max_num_reqs, max_num_blocks]
    src_bt = torch.randint(0, 100, (max_num_reqs, max_num_blocks),
                            dtype=torch.int32, device=DEVICE)

    # dst_block_table: [max_num_reqs, max_num_blocks] (output — same shape)
    dst_bt = torch.zeros_like(src_bt)

    # idx_mapping: batch index -> request index
    idx_mapping = torch.tensor([2, 0, 1], dtype=torch.int32, device=DEVICE)

    # num_blocks: [num_groups, max_num_reqs] — how many blocks each req has
    num_blocks_arr = torch.tensor([[3, 5, 2, 4]], dtype=torch.int32, device=DEVICE)

    # Pointer tensors for src/dst (uint64) — may not be supported on QAIC
    BLOCK_SIZE = 1024
    grid = (num_kv_cache_groups, batch_size)

    with open(log_path, "w") as fp:
        log = logger(fp)
        log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        log(f"Kernel: {KERNEL_NAME}")
        log("\nInputs:")
        log(f"  src_block_table: {tuple(src_bt.shape)} [max_num_reqs,max_blocks], device={src_bt.device}")
        log(f"  idx_mapping (batch->req): {idx_mapping.cpu().tolist()}")
        log(f"  num_blocks per req: {num_blocks_arr.cpu().tolist()}")
        log(f"  num_kv_cache_groups={num_kv_cache_groups}, batch_size={batch_size}")
        log(f"  BLOCK_SIZE={BLOCK_SIZE}")
        try:
            src_ptrs = torch.tensor([src_bt.data_ptr()], dtype=torch.uint64, device=DEVICE)
            dst_ptrs = torch.tensor([dst_bt.data_ptr()], dtype=torch.uint64, device=DEVICE)
            strides = torch.tensor([src_bt.stride(0)], dtype=torch.int64, device=DEVICE)

            _gather_block_tables_kernel[grid](
                idx_mapping,
                src_ptrs,
                dst_ptrs,
                strides,
                num_blocks_arr,
                num_blocks_arr.stride(0),
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=1,
                num_stages=1,
            )

            db = dst_bt.cpu()
            sb = src_bt.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: {grid}  (num_kv_cache_groups, batch_size)")
            log(f"  BLOCK_SIZE={BLOCK_SIZE} (constexpr)")
            log("\nOutput:")
            log(f"  dst_block_table: {tuple(db.shape)}")
            for bi in range(batch_size):
                req = idx_mapping[bi].item()
                n = num_blocks_arr[0, req].item()
                log(f"  batch[{bi}] <- req[{req}]: dst={db[bi,:n].tolist()}")

            # reference
            ref = torch.zeros_like(db)
            for bi in range(batch_size):
                req = idx_mapping[bi].item()
                n = num_blocks_arr[0, req].item()
                ref[bi, :n] = sb[req, :n]
            max_diff = (db - ref).abs().max().item()
            log("\nValidation (vs manual gather):")
            log(f"  max abs diff: {max_diff}")
            assert max_diff == 0, f"mismatch: {max_diff}"
            log("  assert: PASSED")
            log("\nSummary:")
            log(f"  Kernel execution: SUCCESS (grid {grid})")
            log("  Validation vs reference: PASSED")
        except RuntimeError as e:
            if "Unsupported datatype" in str(e) or "UInt64" in str(e):
                log(f"\nStatus: SKIPPED")
                log(f"\nReason: QAIC hardware limitation — {e}")
                log("\nSummary:\n  Kernel execution: SKIPPED (hardware does not support UInt64 pointer tensors)")
            else:
                log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
                log("\nSummary:\n  Kernel execution: FAILED")
                fp.write("\n" + "-" * 36 + "\n")
                sys.exit(1)
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Kernel execution: FAILED")
            fp.write("\n" + "-" * 36 + "\n")
            sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "gather_block_tables")
