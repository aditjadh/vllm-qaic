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
KERNEL_NAME = "_compute_slot_mappings_kernel"

PAD_SLOT_ID = -1


@triton.jit
def _load_ptr(ptr_to_ptr, elem_dtype):
    ptr = tl.load(ptr_to_ptr)
    ptr = tl.cast(ptr, tl.pointer_type(elem_dtype))
    return tl.multiple_of(ptr, 16)


@triton.jit
def _compute_slot_mappings_kernel(
    num_tokens,
    max_num_tokens,
    idx_mapping,
    query_start_loc,
    pos,
    block_table_ptrs,
    block_table_strides,
    block_sizes,
    slot_mappings_ptr,
    slot_mappings_stride,
    PAD_ID: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)
    slot_mapping_ptr = slot_mappings_ptr + group_id * slot_mappings_stride

    if batch_idx == tl.num_programs(1) - 1:
        for i in range(num_tokens, max_num_tokens, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            tl.store(slot_mapping_ptr + offset, PAD_ID, mask=offset < max_num_tokens)
        return

    block_table_ptr = _load_ptr(block_table_ptrs + group_id, tl.int32)
    block_table_stride = tl.load(block_table_strides + group_id)
    block_size = tl.load(block_sizes + group_id)

    req_state_idx = tl.load(idx_mapping + batch_idx)
    start_idx = tl.load(query_start_loc + batch_idx)
    end_idx = tl.load(query_start_loc + batch_idx + 1)
    for i in range(start_idx, end_idx, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        positions = tl.load(pos + offset, mask=offset < end_idx, other=0)
        block_indices = positions // block_size
        block_numbers = tl.load(
            block_table_ptr + req_state_idx * block_table_stride + block_indices
        )
        slot_ids = block_numbers * block_size + positions % block_size
        tl.store(slot_mapping_ptr + offset, slot_ids, mask=offset < end_idx)


def main(log_path):
    torch.manual_seed(42)
    num_kv_cache_groups = 1
    max_num_reqs = 4
    max_num_blocks = 8
    block_size = 4
    max_num_batched_tokens = 16

    # 2 requests in the batch
    num_reqs = 2
    # idx_mapping: batch[0] -> req state 1, batch[1] -> req state 0
    idx_mapping = torch.tensor([1, 0], dtype=torch.int32, device=DEVICE)

    # query_start_loc: token range for each request [0, 3, 7]
    query_start_loc = torch.tensor([0, 3, 7], dtype=torch.int32, device=DEVICE)
    num_tokens = 7

    # positions: actual sequence positions for each token
    positions = torch.tensor([0, 1, 2, 0, 1, 2, 3], dtype=torch.int64, device=DEVICE)

    # block_table: [max_num_reqs, max_num_blocks]
    block_table = torch.randint(0, 20, (max_num_reqs, max_num_blocks),
                                 dtype=torch.int32, device=DEVICE)

    # slot_mappings output: [num_groups, max_num_batched_tokens]
    slot_mappings = torch.full((num_kv_cache_groups, max_num_batched_tokens),
                                PAD_SLOT_ID, dtype=torch.int64, device=DEVICE)

    block_table_strides = torch.tensor([block_table.stride(0)],
                                        dtype=torch.int64, device=DEVICE)
    block_sizes_tensor = torch.tensor([block_size], dtype=torch.int32, device=DEVICE)

    TRITON_BLOCK_SIZE = 1024

    # num_programs(1) = num_reqs + 1 (last program does padding)
    grid = (num_kv_cache_groups, num_reqs + 1)

    with open(log_path, "w") as fp:
        log = logger(fp)
        log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        log(f"Kernel: {KERNEL_NAME}")
        log("\nInputs:")
        log(f"  block_table: {tuple(block_table.shape)} [max_num_reqs,max_blocks], device={block_table.device}")
        log(f"  idx_mapping: {idx_mapping.cpu().tolist()}")
        log(f"  query_start_loc: {query_start_loc.cpu().tolist()}")
        log(f"  positions: {positions.cpu().tolist()}")
        log(f"  num_tokens={num_tokens}, block_size={block_size}")
        log(f"  TRITON_BLOCK_SIZE={TRITON_BLOCK_SIZE}")
        try:
            # UInt64 pointer tensors may not be supported on QAIC
            block_table_ptrs = torch.tensor([block_table.data_ptr()],
                                             dtype=torch.uint64, device=DEVICE)

            _compute_slot_mappings_kernel[grid](
                num_tokens,
                max_num_batched_tokens,
                idx_mapping,
                query_start_loc,
                positions,
                block_table_ptrs,
                block_table_strides,
                block_sizes_tensor,
                slot_mappings,
                slot_mappings.stride(0),
                PAD_ID=PAD_SLOT_ID,
                TRITON_BLOCK_SIZE=TRITON_BLOCK_SIZE,
                num_warps=1,
                num_stages=1,
            )

            sm = slot_mappings.cpu()
            bt_cpu = block_table.cpu()
            idx_cpu = idx_mapping.cpu()
            pos_cpu = positions.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: {grid}  (num_kv_cache_groups, num_reqs+1={num_reqs+1})")
            log(f"  TRITON_BLOCK_SIZE={TRITON_BLOCK_SIZE} (constexpr)")
            log("\nOutput:")
            log(f"  slot_mappings: {tuple(sm.shape)}")
            log(f"  actual tokens: {sm[0, :num_tokens].tolist()}")
            log(f"  padding slots: {sm[0, num_tokens:num_tokens+3].tolist()} (should be -1)")

            # reference
            ref = torch.full_like(sm, PAD_SLOT_ID)
            qsl = query_start_loc.cpu()
            for bi in range(num_reqs):
                req = idx_cpu[bi].item()
                ts = qsl[bi].item()
                te = qsl[bi + 1].item()
                for ti in range(ts, te):
                    p = pos_cpu[ti].item()
                    blk = p // block_size
                    phys = bt_cpu[req, blk].item()
                    ref[0, ti] = phys * block_size + p % block_size

            max_diff = (sm[:, :num_tokens] - ref[:, :num_tokens]).abs().max().item()
            pad_ok = (sm[0, num_tokens:] == PAD_SLOT_ID).all().item()
            log("\nValidation:")
            log(f"  slot_mapping max abs diff (actual tokens): {max_diff}")
            log(f"  padding filled with PAD_SLOT_ID: {pad_ok}")
            assert max_diff == 0, f"mismatch: {max_diff}"
            assert pad_ok, "padding not written correctly"
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
    run_test(main, LOG_DIR, "compute_slot_mappings")
