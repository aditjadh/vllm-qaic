# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
import os
import sys
import traceback
from datetime import datetime

import torch
import triton

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vllm"))
from _ktest import run_test, logger

from vllm.v1.attention.ops.chunked_prefill_paged_decode import kernel_paged_attention_2d

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "kernel_paged_attention_2d"


def main(log_path):
    torch.manual_seed(42)
    num_seqs = 2
    num_query_heads = 4
    num_kv_heads = 4
    head_size = 64
    block_size = 16
    seq_lens_list = [32, 48]
    max_seq_len = max(seq_lens_list)
    max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    total_blocks = num_seqs * max_blocks_per_seq
    x = 8

    # query: [num_seqs, num_query_heads, head_size]  (one decode token per seq)
    query = torch.randn(num_seqs, num_query_heads, head_size, dtype=torch.float16, device=DEVICE)
    output = torch.zeros_like(query)

    # key_cache: [total_blocks, num_kv_heads, head_size//x, block_size, x]
    key_cache = torch.randn(total_blocks, num_kv_heads, head_size // x, block_size, x,
                            dtype=torch.float16, device=DEVICE)
    # value_cache: [total_blocks, num_kv_heads, head_size, block_size]
    value_cache = torch.randn(total_blocks, num_kv_heads, head_size, block_size,
                              dtype=torch.float16, device=DEVICE)

    # block_tables: [num_seqs, max_blocks_per_seq]
    block_tables = torch.zeros(num_seqs, max_blocks_per_seq, dtype=torch.int32, device=DEVICE)
    for i in range(num_seqs):
        nb = (seq_lens_list[i] + block_size - 1) // block_size
        block_tables[i, :nb] = torch.arange(i * max_blocks_per_seq,
                                             i * max_blocks_per_seq + nb,
                                             dtype=torch.int32)

    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=DEVICE)
    k_scale = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
    v_scale = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
    out_scale_inv = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
    sink_ptr = torch.full((num_query_heads,), float("-inf"), dtype=torch.float32, device=DEVICE)
    alibi_slopes_ptr = torch.zeros(num_query_heads, dtype=torch.float32, device=DEVICE)
    query_start_len = torch.arange(num_seqs + 1, dtype=torch.int32, device=DEVICE)

    scale = 1.0 / (head_size ** 0.5)
    head_size_padded = triton.next_power_of_2(head_size)
    num_queries_per_kv = num_query_heads // num_kv_heads
    num_queries_per_kv_padded = max(triton.next_power_of_2(num_queries_per_kv), 16)

    grid = (num_seqs, num_kv_heads)

    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            log(os.path.basename(log_path)[:-4]); log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  query: {tuple(query.shape)} [num_seqs, num_q_heads, head_size], device={query.device}")
            log(f"  key_cache: {tuple(key_cache.shape)}, value_cache: {tuple(value_cache.shape)}")
            log(f"  num_seqs={num_seqs}, num_query_heads={num_query_heads}, num_kv_heads={num_kv_heads}")
            log(f"  head_size={head_size}, block_size={block_size}, seq_lens={seq_lens_list}")

            kernel_paged_attention_2d[grid](
                output,
                query,
                key_cache,
                value_cache,
                sink_ptr,
                block_tables,
                seq_lens,
                alibi_slopes_ptr,
                scale,
                k_scale,
                v_scale,
                out_scale_inv,
                num_query_heads=num_query_heads,
                num_queries_per_kv=num_queries_per_kv,
                num_queries_per_kv_padded=num_queries_per_kv_padded,
                block_table_stride=block_tables.stride(0),
                query_stride_0=query.stride(0),
                query_stride_1=query.stride(1),
                output_stride_0=output.stride(0),
                output_stride_1=output.stride(1),
                BLOCK_SIZE=block_size,
                PHYSICAL_BLOCK_SIZE=block_size,
                HEAD_SIZE=head_size,
                HEAD_SIZE_PADDED=head_size_padded,
                USE_ALIBI_SLOPES=False,
                SLIDING_WINDOW=0,
                x=x,
                stride_k_cache_0=key_cache.stride(0),
                stride_k_cache_1=key_cache.stride(1),
                stride_k_cache_2=key_cache.stride(2),
                stride_k_cache_3=key_cache.stride(3),
                stride_k_cache_4=key_cache.stride(4),
                stride_v_cache_0=value_cache.stride(0),
                stride_v_cache_1=value_cache.stride(1),
                stride_v_cache_2=value_cache.stride(2),
                stride_v_cache_3=value_cache.stride(3),
                filter_by_query_len=False,
                query_start_len_ptr=query_start_len,
                USE_SINKS=False,
                USE_FP8=False,
            )
            oc = output.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: ({num_seqs}, {num_kv_heads})  (num_seqs, num_kv_heads)")
            log("\nOutput:")
            log(f"  output: shape={tuple(oc.shape)}, mean={oc.float().mean().item():.4f}")
            log("\nSummary:")
            log(f"  Kernel execution: SUCCESS (grid ({num_seqs},{num_kv_heads}))")
            log(f"  Output shape: {list(oc.shape)}")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Kernel execution: FAILED")
            fp.write("\n" + "-" * 36 + "\n"); sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "kernel_paged_attention_2d")
