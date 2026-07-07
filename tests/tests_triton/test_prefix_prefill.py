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

from vllm.v1.attention.ops.prefix_prefill import context_attention_fwd

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "_fwd_kernel / _fwd_kernel_alibi"


def main(log_path):
    torch.manual_seed(42)
    # Single sequence prefill: batch=1, heads=4, seq=32, dim=64
    batch = 1
    num_heads = 4
    seq_len = 32
    head_dim = 64
    block_size = 16
    num_blocks = (seq_len + block_size - 1) // block_size

    # q, k, v: [total_tokens, num_heads, head_dim]
    total_tokens = seq_len
    q = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.float16, device=DEVICE)
    k = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.float16, device=DEVICE)
    v = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.float16, device=DEVICE)
    o = torch.empty_like(q)

    # k_cache/v_cache: [num_blocks, num_heads, head_dim//x, block_size, x]
    x = 8
    k_cache = torch.zeros(num_blocks, num_heads, head_dim // x, block_size, x,
                          dtype=torch.float16, device=DEVICE)
    v_cache = torch.zeros(num_blocks, num_heads, head_dim, block_size,
                          dtype=torch.float16, device=DEVICE)

    # block table: [batch, max_blocks_per_seq]  (logical → physical mapping)
    b_loc = torch.arange(num_blocks, dtype=torch.int32, device=DEVICE).unsqueeze(0)

    # b_start_loc: cumulative token start locations [batch+1]
    b_start_loc = torch.tensor([0, total_tokens], dtype=torch.int32, device=DEVICE)
    b_seq_len = torch.tensor([seq_len], dtype=torch.int32, device=DEVICE)

    k_scale = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
    v_scale = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)

    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            log(os.path.basename(log_path)[:-4]); log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  q/k/v: {tuple(q.shape)} [tokens, heads, dim], device={q.device}")
            log(f"  k_cache: {tuple(k_cache.shape)}, v_cache: {tuple(v_cache.shape)}")
            log(f"  batch={batch}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}")
            log(f"  block_size={block_size}, num_blocks={num_blocks}")

            context_attention_fwd(
                q=q, k=k, v=v, o=o,
                kv_cache_dtype="auto",
                k_cache=k_cache, v_cache=v_cache,
                b_loc=b_loc,
                b_start_loc=b_start_loc,
                b_seq_len=b_seq_len,
                max_seq_len=seq_len,
                max_input_len=seq_len,
                k_scale=k_scale, v_scale=v_scale,
            )
            oc = o.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: ({batch}, {num_heads}, cdiv(seq_len, BLOCK))  (_fwd_kernel path)")
            log("\nOutput:")
            log(f"  o: shape={tuple(oc.shape)}, mean={oc.float().mean().item():.4f}")
            log("\nSummary:")
            log(f"  Kernel execution: SUCCESS")
            log(f"  Output shape matches [{total_tokens},{num_heads},{head_dim}]: "
                f"{list(oc.shape) == [total_tokens, num_heads, head_dim]}")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Kernel execution: FAILED")
            fp.write("\n" + "-" * 36 + "\n"); sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "prefix_prefill")
