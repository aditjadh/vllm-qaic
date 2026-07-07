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

from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "_fwd_kernel_stage1 / _fwd_grouped_kernel_stage1 / _fwd_kernel_stage2"


def main(log_path):
    torch.manual_seed(42)
    # MHA decode: 2 seqs, 4 heads, head_dim=64, seq_lens=[32, 48]
    num_seqs = 2
    num_heads = 4
    head_dim = 64
    page_size = 1
    num_kv_splits = 4
    seq_lens_list = [32, 48]
    max_seq_len = max(seq_lens_list)
    total_tokens = sum(seq_lens_list)

    # q: [num_seqs, num_heads, head_dim]
    q = torch.randn(num_seqs, num_heads, head_dim, dtype=torch.float16, device=DEVICE)
    # k_buffer/v_buffer: [total_tokens, num_heads, head_dim]  (flat kv-token pool)
    k_buffer = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.float16, device=DEVICE)
    v_buffer = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.float16, device=DEVICE)

    o = torch.zeros(num_seqs, num_heads, head_dim, dtype=torch.float16, device=DEVICE)
    lse = torch.zeros(num_seqs, num_heads, dtype=torch.float32, device=DEVICE)

    # req_to_token: [num_seqs, max_seq_len] — logical token indices per sequence
    req_to_token = torch.zeros(num_seqs, max_seq_len, dtype=torch.int32, device=DEVICE)
    offset = 0
    for i, sl in enumerate(seq_lens_list):
        req_to_token[i, :sl] = torch.arange(offset, offset + sl, dtype=torch.int32)
        offset += sl

    b_seq_len = torch.tensor(seq_lens_list, dtype=torch.int32, device=DEVICE)
    # attn_logits: [num_seqs, num_heads, num_kv_splits, head_dim+1]
    attn_logits = torch.zeros(num_seqs, num_heads, num_kv_splits, head_dim + 1,
                               dtype=torch.float32, device=DEVICE)
    sm_scale = 1.0 / (head_dim ** 0.5)

    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            log(os.path.basename(log_path)[:-4]); log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  q: {tuple(q.shape)} [num_seqs, num_heads, head_dim], device={q.device}")
            log(f"  k_buffer/v_buffer: {tuple(k_buffer.shape)} [total_tokens, num_heads, head_dim]")
            log(f"  num_seqs={num_seqs}, num_heads={num_heads}, head_dim={head_dim}")
            log(f"  seq_lens={seq_lens_list}, num_kv_splits={num_kv_splits}")
            log(f"  sm_scale={sm_scale:.4f}, page_size={page_size} (MHA path)")

            decode_attention_fwd(
                q=q,
                k_buffer=k_buffer,
                v_buffer=v_buffer,
                o=o,
                lse=lse,
                req_to_token=req_to_token,
                b_seq_len=b_seq_len,
                attn_logits=attn_logits,
                num_kv_splits=num_kv_splits,
                sm_scale=sm_scale,
                page_size=page_size,
            )
            oc = o.cpu()
            lse_c = lse.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  stage1: grid=(num_seqs={num_seqs}, num_heads={num_heads}, num_kv_splits={num_kv_splits})")
            log(f"  stage2: grid=(num_seqs={num_seqs}, num_heads={num_heads})")
            log("\nOutput:")
            log(f"  o: shape={tuple(oc.shape)}, mean={oc.float().mean().item():.4f}")
            log(f"  lse: shape={tuple(lse_c.shape)}, mean={lse_c.mean().item():.4f}")
            log("\nSummary:")
            log(f"  Kernel execution: SUCCESS")
            log(f"  Output shape: {list(oc.shape)}")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Kernel execution: FAILED")
            fp.write("\n" + "-" * 36 + "\n"); sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "decode_attention_stage1_stage2")
