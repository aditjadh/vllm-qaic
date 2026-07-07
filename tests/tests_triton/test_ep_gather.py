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

from vllm.model_executor.layers.fused_moe.deep_gemm_utils import ep_gather

# ── Configuration ────────────────────────────────────────────────────────────
# ep_gather is the inverse of scatter: for each output token it gathers the rows
# from input_tensor at input_index[token, k], weights by recv_topk_weight, and
# sums over top_k into output_tensor.
NUM_TOKENS  = 16
TOP_K       = 2
NUM_SRC     = 64            # rows in the (permuted) input tensor
HIDDEN      = 256
DEVICE      = "qaic"

KERNEL_NAME = "_fwd_kernel_ep_gather (via ep_gather)"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_ep_gather_{timestamp}.log")


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    torch.manual_seed(42)

    # QAIC device-side bfloat16 ops (randn/zeros/copy) are unsupported, so build
    # all bf16 tensors on CPU and move them to the device.
    input_tensor = torch.randn(NUM_SRC, HIDDEN, dtype=torch.float32).to(torch.bfloat16).to(DEVICE)
    # each (token, slot) gathers a source row in [0, NUM_SRC)
    input_index = torch.randint(0, NUM_SRC, (NUM_TOKENS, TOP_K),
                                dtype=torch.int32, device=DEVICE)
    recv_topk_ids = torch.zeros(NUM_TOKENS, TOP_K, dtype=torch.int32, device=DEVICE)
    recv_topk_weight = torch.rand(NUM_TOKENS, TOP_K, dtype=torch.float32, device=DEVICE)
    output_tensor = torch.zeros(NUM_TOKENS, HIDDEN, dtype=torch.bfloat16).to(DEVICE)

    input_ref = input_tensor.clone()
    index_ref = input_index.clone()
    weight_ref = recv_topk_weight.clone()

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")

            log(f, "\nInputs:")
            log(f, f"  input_tensor     : shape={tuple(input_tensor.shape)}, dtype={input_tensor.dtype}, device={input_tensor.device}")
            log(f, f"  input_index      : shape={tuple(input_index.shape)}, dtype={input_index.dtype}")
            log(f, f"  recv_topk_weight : shape={tuple(recv_topk_weight.shape)}")
            log(f, f"  num_tokens={NUM_TOKENS}, top_k={TOP_K}, num_src={NUM_SRC}, hidden={HIDDEN}")

            ep_gather(
                input_tensor=input_tensor,
                recv_topk_ids=recv_topk_ids,
                recv_topk_weight=recv_topk_weight,
                input_index=input_index,
                expert_map=None,
                output_tensor=output_tensor,
            )

            out_host = output_tensor.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: (cdiv(hidden, BLOCK_D), min(num_tokens, 1024))")
            log(f, f"  BLOCK_D=min(hidden, 1024)={min(HIDDEN, 1024)}")

            log(f, "\nOutput:")
            log(f, f"  output_tensor: shape={tuple(output_tensor.shape)}, "
                   f"abs mean={out_host.float().abs().mean().item():.4f}")

            # ── Reference: out[t] = sum_k weight[t,k] * input[index[t,k]] ────
            ref = torch.zeros(NUM_TOKENS, HIDDEN, dtype=torch.float32)
            for t in range(NUM_TOKENS):
                acc = torch.zeros(HIDDEN, dtype=torch.float32)
                for k in range(TOP_K):
                    src = int(index_ref[t, k])
                    acc += input_ref[src].cpu().float() * float(weight_ref[t, k])
                ref[t] = acc
            diff = (out_host.float() - ref).abs().max().item()
            log(f, "\nValidation (vs PyTorch weighted gather-sum):")
            log(f, f"  max abs diff: {diff:.6f}")

            torch.testing.assert_close(out_host.float(), ref, atol=1e-1, rtol=1e-2)
            log(f, "  assert_close: PASSED")

            log(f, "\nSummary:")
            log(f, "  Kernel execution: SUCCESS")
            log(f, "  Validation vs PyTorch reference: PASSED")
            log(f, f"  max abs diff: {diff:.6f}")

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
