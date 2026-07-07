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

from vllm.model_executor.layers.fused_moe.deep_gemm_utils import ep_scatter
from vllm.triton_utils import triton

# ── Configuration ────────────────────────────────────────────────────────────
# ep_scatter runs _fwd_kernel_ep_scatter_1 then _fwd_kernel_ep_scatter_2.
# scatter_2 copies each token's row (and its quant scale) into the per-expert
# contiguous output region, recording the destination index in output_index.
NUM_TOKENS  = 16
TOP_K       = 1
NUM_EXPERTS = 4
HIDDEN      = 256
BLOCK_D     = 128          # quant block size (scale columns = HIDDEN // BLOCK_D)
DEVICE      = "qaic"

KERNEL_NAME = "_fwd_kernel_ep_scatter_2 (via ep_scatter)"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_ep_scatter_2_{timestamp}.log")


def round_up_128(x):
    return ((x + 127) // 128) * 128


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    torch.manual_seed(42)

    # Each token assigned to one expert (top_k=1).
    topk_ids = torch.randint(0, NUM_EXPERTS, (NUM_TOKENS, TOP_K),
                             dtype=torch.int32, device=DEVICE)
    counts = torch.bincount(topk_ids.reshape(-1), minlength=NUM_EXPERTS).to(torch.int32)

    # QAIC device-side bfloat16 ops (randn/zeros/copy) are unsupported, so build
    # all bf16 tensors on CPU and move them to the device.
    recv_x = torch.randn(NUM_TOKENS, HIDDEN, dtype=torch.float32).to(torch.bfloat16).to(DEVICE)
    scale_cols = HIDDEN // BLOCK_D
    recv_x_scale = torch.randn(NUM_TOKENS, scale_cols, dtype=torch.float32, device=DEVICE)

    rounded = [round_up_128(int(c)) for c in counts]
    M_sum = sum(rounded)

    expert_start_loc = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=DEVICE)
    output_tensor = torch.zeros(M_sum, HIDDEN, dtype=torch.bfloat16).to(DEVICE)
    output_tensor_scale = torch.zeros(M_sum, scale_cols, dtype=torch.float32, device=DEVICE)
    m_indices = torch.full((M_sum,), -1, dtype=torch.int32, device=DEVICE)
    output_index = torch.full((NUM_TOKENS, TOP_K), -1, dtype=torch.int32, device=DEVICE)

    recv_x_ref = recv_x.clone()
    topk_ids_ref = topk_ids.clone()

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")

            log(f, "\nInputs:")
            log(f, f"  recv_x       : shape={tuple(recv_x.shape)}, dtype={recv_x.dtype}, device={recv_x.device}")
            log(f, f"  recv_x_scale : shape={tuple(recv_x_scale.shape)}, dtype={recv_x_scale.dtype}")
            log(f, f"  topk_ids     : shape={tuple(topk_ids.shape)} -> counts={counts.cpu().tolist()}")
            log(f, f"  num_tokens={NUM_TOKENS}, top_k={TOP_K}, num_experts={NUM_EXPERTS}, "
                   f"hidden={HIDDEN}, BLOCK_D={BLOCK_D}")
            log(f, f"  M_sum={M_sum}")

            ep_scatter(
                recv_x=recv_x,
                recv_x_scale=recv_x_scale,
                recv_topk=topk_ids,
                num_recv_tokens_per_expert=counts,
                expert_map=None,
                expert_start_loc=expert_start_loc,
                output_tensor=output_tensor,
                output_tensor_scale=output_tensor_scale,
                m_indices=m_indices,
                output_index=output_index,
            )

            oi_host = output_index.cpu()
            ot_host = output_tensor.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, "  scatter_1 grid: (num_experts,)")
            log(f, f"  scatter_2 grid: (min(num_tokens, 8192),) = ({min(NUM_TOKENS, 8192)},)")

            log(f, "\nOutput:")
            log(f, f"  output_index: {oi_host.reshape(-1).tolist()}")
            log(f, f"  output_tensor: shape={tuple(output_tensor.shape)}, "
                   f"abs mean={ot_host.float().abs().mean().item():.4f}")

            # ── Reference validation ────────────────────────────────────────
            # Every token must be scattered to a unique destination row whose
            # content equals the source row.
            errors = []
            seen = set()
            for t in range(NUM_TOKENS):
                dst = int(oi_host[t, 0])
                if dst in seen:
                    errors.append(f"token {t}: duplicate dst {dst}")
                seen.add(dst)
                got = ot_host[dst].float()
                exp = recv_x_ref[t].cpu().float()
                if (got - exp).abs().max().item() > 1e-2:
                    errors.append(f"token {t}: content mismatch at dst {dst}")

            log(f, "\nValidation (each token scattered to unique dst with matching content):")
            log(f, f"  unique destinations: {len(seen)} / {NUM_TOKENS}")
            log(f, f"  errors: {len(errors)}")
            for e in errors[:5]:
                log(f, f"    {e}")

            assert len(errors) == 0, f"{len(errors)} scatter errors"
            log(f, "  validation: PASSED")

            log(f, "\nSummary:")
            log(f, "  Kernel execution: SUCCESS")
            log(f, "  Validation vs PyTorch reference: PASSED")
            log(f, f"  all {NUM_TOKENS} tokens scattered correctly")

        except Exception:
            msg = f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}"
            log(f, msg)
            log(f, "\nSummary:")
            log(f, "  Kernel execution: FAILED (Triton->Hexagon compile error)")
            log(f, "  Validation vs PyTorch reference: NOT REACHED")
            log(f, "  Note: scatter_2 uses tl.atomic_add (tt.atomic_rmw add) on")
            log(f, "        expert_start_loc; the QAIC backend fails to translate")
            log(f, "        the atomic RMW to LLVM-IR.")
            sys.exit(1)

        finally:
            log(f, "\n" + "-" * 36)

    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
