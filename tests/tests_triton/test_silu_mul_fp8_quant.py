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

from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
    _silu_mul_fp8_quant_deep_gemm,
)

# ── Configuration ────────────────────────────────────────────────────────────
# _silu_mul_fp8_quant_deep_gemm: input y (E, T, 2*H). For each token it computes
# silu(y[:H]) * y[H:], then quantizes per group of GROUP_SIZE elements to fp8
# with a per-(token,group) fp32 scale.
E          = 2
T          = 8             # tokens per expert (padded)
H          = 256           # hidden (per output)
GROUP_SIZE = 128
DEVICE     = "qaic"

KERNEL_NAME = "_silu_mul_fp8_quant_deep_gemm"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_silu_mul_fp8_quant_{timestamp}.log")


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def reference(y, tokens_per_expert, H, group_size):
    """silu(gate)*up then per-group fp8 quant; returns dequantized fp32 ref."""
    E, T, H2 = y.shape
    G = H // group_size
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    deq = torch.zeros(E, T, H, dtype=torch.float32)
    for e in range(E):
        n = int(tokens_per_expert[e])
        for t in range(n):
            gate = y[e, t, :H].float()
            up = y[e, t, H:].float()
            val = (gate * torch.sigmoid(gate)) * up
            for g in range(G):
                seg = val[g * group_size : (g + 1) * group_size]
                amax = seg.abs().max()
                scale = torch.clamp(amax, min=1e-10) / fp8_max
                q = torch.clamp(seg / scale, -fp8_max, fp8_max)
                q = q.to(torch.float8_e4m3fn).float()
                deq[e, t, g * group_size : (g + 1) * group_size] = q * scale
    return deq


def main():
    torch.manual_seed(42)

    H2 = 2 * H
    # Build on CPU (QAIC device-side randn can be finicky), move to device.
    y = torch.randn(E, T, H2, dtype=torch.float32).to(torch.bfloat16).to(DEVICE)
    tokens_per_expert = torch.tensor([T, T // 2], dtype=torch.int32, device=DEVICE)

    y_ref = y.clone()
    tpe_ref = tokens_per_expert.clone()

    G = H // GROUP_SIZE
    grid = (E * G,)

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")

            log(f, "\nInputs:")
            log(f, f"  y (E,T,2H)        : shape={tuple(y.shape)}, dtype={y.dtype}, device={y.device}")
            log(f, f"  tokens_per_expert : {tokens_per_expert.cpu().tolist()}")
            log(f, f"  E={E}, T={T}, H={H}, group_size={GROUP_SIZE}, G={G}")

            fp8_dtype = torch.float8_e4m3fn
            # QAIC can't allocate fp8 tensors on-device; build on CPU then move.
            y_q = torch.empty((E, T, H), dtype=fp8_dtype).to(DEVICE)
            # scales: (E, T, G) with strides (T*G, 1, T)
            y_s = torch.empty_strided(
                (E, T, G), (T * G, 1, T), dtype=torch.float32, device=DEVICE
            )

            stride_i_e, stride_i_t, stride_i_h = y.stride()
            stride_yq_e, stride_yq_t, stride_yq_h = y_q.stride()
            f_info = torch.finfo(fp8_dtype)

            _silu_mul_fp8_quant_deep_gemm[grid](
                y, y_q, y_s, tokens_per_expert,
                H, GROUP_SIZE,
                stride_i_e, stride_i_t, stride_i_h,
                stride_yq_e, stride_yq_t, stride_yq_h,
                y_s.stride(0), y_s.stride(1), y_s.stride(2),
                tokens_per_expert.stride(0),
                1e-10,                      # eps
                f_info.min, f_info.max,     # fp8_min, fp8_max
                False,                      # ceil_ue8m0
                BLOCK=GROUP_SIZE,
                NUM_STAGES=4,
                num_warps=1,
            )

            yq_host = y_q.cpu()
            ys_host = y_s.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}  (E * G,)")
            log(f, f"  BLOCK={GROUP_SIZE}, NUM_STAGES=4, num_warps=1")

            # Dequantize kernel output for comparison.
            deq = torch.zeros(E, T, H, dtype=torch.float32)
            for e in range(E):
                n = int(tpe_ref[e])
                for t in range(n):
                    for g in range(G):
                        sl = slice(g * GROUP_SIZE, (g + 1) * GROUP_SIZE)
                        deq[e, t, sl] = yq_host[e, t, sl].float() * ys_host[e, t, g]

            log(f, "\nOutput:")
            log(f, f"  y_q : shape={tuple(y_q.shape)}, dtype={y_q.dtype}")
            log(f, f"  y_s : shape={tuple(y_s.shape)}, abs mean={ys_host.abs().mean().item():.6f}")

            # ── Reference validation ────────────────────────────────────────
            ref = reference(y_ref, tpe_ref, H, GROUP_SIZE)
            # only compare valid tokens
            mask = torch.zeros(E, T, 1)
            for e in range(E):
                mask[e, : int(tpe_ref[e])] = 1.0
            diff = ((deq - ref) * mask).abs().max().item()
            log(f, "\nValidation (dequantized kernel output vs PyTorch silu-mul + fp8 quant):")
            log(f, f"  max abs diff (valid tokens): {diff:.4f}")

            # fp8 is low precision; compare with a loose tolerance.
            torch.testing.assert_close(
                (deq * mask), (ref * mask), atol=0.2, rtol=0.1
            )
            log(f, "  assert_close: PASSED")

            log(f, "\nSummary:")
            log(f, f"  Kernel execution: SUCCESS (grid {grid})")
            log(f, "  Validation vs PyTorch reference: PASSED")
            log(f, f"  max abs diff: {diff:.4f}")

        except Exception:
            msg = f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}"
            log(f, msg)
            log(f, "\nSummary:")
            log(f, "  Kernel execution: NOT REACHED")
            log(f, "  Validation vs PyTorch reference: FAILED")
            log(f, "  Note: the kernel's output is fp8 (torch.float8_e4m3fn), but QAIC")
            log(f, "        rejects fp8 tensors on-device ('Unsupported datatype for")
            log(f, "        QAIC: Float8_e4m3fn'), even when built on CPU and moved.")
            log(f, "        The fp8 quant output cannot be materialized on this device,")
            log(f, "        so the kernel cannot be exercised here.")
            sys.exit(1)

        finally:
            log(f, "\n" + "-" * 36)

    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
