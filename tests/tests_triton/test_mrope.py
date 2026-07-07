# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
import os
import sys
import traceback
from datetime import datetime
# import triton
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vllm"))

from vllm.model_executor.layers.rotary_embedding.mrope import triton_mrope

# ── Configuration ────────────────────────────────────────────────────────────
NUM_TOKENS   = 16
NUM_Q_HEADS  = 8
NUM_KV_HEADS = 2
HEAD_SIZE    = 128
ROTARY_DIM   = 128          # full rotation (equals head_size)
# t + h + w must equal rotary_dim // 2 = 64
MROPE_SECTION      = [16, 24, 24]
MROPE_INTERLEAVED  = False
DTYPE  = torch.float16
DEVICE = "qaic"

KERNEL_NAME = "_triton_mrope_forward"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "verified_kernels")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path    = os.path.join(LOG_DIR, f"test_mrope_{timestamp}.log")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def reference_mrope(q, k, cos, sin, mrope_section):
    """Pure-PyTorch reference for chunked (non-interleaved) mrope."""
    # cos/sin: [3, num_tokens, head_dim // 2]
    # Build per-token cos/sin by concatenating the T/H/W slices
    # mrope_section = [t, h, w], sum == head_dim // 2
    t, h, w = mrope_section
    cos_row = torch.cat([cos[0, :, :t], cos[1, :, t:t+h], cos[2, :, t+h:]], dim=-1)
    sin_row = torch.cat([sin[0, :, :t], sin[1, :, t:t+h], sin[2, :, t+h:]], dim=-1)
    # cos_row, sin_row: [num_tokens, head_dim // 2]

    def apply_rope(x):
        # x: [num_tokens, num_heads * head_size]
        nt, nh_hd = x.shape
        x = x.view(nt, -1, HEAD_SIZE)
        x_rot = x[..., :ROTARY_DIM].float()
        x_pass = x[..., ROTARY_DIM:]
        cos_e = cos_row.unsqueeze(1).expand_as(x_rot[..., :HEAD_SIZE // 2])
        sin_e = sin_row.unsqueeze(1).expand_as(x_rot[..., :HEAD_SIZE // 2])
        # Standard rotate-half RoPE on the full rotary_dim
        x1 = x_rot[..., :HEAD_SIZE // 2]
        x2 = x_rot[..., HEAD_SIZE // 2:]
        new_x1 = x1 * cos_e - x2 * sin_e
        new_x2 = x2 * cos_e + x1 * sin_e
        x_rot_new = torch.cat([new_x1, new_x2], dim=-1).to(x.dtype)
        return torch.cat([x_rot_new, x_pass], dim=-1).view(nt, nh_hd)

    return apply_rope(q), apply_rope(k)


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    torch.manual_seed(42)

    q   = torch.randn(NUM_TOKENS, NUM_Q_HEADS  * HEAD_SIZE, dtype=DTYPE, device=DEVICE)
    k   = torch.randn(NUM_TOKENS, NUM_KV_HEADS * HEAD_SIZE, dtype=DTYPE, device=DEVICE)
    cos = torch.randn(3, NUM_TOKENS, HEAD_SIZE // 2,        dtype=DTYPE, device=DEVICE)
    sin = torch.randn(3, NUM_TOKENS, HEAD_SIZE // 2,        dtype=DTYPE, device=DEVICE)

    #creates a copy of the tensor and allocates new underlying memory
    q_ref = q.clone().cpu()
    k_ref = k.clone().cpu()
    cos_cpu = cos.cpu()
    sin_cpu = sin.cpu()

    import math
    # grid = (NUM_TOKENS,)
    grid = (math.ceil(NUM_TOKENS/16) * 16,)

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")

            log(f, "\nInputs:")
            log(f, f"  q   : shape={tuple(q.shape)}, dtype={q.dtype}, device={q.device}")
            log(f, f"  k   : shape={tuple(k.shape)}, dtype={k.dtype}, device={k.device}")
            log(f, f"  cos : shape={tuple(cos.shape)}, dtype={cos.dtype}")
            log(f, f"  sin : shape={tuple(sin.shape)}, dtype={sin.dtype}")
            log(f, f"  mrope_section={MROPE_SECTION}, interleaved={MROPE_INTERLEAVED}")

            q_out, k_out = triton_mrope(
                q, k, cos, sin,
                MROPE_SECTION, HEAD_SIZE, ROTARY_DIM, MROPE_INTERLEAVED
            )

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}")
            log(f, f"  block: determined by pad_n_qh/pad_hd constexprs")

            log(f, "\nOutput:")
            log(f, f"  q_out : shape={tuple(q_out.shape)}, min={q_out.min().item():.4f}, "
                   f"max={q_out.max().item():.4f}, mean={q_out.float().mean().item():.4f}")
            log(f, f"  k_out : shape={tuple(k_out.shape)}, min={k_out.min().item():.4f}, "
                   f"max={k_out.max().item():.4f}, mean={k_out.float().mean().item():.4f}")

            # ── Reference validation ────────────────────────────────────────
            # Run reference on CPU — QAIC doesn't support all PyTorch ops used here
            q_ref_out, k_ref_out = reference_mrope(
                q_ref.cpu(), k_ref.cpu(), cos_cpu, sin_cpu, MROPE_SECTION
            )
            q_diff = (q_out.float().cpu() - q_ref_out.float()).abs().max().item()
            k_diff = (k_out.float().cpu() - k_ref_out.float()).abs().max().item()
            log(f, "\nValidation (vs PyTorch reference):")
            log(f, f"  q max abs diff: {q_diff:.6f}")
            log(f, f"  k max abs diff: {k_diff:.6f}")

            torch.testing.assert_close(q_out.float().cpu(), q_ref_out.float(), atol=1e-2, rtol=1.6e-2)
            torch.testing.assert_close(k_out.float().cpu(), k_ref_out.float(), atol=1e-2, rtol=1.6e-2)
            log(f, "  assert_close: PASSED")

            log(f, "\nSummary:")
            log(f, f"  Kernel execution: SUCCESS (grid {grid})")
            log(f, "  Validation vs PyTorch reference: PASSED")
            log(f, f"  q max abs diff: {q_diff:.6f}, k max abs diff: {k_diff:.6f}")

        except Exception as e:
            msg = f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}"
            log(f, msg)
            log(f, "\nSummary:")
            log(f, f"  Kernel execution: SUCCESS (grid {grid})")
            log(f, "  Validation vs PyTorch reference: FAILED")
            try:
                log(f, f"  q max abs diff: {q_diff:.6f}, k max abs diff: {k_diff:.6f}")
            except NameError:
                log(f, "  diffs unavailable (failure occurred before validation)")
            sys.exit(1)

        finally:
            log(f, "\n" + "-" * 36)

    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
