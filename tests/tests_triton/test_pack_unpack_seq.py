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

from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton

# ── Configuration ────────────────────────────────────────────────────────────
# pack_seq packs variable-length sequences [N, D] into [B, Lmax, D] (padded);
# unpack_seq inverts it back to [N, D]. We test both round-trip.
LENGTHS = [5, 3, 8, 1]
D       = 16
DEVICE  = "qaic"

KERNEL_NAME = "_pack_seq_kernel / _unpack_seq_triton_kernel"
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = os.path.join(LOG_DIR, f"test_pack_unpack_seq_{timestamp}.log")


def log(f, msg):
    print(msg)
    f.write(msg + "\n")


def main():
    torch.manual_seed(42)
    N = sum(LENGTHS)
    B = len(LENGTHS)
    Lmax = max(LENGTHS)

    x = torch.randn(N, D, dtype=torch.float32, device=DEVICE)
    lengths = torch.tensor(LENGTHS, dtype=torch.int32, device=DEVICE)
    x_ref = x.clone()

    with open(log_path, "w") as f:
        try:
            log(f, timestamp); log(f, f"Kernel: {KERNEL_NAME}")
            log(f, "\nInputs:")
            log(f, f"  x : shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}")
            log(f, f"  lengths={LENGTHS}, N={N}, B={B}, Lmax={Lmax}, D={D}")

            packed = pack_seq_triton(x, lengths, pad_value=0.0)
            packed_h = packed.cpu()
            log(f, "\nStatus: SUCCESS (pack)")
            log(f, "\nGrid Configuration:")
            log(f, f"  pack grid: (B={B}, cdiv(Lmax,64), cdiv(D,64))")
            log(f, "\nOutput:")
            log(f, f"  packed : shape={tuple(packed.shape)}")

            # validate pack: packed[b, :len_b] == x rows for that segment
            errors = 0
            start = 0
            for b, L in enumerate(LENGTHS):
                seg = x_ref[start:start + L].cpu()
                got = packed_h[b, :L]
                if (got - seg).abs().max().item() > 1e-5:
                    errors += 1
                # padding region must be 0
                if L < Lmax and packed_h[b, L:].abs().max().item() > 1e-5:
                    errors += 1
                start += L
            log(f, "\nValidation (pack):")
            log(f, f"  errors: {errors}")
            assert errors == 0, "pack mismatch"

            # unpack round-trip
            unpacked = unpack_seq_triton(packed, lengths)
            unpacked_h = unpacked.cpu()
            diff = (unpacked_h - x_ref.cpu()).abs().max().item()
            log(f, "\nValidation (unpack round-trip vs original x):")
            log(f, f"  unpacked shape: {tuple(unpacked.shape)}")
            log(f, f"  max abs diff: {diff:.6f}")
            torch.testing.assert_close(unpacked_h, x_ref.cpu(), atol=1e-5, rtol=1e-5)
            log(f, "  assert_close: PASSED")

            log(f, "\nSummary:")
            log(f, "  Kernel execution: SUCCESS (pack + unpack)")
            log(f, "  Validation vs PyTorch reference: PASSED")
            log(f, f"  pack errors: {errors}, unpack round-trip diff: {diff:.6f}")
        except Exception:
            log(f, f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log(f, "\nSummary:\n  Validation vs PyTorch reference: FAILED")
            sys.exit(1)
        finally:
            log(f, "\n" + "-" * 36)
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
