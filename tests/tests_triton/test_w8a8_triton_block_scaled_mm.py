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

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "_w8a8_triton_block_scaled_mm"


def main(log_path):
    torch.manual_seed(42)
    with open(log_path, "w") as f:
        log = logger(f)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(ts)
        log(f"Kernel: {KERNEL_NAME}")
        log("")
        try:
            from vllm.model_executor.layers.quantization.utils.fp8_utils import (
                w8a8_triton_block_scaled_mm,
            )

            # A: [M, K] fp8, B: [N, K] fp8 (transposed in kernel)
            # As: [M, K//block_k] fp32, Bs: [N//block_n, K//block_k] fp32
            M, N, K = 16, 256, 256
            block_size = [128, 128]
            block_n, block_k = block_size

            import triton

            fp8_dtype = torch.float8_e4m3fn
            A = torch.randn(M, K, dtype=torch.float16).to(fp8_dtype).to(device=DEVICE)
            B = torch.randn(N, K, dtype=torch.float16).to(fp8_dtype).to(device=DEVICE)
            As = torch.ones(M, K // block_k, dtype=torch.float32, device=DEVICE)
            Bs = torch.ones(triton.cdiv(N, block_n), triton.cdiv(K, block_k), dtype=torch.float32, device=DEVICE)

            log("Inputs:")
            log(f"  A: {tuple(A.shape)}, dtype={A.dtype}, device={A.device}")
            log(f"  B: {tuple(B.shape)}, dtype={B.dtype}")
            log(f"  As: {tuple(As.shape)}, dtype={As.dtype}")
            log(f"  Bs: {tuple(Bs.shape)}, dtype={Bs.dtype}")
            log(f"  M={M}, N={N}, K={K}, block_size={block_size}")
            log("")

            BLOCK_SIZE_M = 64
            num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
            num_pid_n = (N + block_n - 1) // block_n
            log("Grid Configuration:")
            log(f"  grid: ({num_pid_m * num_pid_n},)")
            log(f"  BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_N={block_n}, BLOCK_SIZE_K={block_k}")
            log("")

            C = w8a8_triton_block_scaled_mm(
                A=A,
                B=B,
                As=As,
                Bs=Bs,
                block_size=block_size,
                output_dtype=torch.bfloat16,
            )

            log("Status: SUCCESS")
            log("")
            log("Output:")
            log(f"  C: {tuple(C.shape)}, dtype={C.dtype}, device={C.device}")
            log(f"  C[0, :4]: {C[0, :4].tolist()}")

        except Exception as e:
            log("Status: FAILURE")
            log("")
            log("Error:")
            log(traceback.format_exc())

        log("")
        log("-" * 36)
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "w8a8_triton_block_scaled_mm")
