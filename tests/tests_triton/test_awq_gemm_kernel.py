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
KERNEL_NAME = "awq_gemm_kernel"


def main(log_path):
    torch.manual_seed(42)
    with open(log_path, "w") as f:
        log = logger(f)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(ts)
        log(f"Kernel: {KERNEL_NAME}")
        log("")
        try:
            from vllm.model_executor.layers.quantization.awq_triton import (
                awq_gemm_triton,
            )

            # input: [M, K], qweight: [K, N//8], qzeros: [K//G, N//8], scales: [K//G, N]
            M, K, N = 4, 256, 256
            group_size = 128
            split_k_iters = 1

            inp = torch.randn(M, K, dtype=torch.float16, device=DEVICE)
            qweight = torch.randint(0, 2**31, (K, N // 8), dtype=torch.int32, device=DEVICE)
            qzeros = torch.randint(0, 2**31, (K // group_size, N // 8), dtype=torch.int32, device=DEVICE)
            scales = torch.randn(K // group_size, N, dtype=torch.float16, device=DEVICE)

            block_size_m = 32
            block_size_n = 32
            block_size_k = 32

            log("Inputs:")
            log(f"  input: {tuple(inp.shape)}, dtype={inp.dtype}, device={inp.device}")
            log(f"  qweight: {tuple(qweight.shape)}, dtype={qweight.dtype}")
            log(f"  qzeros: {tuple(qzeros.shape)}, dtype={qzeros.dtype}")
            log(f"  scales: {tuple(scales.shape)}, dtype={scales.dtype}")
            log(f"  M={M}, K={K}, N={N}, group_size={group_size}, split_k_iters={split_k_iters}")
            log("")

            num_pid_m = (M + block_size_m - 1) // block_size_m
            num_pid_n = (N + block_size_n - 1) // block_size_n
            log("Grid Configuration:")
            log(f"  grid: ({num_pid_m * num_pid_n}, {split_k_iters})")
            log(f"  BLOCK_SIZE_M={block_size_m}, BLOCK_SIZE_N={block_size_n}, BLOCK_SIZE_K={block_size_k}")
            log(f"  SPLIT_K={split_k_iters}")
            log("")

            result = awq_gemm_triton(
                input=inp,
                qweight=qweight,
                scales=scales,
                qzeros=qzeros,
                split_k_iters=split_k_iters,
                block_size_m=block_size_m,
                block_size_n=block_size_n,
                block_size_k=block_size_k,
            )

            log("Status: SUCCESS")
            log("")
            log("Output:")
            log(f"  result: {tuple(result.shape)}, dtype={result.dtype}, device={result.device}")
            log(f"  result[0, :4]: {result[0, :4].tolist()}")

        except Exception as e:
            log("Status: FAILURE")
            log("")
            log("Error:")
            log(traceback.format_exc())

        log("")
        log("-" * 36)
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "awq_gemm_kernel")
