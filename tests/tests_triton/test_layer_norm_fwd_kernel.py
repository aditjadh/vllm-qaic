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
KERNEL_NAME = "layer_norm_fwd_kernel"


def main(log_path):
    torch.manual_seed(42)
    with open(log_path, "w") as f:
        log = logger(f)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(ts)
        log(f"Kernel: {KERNEL_NAME}")
        log("")
        try:
            import vllm.model_executor.layers.fla.ops.layernorm_guard as _ln_mod
            # calc_rows_per_block calls torch.cuda.get_device_properties which is
            # CUDA-only. Patch it to return a fixed value for QAIC.
            _ln_mod.calc_rows_per_block = lambda M, device: 1
            from vllm.model_executor.layers.fla.ops.layernorm_guard import layer_norm_fwd

            M, N = 16, 256
            x = torch.randn(M, N, dtype=torch.float16).to(device=DEVICE)
            weight = torch.ones(N, dtype=torch.float16).to(device=DEVICE)
            bias = torch.zeros(N, dtype=torch.float16).to(device=DEVICE)
            eps = 1e-6

            log("Inputs:")
            log(f"  x: {tuple(x.shape)}, dtype={x.dtype}, device={x.device}")
            log(f"  weight: {tuple(weight.shape)}, dtype={weight.dtype}")
            log(f"  bias: {tuple(bias.shape)}, dtype={bias.dtype}")
            log(f"  M={M}, N={N}, eps={eps}, is_rms_norm=False")
            log("")

            BLOCK_N = min(65536 // x.element_size(), 256)
            log("Grid Configuration:")
            log(f"  grid: (cdiv(M, rows_per_block), 1)")
            log(f"  BLOCK_N={BLOCK_N}")
            log("")

            out, mean, rstd = layer_norm_fwd(x, weight, bias, eps, is_rms_norm=False)

            log("Status: SUCCESS")
            log("")
            log("Output:")
            log(f"  out: {tuple(out.shape)}, dtype={out.dtype}, device={out.device}")
            log(f"  mean: {tuple(mean.shape)}, dtype={mean.dtype}")
            log(f"  rstd: {tuple(rstd.shape)}, dtype={rstd.dtype}")
            log(f"  out[0, :4]: {out[0, :4].cpu().tolist()}")

        except Exception as e:
            log("Status: FAILURE")
            log("")
            log("Error:")
            log(traceback.format_exc())

        log("")
        log("-" * 36)
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "layer_norm_fwd_kernel")
