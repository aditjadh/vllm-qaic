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
KERNEL_NAME = "_layer_norm_fwd_1pass_kernel"


def main(log_path):
    torch.manual_seed(42)
    with open(log_path, "w") as f:
        log = logger(f)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(ts)
        log(f"Kernel: {KERNEL_NAME}")
        log("")
        try:
            import unittest.mock as _mock
            import vllm.model_executor.layers.mamba.ops.layernorm_gated as _lg_mod
            # _layer_norm_fwd uses torch.cuda.device() context manager which is
            # CUDA-only. Patch it to a no-op for QAIC.
            _cuda_device_patch = _mock.patch(
                "vllm.model_executor.layers.mamba.ops.layernorm_gated.torch.cuda.device",
                new=lambda idx: _mock.MagicMock(__enter__=lambda s, *a: None, __exit__=lambda s, *a: None),
            )
            _cuda_device_patch.start()
            from vllm.model_executor.layers.mamba.ops.layernorm_gated import (
                rms_norm_gated,
            )

            M, N = 16, 256
            x = torch.randn(M, N, dtype=torch.float16).to(device=DEVICE)
            weight = torch.ones(N, dtype=torch.float16).to(device=DEVICE)
            eps = 1e-6

            log("Inputs:")
            log(f"  x: {tuple(x.shape)}, dtype={x.dtype}, device={x.device}")
            log(f"  weight: {tuple(weight.shape)}, dtype={weight.dtype}")
            log(f"  M={M}, N={N}, eps={eps}")
            log(f"  (bias=None, is_rms_norm=True)")
            log("")

            BLOCK_N = min(65536 // x.element_size(), 256)
            log("Grid Configuration:")
            log(f"  grid: ({M}, 1)")
            log(f"  BLOCK_N={BLOCK_N}")
            log("")

            out = rms_norm_gated(x, weight, bias=None, eps=eps)

            try:
                out_sample = out[0, :4].cpu().tolist()
                log("Status: SUCCESS")
                log("")
                log("Output:")
                log(f"  out: {tuple(out.shape)}, dtype={out.dtype}, device={out.device}")
                log(f"  out[0, :4]: {out_sample}")
            except RuntimeError as sync_err:
                log("Status: FAILURE")
                log("")
                log("Error:")
                log(f"  Kernel launched (output shape={tuple(out.shape)}) but device stream sync failed:")
                log(f"  {sync_err}")

        except Exception as e:
            log("Status: FAILURE")
            log("")
            log("Error:")
            log(traceback.format_exc())

        log("")
        log("-" * 36)
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "layer_norm_fwd_1pass_kernel")
