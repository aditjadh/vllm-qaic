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
KERNEL_NAME = "l2norm_fwd_kernel"


def main(log_path):
    torch.manual_seed(42)
    with open(log_path, "w") as f:
        log = logger(f)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(ts)
        log(f"Kernel: {KERNEL_NAME}")
        log("")
        try:
            import os as _os
            _os.environ["USE_DEFAULT_FLA_NORM"] = "1"

            from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd

            # D <= 512 AND USE_DEFAULT_FLA_NORM=1 -> l2norm_fwd_kernel (BT-tiled)
            T, D = 64, 256
            x = torch.randn(T, D, dtype=torch.float16).to(device=DEVICE)
            eps = 1e-6

            log("Inputs:")
            log(f"  x: {tuple(x.shape)}, dtype={x.dtype}, device={x.device}")
            log(f"  T={T}, D={D}, eps={eps}")
            log(f"  USE_DEFAULT_FLA_NORM=1, D<=512 => l2norm_fwd_kernel")
            log("")

            log("Grid Configuration:")
            log(f"  grid: (cdiv(T, BT),) — BT from autotune")
            log(f"  BD=next_power_of_2({D})=256")
            log("")

            out = l2norm_fwd(x, eps=eps)

            log("Status: SUCCESS")
            log("")
            log("Output:")
            log(f"  out: {tuple(out.shape)}, dtype={out.dtype}, device={out.device}")
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
    run_test(main, LOG_DIR, "l2norm_fwd_kernel")
