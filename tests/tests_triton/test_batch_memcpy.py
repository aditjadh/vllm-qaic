# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
import os
import sys
import subprocess
import traceback
from datetime import datetime

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vllm"))

from vllm.v1.worker.mamba_utils import batch_memcpy

# ── Configuration ────────────────────────────────────────────────────────────
# batch_memcpy copies `sizes[i]` bytes from src_ptrs[i] to dst_ptrs[i]. The
# kernel casts integer addresses to uint8 pointers (tt.int_to_ptr).
DEVICE = "qaic"

KERNEL_NAME = "batch_memcpy_kernel"
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
if "BMEMCPY_LOG_PATH" in os.environ:
    log_path  = os.environ["BMEMCPY_LOG_PATH"]
    timestamp = os.path.basename(log_path)[len("test_batch_memcpy_"):-len(".log")]
else:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path  = os.path.join(LOG_DIR, f"test_batch_memcpy_{timestamp}.log")


def log(f, msg):
    print(msg); f.write(msg + "\n"); f.flush()


def main():
    torch.manual_seed(42)
    # 3 source / dest int32 buffers.
    srcs = [torch.arange(i * 10, i * 10 + 8, dtype=torch.int32, device=DEVICE) for i in range(3)]
    dsts = [torch.zeros(8, dtype=torch.int32, device=DEVICE) for _ in range(3)]
    # sizes in BYTES (kernel copies uint8); 8 int32 = 32 bytes.
    sizes = torch.tensor([32, 32, 32], dtype=torch.int32, device=DEVICE)
    src_ptrs = torch.tensor([s.data_ptr() for s in srcs], dtype=torch.int64, device=DEVICE)
    dst_ptrs = torch.tensor([d.data_ptr() for d in dsts], dtype=torch.int64, device=DEVICE)

    srcs_ref = [s.clone() for s in srcs]
    with open(log_path, "w") as f:
        try:
            log(f, timestamp); log(f, f"Kernel: {KERNEL_NAME}")
            log(f, "\nInputs:")
            log(f, f"  batch=3, each 8 int32 (32 bytes), device={DEVICE}")
            log(f, f"  sizes(bytes)={sizes.cpu().tolist()}")

            batch_memcpy(src_ptrs, dst_ptrs, sizes)
            outs = [d.cpu() for d in dsts]

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, "  grid: (batch,) = (3,), BLOCK_SIZE: 1024")
            log(f, "\nOutput:")
            for i, o in enumerate(outs):
                log(f, f"  dst[{i}]: {o.tolist()}")

            errors = sum(int((outs[i] != srcs_ref[i].cpu()).any()) for i in range(3))
            log(f, "\nValidation (each dst == src):")
            log(f, f"  mismatched buffers: {errors}")
            for i in range(3):
                torch.testing.assert_close(outs[i], srcs_ref[i].cpu(), atol=0, rtol=0)
            log(f, "  assert_close: PASSED")
            log(f, "\nSummary:")
            log(f, "  Kernel execution: SUCCESS (grid (3,))")
            log(f, "  Validation vs PyTorch reference: PASSED")
            log(f, f"  mismatched buffers: {errors}")
        except Exception:
            log(f, f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log(f, "\nSummary:")
            log(f, "  Kernel execution: FAILED")
            log(f, "  Validation vs PyTorch reference: NOT REACHED")
            log(f, "  Note: batch_memcpy_kernel casts int addresses to uint8 pointers")
            log(f, "        (tt.int_to_ptr); the QAIC backend may not lower this.")
            sys.exit(1)
        finally:
            log(f, "\n" + "-" * 36)
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    if os.environ.get("BMEMCPY_CHILD") == "1":
        main(); sys.exit(0)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(LOG_DIR, f"test_batch_memcpy_{timestamp}.log")
    env = dict(os.environ, BMEMCPY_CHILD="1", BMEMCPY_LOG_PATH=log_path)
    proc = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True)
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="")
    if proc.returncode < 0 and not (proc.stdout and "Status: SUCCESS" in proc.stdout):
        with open(log_path, "a") as f:
            f.write(f"\nStatus: FAILURE\n\nError:\nChild killed by signal "
                    f"(exit {proc.returncode}; -6/134 => SIGABRT) during compile.\n")
            tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
            f.write(tail + "\n")
            f.write("\nSummary:\n  Kernel execution: FAILED (compiler abort)\n")
            f.write("  Validation vs PyTorch reference: NOT REACHED\n")
            f.write("\n" + "-" * 36 + "\n")
        print(f"\nLog saved to: {log_path}")
    sys.exit(proc.returncode)
