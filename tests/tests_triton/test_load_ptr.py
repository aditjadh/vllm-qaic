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

from vllm.v1.worker.gpu.block_table import _load_ptr
from vllm.triton_utils import tl, triton

# ── Configuration ────────────────────────────────────────────────────────────
# _load_ptr loads an integer address from a pointer-table, casts it to a typed
# pointer (tl.cast -> tt.int_to_ptr), and applies a 16-alignment hint. We wrap it
# to dereference one int32 tensor selected from a table of two.
N      = 16
DEVICE = "qaic"

KERNEL_NAME = "_load_ptr"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
if "LOADPTR_LOG_PATH" in os.environ:
    log_path  = os.environ["LOADPTR_LOG_PATH"]
    timestamp = os.path.basename(log_path)[len("test_load_ptr_"):-len(".log")]
else:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path  = os.path.join(LOG_DIR, f"test_load_ptr_{timestamp}.log")


@triton.jit
def load_ptr_wrapper(table_ptr, out_ptr, group_id, n, BLOCK_SIZE: tl.constexpr):
    # Mirror block_table.py usage: data_ptr = _load_ptr(table + group_id, int32)
    data_ptr = _load_ptr(table_ptr + group_id, tl.int32)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    vals = tl.load(data_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, vals, mask=mask)


def log(f, msg):
    print(msg)
    f.write(msg + "\n")
    f.flush()


def main():
    torch.manual_seed(42)

    # Two candidate int32 buffers; the table holds their addresses.
    buf0 = torch.arange(0, N, dtype=torch.int32, device=DEVICE)
    buf1 = torch.arange(100, 100 + N, dtype=torch.int32, device=DEVICE)
    # int64 pointer table (QAIC rejects uint64).
    table = torch.tensor([buf0.data_ptr(), buf1.data_ptr()],
                         dtype=torch.int64, device=DEVICE)
    out = torch.empty(N, dtype=torch.int32, device=DEVICE)

    GROUP_ID = 1   # select buf1
    buf1_ref = buf1.clone()

    grid = (1,)
    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")
            log(f, "\nInputs:")
            log(f, f"  pointer table : 2 int32 buffers (addresses), dtype={table.dtype}, device={table.device}")
            log(f, f"  group_id={GROUP_ID} (select buf1 = arange(100, 100+{N}))")
            log(f, f"  N={N}")

            load_ptr_wrapper[grid](table, out, GROUP_ID, N, BLOCK_SIZE=N)
            out_host = out.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}, BLOCK_SIZE: {N}")
            log(f, "\nOutput:")
            log(f, f"  loaded : {out_host.tolist()}")

            ref = buf1_ref.cpu()
            diff = (out_host - ref).abs().max().item()
            log(f, "\nValidation (vs buf1 contents via pointer table):")
            log(f, f"  reference : {ref.tolist()}")
            log(f, f"  max abs diff: {diff}")

            torch.testing.assert_close(out_host, ref, atol=0, rtol=0)
            log(f, "  assert_close: PASSED")

            log(f, "\nSummary:")
            log(f, f"  Kernel execution: SUCCESS (grid {grid})")
            log(f, "  Validation vs PyTorch reference: PASSED")
            log(f, f"  max abs diff: {diff}")

        except Exception:
            log(f, f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log(f, "\nSummary:")
            log(f, "  Kernel execution: FAILED")
            log(f, "  Validation vs PyTorch reference: NOT REACHED")
            log(f, "  Note: _load_ptr casts an integer address to a pointer")
            log(f, "        (tt.int_to_ptr); the QAIC backend may not lower this.")
            sys.exit(1)
        finally:
            log(f, "\n" + "-" * 36)

    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    if os.environ.get("LOADPTR_CHILD") == "1":
        main()
        sys.exit(0)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path  = os.path.join(LOG_DIR, f"test_load_ptr_{timestamp}.log")
    env = dict(os.environ, LOADPTR_CHILD="1", LOADPTR_LOG_PATH=log_path)
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
