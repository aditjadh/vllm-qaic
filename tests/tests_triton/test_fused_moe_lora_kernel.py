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

import vllm.lora.ops.triton_ops.fused_moe_lora_op as moe_lora_op
from vllm.lora.ops.triton_ops.fused_moe_lora_op import _fused_moe_lora_shrink


# QAIC does not support the uint64 dtype that the upstream _get_ptr uses to
# build the LoRA pointer LUT. data_ptr() values are positive and fit in int64,
# so patch it to use int64 (same bit pattern for realistic addresses).
def _get_ptr_int64(lora_weights, device):
    key = tuple(w.data_ptr() for w in lora_weights)
    cached = moe_lora_op._LORA_PTR_DICT.get(key)
    if cached is not None:
        return cached
    ptrs = [w.data_ptr() for w in lora_weights]
    ptr_tensor = torch.tensor(ptrs, device=device, dtype=torch.int64)
    moe_lora_op._LORA_PTR_DICT[key] = ptr_tensor
    return ptr_tensor


moe_lora_op._get_ptr = _get_ptr_int64

# ── Configuration ────────────────────────────────────────────────────────────
# Drives _fused_moe_lora_kernel via the shrink wrapper. Shrink computes, per
# output row t (0..M*top_k-1) with m = t // top_k:
#   out[t, :] = A[m] @ lora_a[lora_id, expert_id, :, :]^T   (rank x K weight)
MAX_LORAS    = 1
NUM_EXPERTS  = 1
M            = 8            # number of tokens
TOP_K        = 2
K            = 64           # hidden dim
RANK         = 16           # max_lora_rank == output N for shrink
NUM_SLICES   = 1
DTYPE        = torch.float16
DEVICE       = "qaic"

BLOCK_SIZE_M = 16
BLOCK_SIZE_N = 16
BLOCK_SIZE_K = 64
GROUP_SIZE_M = 1
NUM_WARPS    = 4
NUM_STAGES   = 1
SPLIT_K      = 1

KERNEL_NAME = "_fused_moe_lora_kernel (via shrink)"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
if "MOELORA_LOG_PATH" in os.environ:
    log_path  = os.environ["MOELORA_LOG_PATH"]
    timestamp = os.path.basename(log_path)[len("test_fused_moe_lora_kernel_"):-len(".log")]
else:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path  = os.path.join(LOG_DIR, f"test_fused_moe_lora_kernel_{timestamp}.log")


def log(f, msg):
    print(msg)
    f.write(msg + "\n")
    f.flush()


def main():
    torch.manual_seed(42)

    num_valid_tokens = M * TOP_K       # 16
    EM = BLOCK_SIZE_M                   # one M-block (16)

    # A: (M, K)
    qcurr = torch.randn(M, K, dtype=DTYPE, device=DEVICE)
    # lora_a_stacked: list of (max_loras, num_experts, rank, K)
    lora_a = torch.randn(MAX_LORAS, NUM_EXPERTS, RANK, K, dtype=DTYPE, device=DEVICE)
    lora_a_stacked = [lora_a]

    # Output cache: (num_slices, M, top_k, rank)
    a_intermediate_cache1 = torch.zeros(
        NUM_SLICES, M, TOP_K, RANK, dtype=DTYPE, device=DEVICE
    )

    topk_weights = torch.rand(M, TOP_K, dtype=torch.float32, device=DEVICE)

    # MoE alignment metadata, shaped (max_loras, ...).
    sorted_token_ids = torch.arange(
        EM, dtype=torch.int32, device=DEVICE
    ).reshape(MAX_LORAS, EM)              # rows 0..15 -> tokens 0..15
    expert_ids = torch.zeros(MAX_LORAS, 1, dtype=torch.int32, device=DEVICE)
    num_tokens_post_padded = torch.tensor(
        [num_valid_tokens], dtype=torch.int32, device=DEVICE
    )

    # grid axis2 = max_loras + 1 (last handles the no-lora case)
    lora_ids = torch.tensor([0, -1], dtype=torch.int32, device=DEVICE)
    adapter_enabled = torch.tensor([1], dtype=torch.int32, device=DEVICE)

    qcurr_ref  = qcurr.clone()
    lora_a_ref = lora_a.clone()

    grid = (
        SPLIT_K
        * ((EM + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M)
        * ((RANK + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N),
        NUM_SLICES,
        MAX_LORAS + 1,
    )

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")

            log(f, "\nInputs:")
            log(f, f"  qcurr_hidden (A) : shape={tuple(qcurr.shape)}, dtype={qcurr.dtype}, device={qcurr.device}")
            log(f, f"  lora_a (L,E,R,K) : shape={tuple(lora_a.shape)}, dtype={lora_a.dtype}")
            log(f, f"  out cache        : shape={tuple(a_intermediate_cache1.shape)}")
            log(f, f"  topk_weights     : shape={tuple(topk_weights.shape)}")
            log(f, f"  sorted_token_ids : shape={tuple(sorted_token_ids.shape)}")
            log(f, f"  expert_ids       : shape={tuple(expert_ids.shape)}")
            log(f, f"  lora_ids={lora_ids.tolist()}, adapter_enabled={adapter_enabled.tolist()}")
            log(f, f"  max_loras={MAX_LORAS}, num_experts={NUM_EXPERTS}, M={M}, top_k={TOP_K}, "
                   f"K={K}, rank(N)={RANK}, num_slices={NUM_SLICES}")
            log(f, f"  num_valid_tokens={num_valid_tokens}, EM={EM}")
            log(f, f"  BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_N={BLOCK_SIZE_N}, "
                   f"BLOCK_SIZE_K={BLOCK_SIZE_K}, GROUP_SIZE_M={GROUP_SIZE_M}, SPLIT_K={SPLIT_K}")
            log(f, "  NOTE: _get_ptr patched to build the LoRA pointer LUT as int64 "
                   "(QAIC rejects uint64).")

            _fused_moe_lora_shrink(
                a_intermediate_cache1,
                qcurr,
                lora_a_stacked,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                TOP_K,
                lora_ids,
                adapter_enabled,
                DEVICE,
                RANK,                # N
                M,
                EM,
                K,
                num_valid_tokens,    # num_tokens
                NUM_EXPERTS,
                NUM_SLICES,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                GROUP_SIZE_M,
                NUM_WARPS,
                NUM_STAGES,
                SPLIT_K,
                mul_routed_weight=False,
                use_gdc=False,
            )

            # Force a device sync so any kernel compile/exec failure is raised
            # here rather than later at the first device read.
            out_host = a_intermediate_cache1.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}  (tiles, num_slices, max_loras+1)")
            log(f, f"  block: BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_N={BLOCK_SIZE_N}, "
                   f"BLOCK_SIZE_K={BLOCK_SIZE_K}")

            log(f, "\nOutput:")
            log(f, f"  out : shape={tuple(a_intermediate_cache1.shape)}, "
                   f"min={out_host.min().item():.4f}, max={out_host.max().item():.4f}, "
                   f"mean={out_host.float().mean().item():.4f}")

            # ── Reference validation ────────────────────────────────────────
            # out_flat[t, :] = A[t // top_k] @ lora_a[0, 0]^T
            ref = torch.zeros(M * TOP_K, RANK, dtype=torch.float32)
            wa = lora_a_ref[0, 0].float()        # (rank, K)
            for t in range(M * TOP_K):
                m = t // TOP_K
                ref[t] = qcurr_ref[m].float() @ wa.T
            ref = ref.reshape(M, TOP_K, RANK)
            got = out_host[0].float()
            diff = (got - ref).abs().max().item()
            log(f, "\nValidation (vs PyTorch A @ lora_a^T):")
            log(f, f"  max abs diff: {diff:.6f}")

            torch.testing.assert_close(got, ref, atol=1e-1, rtol=1e-2)
            log(f, "  assert_close: PASSED")

            log(f, "\nSummary:")
            log(f, f"  Kernel execution: SUCCESS (grid {grid})")
            log(f, "  Validation vs PyTorch reference: PASSED")
            log(f, f"  max abs diff: {diff:.6f}")

        except Exception:
            msg = f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}"
            log(f, msg)
            log(f, "\nSummary:")
            log(f, "  Validation vs PyTorch reference: FAILED")
            sys.exit(1)

        finally:
            log(f, "\n" + "-" * 36)

    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    # Child role: actually run the kernel.
    if os.environ.get("MOELORA_CHILD") == "1":
        main()
        sys.exit(0)

    # Parent role: own the log file, run the launch in a child process so an
    # uncatchable compiler abort (SIGABRT) can still be recorded.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path  = os.path.join(LOG_DIR, f"test_fused_moe_lora_kernel_{timestamp}.log")

    env = dict(os.environ, MOELORA_CHILD="1", MOELORA_LOG_PATH=log_path)
    proc = subprocess.run([sys.executable, __file__], env=env,
                          capture_output=True, text=True)
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="")

    # Only append a compiler-abort record if the child was KILLED BY A SIGNAL
    # (negative return code). A positive code means main() already caught the
    # exception and logged it via try/except.
    if proc.returncode < 0 and not (proc.stdout and "Status: SUCCESS" in proc.stdout):
        with open(log_path, "a") as f:
            f.write(f"\nStatus: FAILURE\n\nError:\n")
            f.write(f"Child process exited with code {proc.returncode} "
                    f"(negative => killed by signal; -6/134 => SIGABRT).\n")
            f.write("The Triton->Hexagon compiler aborted with an MLIR "
                    "assertion before the kernel could run:\n")
            tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
            f.write(tail + "\n")
            f.write("\nSummary:\n")
            f.write("  Kernel execution: FAILED (compiler SIGABRT, uncatchable)\n")
            f.write("  Validation vs PyTorch reference: NOT REACHED\n")
            f.write("\n" + "-" * 36 + "\n")
        print(f"\nLog saved to: {log_path}")
    sys.exit(proc.returncode)
