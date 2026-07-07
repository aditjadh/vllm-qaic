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

from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe_kernel
from vllm.triton_utils import tl, triton

# ── Configuration ────────────────────────────────────────────────────────────
NUM_EXPERTS  = 4
M            = 8            # number of tokens
TOP_K        = 2
K            = 128          # input feature dim
N            = 64           # output feature dim
DTYPE        = torch.float16
DEVICE       = "qaic"

BLOCK_SIZE_M = 16
BLOCK_SIZE_N = 64
BLOCK_SIZE_K = 64
GROUP_SIZE_M = 1
MUL_ROUTED_WEIGHT = True

KERNEL_NAME = "fused_moe_kernel"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
# The parent owns the log path and passes it to the child via env so that an
# uncatchable compiler SIGABRT can still be recorded by the parent.
if "MOE_LOG_PATH" in os.environ:
    log_path  = os.environ["MOE_LOG_PATH"]
    timestamp = os.path.basename(log_path)[len("test_fused_moe_kernel_"):-len(".log")]
else:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path  = os.path.join(LOG_DIR, f"test_fused_moe_kernel_{timestamp}.log")


def align_block_size(topk_ids, block_size, num_experts):
    """Pure-PyTorch moe_align_block_size (replaces the unavailable custom op).

    topk_ids: [M, top_k] expert index per (token, slot), flattened row-major so
    token index t = m*top_k + k. Groups token indices by expert, padding each
    group up to a multiple of block_size with the value num_valid_tokens.
    """
    M, top_k = topk_ids.shape
    num_valid = M * top_k
    flat = topk_ids.reshape(-1)                  # [M*top_k], indexed by t
    sorted_ids = []
    expert_ids = []
    for e in range(num_experts):
        toks = torch.nonzero(flat == e, as_tuple=False).reshape(-1).tolist()
        if len(toks) == 0:
            continue
        # pad this expert's group to a multiple of block_size
        pad = (-len(toks)) % block_size
        toks = toks + [num_valid] * pad
        for b in range(0, len(toks), block_size):
            expert_ids.append(e)
        sorted_ids.extend(toks)
    sorted_ids = torch.tensor(sorted_ids, dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.tensor(expert_ids, dtype=torch.int32, device=topk_ids.device)
    num_post = torch.tensor([sorted_ids.numel()], dtype=torch.int32,
                            device=topk_ids.device)
    return sorted_ids, expert_ids, num_post


def reference_moe(a, b, topk_ids, topk_weights, mul_routed_weight):
    """out[m,k,:] = (a[m] @ b[expert].T) * weight."""
    M, top_k = topk_ids.shape
    N = b.shape[1]
    out = torch.zeros(M, top_k, N, dtype=torch.float32, device=a.device)
    for m in range(M):
        for k in range(top_k):
            e = int(topk_ids[m, k])
            r = a[m].float() @ b[e].float().T          # [N]
            if mul_routed_weight:
                r = r * float(topk_weights[m, k])
            out[m, k] = r
    return out.reshape(M * top_k, N)


def log(f, msg):
    print(msg)
    f.write(msg + "\n")
    f.flush()


def main():
    torch.manual_seed(42)

    a = torch.randn(M, K, dtype=DTYPE, device=DEVICE)
    b = torch.randn(NUM_EXPERTS, N, K, dtype=DTYPE, device=DEVICE)   # [E, N, K]
    topk_ids = torch.randint(0, NUM_EXPERTS, (M, TOP_K), dtype=torch.int32, device=DEVICE)
    topk_weights = torch.rand(M, TOP_K, dtype=torch.float32, device=DEVICE)

    # C is laid out as [M, top_k, N] flattened to [M*top_k, N].
    c = torch.zeros(M * TOP_K, N, dtype=DTYPE, device=DEVICE)

    a_ref = a.clone()
    b_ref = b.clone()
    topk_ids_ref = topk_ids.clone()
    topk_weights_ref = topk_weights.clone()

    sorted_token_ids, expert_ids, num_tokens_post_padded = align_block_size(
        topk_ids, BLOCK_SIZE_M, NUM_EXPERTS
    )
    num_valid_tokens = M * TOP_K
    EM = sorted_token_ids.numel()

    grid = (triton.cdiv(EM, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    with open(log_path, "w") as f:
        try:
            log(f, timestamp)
            log(f, f"Kernel: {KERNEL_NAME}")

            log(f, "\nInputs:")
            log(f, f"  a            : shape={tuple(a.shape)}, dtype={a.dtype}, device={a.device}")
            log(f, f"  b (E,N,K)    : shape={tuple(b.shape)}, dtype={b.dtype}")
            log(f, f"  topk_ids     : shape={tuple(topk_ids.shape)}, dtype={topk_ids.dtype}")
            log(f, f"  topk_weights : shape={tuple(topk_weights.shape)}, dtype={topk_weights.dtype}")
            log(f, f"  num_experts={NUM_EXPERTS}, M={M}, top_k={TOP_K}, K={K}, N={N}")
            log(f, f"  sorted_token_ids: {sorted_token_ids.tolist()}")
            log(f, f"  expert_ids      : {expert_ids.tolist()}")
            log(f, f"  num_tokens_post_padded: {int(num_tokens_post_padded)}, EM={EM}")
            log(f, f"  BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_N={BLOCK_SIZE_N}, "
                   f"BLOCK_SIZE_K={BLOCK_SIZE_K}, GROUP_SIZE_M={GROUP_SIZE_M}, "
                   f"MUL_ROUTED_WEIGHT={MUL_ROUTED_WEIGHT}")

            fused_moe_kernel[grid](
                a, b, c,
                None,                  # b_bias_ptr
                None,                  # a_scale_ptr
                None,                  # b_scale_ptr
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                N, K, EM, num_valid_tokens,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(2), b.stride(1),
                c.stride(0), c.stride(1),
                0, 0,                  # stride_asm, stride_ask
                0, 0, 0,               # stride_bse, stride_bsk, stride_bsn
                0, 0,                  # stride_bbe, stride_bbn
                group_n=0,
                group_k=0,
                naive_block_assignment=False,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                GROUP_SIZE_M=GROUP_SIZE_M,
                SPLIT_K=1,
                MUL_ROUTED_WEIGHT=MUL_ROUTED_WEIGHT,
                top_k=TOP_K,
                compute_type=tl.float16,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                per_channel_quant=False,
                HAS_BIAS=False,
            )

            # Force a device sync so any kernel compile/exec failure is raised
            # here rather than later at the first device read.
            c_host = c.cpu()

            log(f, "\nStatus: SUCCESS")
            log(f, "\nGrid Configuration:")
            log(f, f"  grid: {grid}  (cdiv(EM,BM) * cdiv(N,BN))")
            log(f, f"  block: BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_N={BLOCK_SIZE_N}, "
                   f"BLOCK_SIZE_K={BLOCK_SIZE_K}")

            log(f, "\nOutput:")
            log(f, f"  c : shape={tuple(c.shape)}, min={c_host.min().item():.4f}, "
                   f"max={c_host.max().item():.4f}, mean={c_host.float().mean().item():.4f}")

            # ── Reference validation ────────────────────────────────────────
            ref = reference_moe(a_ref, b_ref, topk_ids_ref, topk_weights_ref,
                                MUL_ROUTED_WEIGHT)
            diff = (c_host.float() - ref.cpu()).abs().max().item()
            log(f, "\nValidation (vs PyTorch reference per-expert matmul):")
            log(f, f"  max abs diff: {diff:.6f}")

            torch.testing.assert_close(c_host.float(), ref.cpu(), atol=1e-1, rtol=1e-2)
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
    if os.environ.get("MOE_CHILD") == "1":
        main()
        sys.exit(0)

    # Parent role: own the log file, run the kernel launch in a child process,
    # and if the child is killed by an uncatchable compiler abort (SIGABRT),
    # record a FAILURE entry that the child could not write itself.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path  = os.path.join(LOG_DIR, f"test_fused_moe_kernel_{timestamp}.log")

    env = dict(os.environ, MOE_CHILD="1", MOE_LOG_PATH=log_path)
    proc = subprocess.run([sys.executable, __file__], env=env,
                          capture_output=True, text=True)
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="")

    if proc.returncode != 0 and not (proc.stdout and "Status: SUCCESS" in proc.stdout):
        # Append a failure record if the child aborted before logging it.
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
