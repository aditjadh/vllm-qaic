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

from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe_kernel_gptq_awq
from vllm.triton_utils import tl, triton

# ── Configuration ────────────────────────────────────────────────────────────
# Tests the use_int8_w8a16 path (no zero point): B is int8 [E, N, K], dequantized
# as (b - 128) * b_scale, with b_scale [E, N, K//group_size].
NUM_EXPERTS  = 4
M            = 8            # number of tokens
TOP_K        = 2
K            = 128          # input feature dim
N            = 64           # output feature dim
GROUP_SIZE   = 128          # one scale group along K
DTYPE        = torch.float16
DEVICE       = "qaic"

BLOCK_SIZE_M = 16
BLOCK_SIZE_N = 64
BLOCK_SIZE_K = 64
GROUP_SIZE_M = 1
MUL_ROUTED_WEIGHT = True
ZP_NUM       = 128          # b_zp_num when has_zp=False and use_int8_w8a16

KERNEL_NAME = "fused_moe_kernel_gptq_awq"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
if "MOEQ_LOG_PATH" in os.environ:
    log_path  = os.environ["MOEQ_LOG_PATH"]
    timestamp = os.path.basename(log_path)[len("test_fused_moe_kernel_gptq_awq_"):-len(".log")]
else:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path  = os.path.join(LOG_DIR, f"test_fused_moe_kernel_gptq_awq_{timestamp}.log")


def align_block_size(topk_ids, block_size, num_experts):
    """Pure-PyTorch moe_align_block_size (replaces the unavailable custom op)."""
    M, top_k = topk_ids.shape
    num_valid = M * top_k
    flat = topk_ids.reshape(-1)
    sorted_ids = []
    expert_ids = []
    for e in range(num_experts):
        toks = torch.nonzero(flat == e, as_tuple=False).reshape(-1).tolist()
        if len(toks) == 0:
            continue
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


def reference_moe(a, b_int, b_scale, topk_ids, topk_weights):
    """out[m,k,:] = (a[m] @ dequant(b[e]).T) * weight, b dequant = (b-128)*scale."""
    M, top_k = topk_ids.shape
    E, N, K = b_int.shape
    num_groups = b_scale.shape[2]
    gsize = K // num_groups
    out = torch.zeros(M * top_k, N, dtype=torch.float32, device=a.device)
    for m in range(M):
        for k in range(top_k):
            e = int(topk_ids[m, k])
            # dequantize expert e: [N, K]
            bf = (b_int[e].float() - ZP_NUM)
            scale_full = b_scale[e].repeat_interleave(gsize, dim=1)  # [N, K]
            bf = bf * scale_full.float()
            r = a[m].float() @ bf.T          # [N]
            r = r * float(topk_weights[m, k])
            out[m * top_k + k] = r
    return out


def log(f, msg):
    print(msg)
    f.write(msg + "\n")
    f.flush()


def main():
    torch.manual_seed(42)

    a = torch.randn(M, K, dtype=DTYPE, device=DEVICE)
    b_int = torch.randint(0, 256, (NUM_EXPERTS, N, K), dtype=torch.uint8, device=DEVICE)
    num_groups = K // GROUP_SIZE
    b_scale = (torch.rand(NUM_EXPERTS, N, num_groups, dtype=DTYPE, device=DEVICE) * 0.02 + 0.01)
    topk_ids = torch.randint(0, NUM_EXPERTS, (M, TOP_K), dtype=torch.int32, device=DEVICE)
    topk_weights = torch.rand(M, TOP_K, dtype=torch.float32, device=DEVICE)

    c = torch.zeros(M * TOP_K, N, dtype=DTYPE, device=DEVICE)

    a_ref = a.clone()
    b_int_ref = b_int.clone()
    b_scale_ref = b_scale.clone()
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
            log(f, f"  b_int (E,N,K): shape={tuple(b_int.shape)}, dtype={b_int.dtype}")
            log(f, f"  b_scale      : shape={tuple(b_scale.shape)}, dtype={b_scale.dtype}")
            log(f, f"  topk_ids     : shape={tuple(topk_ids.shape)}, dtype={topk_ids.dtype}")
            log(f, f"  topk_weights : shape={tuple(topk_weights.shape)}")
            log(f, f"  num_experts={NUM_EXPERTS}, M={M}, top_k={TOP_K}, K={K}, N={N}, "
                   f"group_size={GROUP_SIZE}")
            log(f, f"  quant: use_int8_w8a16=True, has_zp=False (b_zp_num={ZP_NUM})")
            log(f, f"  num_tokens_post_padded: {int(num_tokens_post_padded)}, EM={EM}")
            log(f, f"  BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_N={BLOCK_SIZE_N}, "
                   f"BLOCK_SIZE_K={BLOCK_SIZE_K}, GROUP_SIZE_M={GROUP_SIZE_M}")

            block_k_diviable = (K % BLOCK_SIZE_K == 0)

            fused_moe_kernel_gptq_awq[grid](
                a, b_int, c,
                b_scale,
                None,                  # b_zp_ptr
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                N, K, EM, num_valid_tokens,
                a.stride(0), a.stride(1),
                b_int.stride(0), b_int.stride(2), b_int.stride(1),
                c.stride(0), c.stride(1),
                b_scale.stride(0), b_scale.stride(2), b_scale.stride(1),
                0, 0, 0,               # stride_bze, stride_bzk, stride_bzn
                block_k_diviable=block_k_diviable,
                group_size=GROUP_SIZE,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                GROUP_SIZE_M=GROUP_SIZE_M,
                SPLIT_K=1,
                MUL_ROUTED_WEIGHT=MUL_ROUTED_WEIGHT,
                top_k=TOP_K,
                compute_type=tl.float16,
                has_zp=False,
                use_int4_w4a16=False,
                use_int8_w8a16=True,
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
            ref = reference_moe(a_ref, b_int_ref, b_scale_ref, topk_ids_ref,
                                topk_weights_ref)
            diff = (c_host.float() - ref.cpu()).abs().max().item()
            log(f, "\nValidation (vs PyTorch dequant + per-expert matmul):")
            log(f, f"  max abs diff: {diff:.6f}")

            torch.testing.assert_close(c_host.float(), ref.cpu(), atol=2e-1, rtol=2e-2)
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
    if os.environ.get("MOEQ_CHILD") == "1":
        main()
        sys.exit(0)

    # Parent role: own the log file, run the kernel launch in a child process,
    # and if the child is killed by an uncatchable compiler abort (SIGABRT),
    # record a FAILURE entry that the child could not write itself.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path  = os.path.join(LOG_DIR, f"test_fused_moe_kernel_gptq_awq_{timestamp}.log")

    env = dict(os.environ, MOEQ_CHILD="1", MOEQ_LOG_PATH=log_path)
    proc = subprocess.run([sys.executable, __file__], env=env,
                          capture_output=True, text=True)
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="")

    if proc.returncode != 0 and not (proc.stdout and "Status: SUCCESS" in proc.stdout):
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
