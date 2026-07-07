# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
import sys, os, datetime, traceback, torch

sys.path.insert(0, "/local/mnt/workspace/aditjadh/aisyssol/aditjadh/triton/vllm")

LOG_FILE = "/local/mnt/workspace/aditjadh/aisyssol/aditjadh/triton/logs/log_temperature_kernel.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def log(content):
    with open(LOG_FILE, "a") as f:
        f.write(content + "\n")

timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

try:
    import triton
    from vllm.v1.worker.gpu.sample.gumbel import _temperature_kernel

    num_reqs, vocab_size, BLOCK_SIZE = 4, 32000, 8192
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    device = "qaic"

    logits = torch.randn(num_reqs, vocab_size, dtype=torch.float16, device=device)
    logits_before = logits.clone()
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    temperature = torch.tensor([0.5, 1.0, 0.8, 2.0], dtype=torch.float32, device=device)

    _temperature_kernel[(num_reqs, num_blocks)](
        logits, logits.stride(0), idx_mapping, temperature, vocab_size, BLOCK_SIZE=BLOCK_SIZE,
    )

    logits_cpu = logits.cpu().float()
    before_cpu = logits_before.cpu().float()
    temp_cpu = temperature.cpu()

    # PyTorch reference: divide by temperature, skip if 0.0 or 1.0
    ref = before_cpu.clone()
    for i in range(num_reqs):
        t = temp_cpu[i].item()
        if t not in (0.0, 1.0):
            ref[i] = before_cpu[i] / t

    results = []
    for i in range(num_reqs):
        t = temp_cpu[i].item()
        match = torch.allclose(logits_cpu[i], ref[i], atol=1e-2)
        if t in (0.0, 1.0):
            results.append(f"  req {i}: temp={t} (skipped), kernel_matches_ref={match}")
        else:
            results.append(f"  req {i}: temp={t}, kernel_matches_ref={match}, kernel={logits_cpu[i][:5].tolist()}, ref={ref[i][:5].tolist()}")

    output = "\n".join(results)
    log(f"{timestamp}\nKernel: _temperature_kernel\nStatus: SUCCESS\n\nInputs:\n- num_reqs: {num_reqs}\n- vocab_size: {vocab_size}\n- temperature: {temp_cpu.tolist()}\n\nGrid Configuration:\n- grid: ({num_reqs}, {num_blocks}, 1)\n- BLOCK_SIZE: {BLOCK_SIZE}\n\nOutput:\n{output}\n\n------------------------------------\n")
    print(f"Status: SUCCESS\nGrid: ({num_reqs}, {num_blocks}, 1), BLOCK_SIZE={BLOCK_SIZE}\n{output}")

except Exception:
    tb = traceback.format_exc()
    log(f"{timestamp}\nKernel: _temperature_kernel\nStatus: FAILURE\n\nError:\n{tb}------------------------------------\n")
    print(f"Status: FAILURE\n{tb}")
