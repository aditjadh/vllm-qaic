# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
import os
import sys
import traceback
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vllm"))
from _ktest import run_test, logger

from vllm.v1.worker.gpu.spec_decode.eagle import prepare_eagle_inputs

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "_prepare_eagle_inputs_kernel"


def main(log_path):
    torch.manual_seed(42)
    num_reqs = 2
    max_tokens = 16
    max_reqs = 4

    qlens = [4, 3]
    qsl = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    qsl[1:] = torch.tensor(qlens, dtype=torch.int32).cumsum(0)
    num_tokens = int(qsl[-1])

    target_input_ids = torch.arange(100, 100 + num_tokens, dtype=torch.int64, device=DEVICE)
    target_positions = torch.tensor([0, 1, 2, 3, 0, 1, 2], dtype=torch.int64, device=DEVICE)

    input_buffers = SimpleNamespace(
        input_ids=torch.full((max_tokens,), -1, dtype=torch.int64, device=DEVICE),
        positions=torch.full((max_tokens,), -1, dtype=torch.int64, device=DEVICE),
    )
    input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        input_ids=target_input_ids,
        positions=target_positions,
        idx_mapping=torch.tensor([0, 1], dtype=torch.int32, device=DEVICE),
        query_start_loc=qsl,
    )
    num_sampled = torch.tensor([1, 1], dtype=torch.int32, device=DEVICE)  # decode
    num_rejected = torch.tensor([0, 0], dtype=torch.int32, device=DEVICE)
    last_sampled = torch.tensor([555, 666, 0, 0], dtype=torch.int64, device=DEVICE)
    next_prefill = torch.tensor([0, 0, 0, 0], dtype=torch.int32, device=DEVICE)

    tii_r, tp_r, qsl_r, ls_r = (target_input_ids.clone(), target_positions.clone(),
                                qsl.clone(), last_sampled.clone())
    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(os.path.basename(log_path)[:-4]); log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  query_lens={qlens}, query_start_loc={qsl.cpu().tolist()}")
            log(f"  target_input_ids={target_input_ids.cpu().tolist()}")
            log(f"  target_positions={target_positions.cpu().tolist()}")
            log(f"  num_sampled={num_sampled.cpu().tolist()}, num_rejected={num_rejected.cpu().tolist()}")
            log(f"  last_sampled={last_sampled.cpu().tolist()}")

            last_token_indices = prepare_eagle_inputs(
                input_buffers, input_batch, num_sampled, num_rejected,
                last_sampled, next_prefill,
            )
            lti = last_token_indices.cpu()
            ii = input_buffers.input_ids.cpu()
            pos = input_buffers.positions.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: ({num_reqs},), BLOCK_SIZE=1024")
            log("\nOutput:")
            log(f"  last_token_indices={lti.tolist()}")
            log(f"  eagle input_ids[:{num_tokens}]={ii[:num_tokens].tolist()}")
            log(f"  eagle positions[:{num_tokens}]={pos[:num_tokens].tolist()}")

            # reference: shift target_input_ids left by 1 within each query; set
            # last token = next sampled token; copy positions.
            ref_ii = [-1] * max_tokens
            ref_pos = [-1] * max_tokens
            ref_lti = []
            for b in range(num_reqs):
                qs = int(qsl_r[b]); qe = int(qsl_r[b + 1]); ql = qe - qs
                for j in range(1, ql):
                    ref_ii[qs + j - 1] = int(tii_r[qs + j])
                last_idx = qs + ql - 1
                ref_ii[last_idx] = int(ls_r[b])
                ref_lti.append(last_idx)
                for j in range(ql):
                    ref_pos[qs + j] = int(tp_r[qs + j])
            errs = 0
            for b in range(num_reqs):
                if int(lti[b]) != ref_lti[b]:
                    errs += 1
            for i in range(num_tokens):
                if ref_ii[i] != -1 and int(ii[i]) != ref_ii[i]:
                    errs += 1
                if ref_pos[i] != -1 and int(pos[i]) != ref_pos[i]:
                    errs += 1
            log("\nValidation (shift input_ids, set next token, copy positions):")
            log(f"  ref last_token_indices={ref_lti}")
            log(f"  errors: {errs}")
            assert errs == 0
            log("  validation: PASSED")
            log("\nSummary:")
            log(f"  Kernel execution: SUCCESS (grid ({num_reqs},))")
            log("  Validation vs PyTorch reference: PASSED")
            log(f"  errors: {errs}")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Validation vs PyTorch reference: FAILED")
            fp.write("\n" + "-" * 36 + "\n"); sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "prepare_eagle_inputs")
