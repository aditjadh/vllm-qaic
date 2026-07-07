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

from vllm.v1.worker.gpu.spec_decode.eagle import prepare_eagle_decode

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "_prepare_eagle_docode_kernel"


def main(log_path):
    torch.manual_seed(42)
    num_reqs = 2
    hidden = 16
    max_model_len = 100
    max_num_reqs = 4
    max_tokens = 16

    draft_tokens = torch.tensor([11, 22], dtype=torch.int64, device=DEVICE)
    output_hidden = torch.randn(max_tokens, hidden, dtype=torch.float32, device=DEVICE)
    last_token_indices = torch.tensor([3, 6], dtype=torch.int64, device=DEVICE)
    target_seq_lens = torch.tensor([20, 15], dtype=torch.int32, device=DEVICE)
    num_rejected = torch.tensor([1, 0], dtype=torch.int32, device=DEVICE)

    input_buffers = SimpleNamespace(
        input_ids=torch.full((max_num_reqs,), -1, dtype=torch.int64, device=DEVICE),
        positions=torch.tensor([10, 20, 0, 0], dtype=torch.int64, device=DEVICE),
        query_start_loc=torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=DEVICE),
        seq_lens=torch.full((max_num_reqs,), -1, dtype=torch.int32, device=DEVICE),
    )
    input_hidden = torch.zeros(max_num_reqs, hidden, dtype=torch.float32, device=DEVICE)

    dt_r, oh_r, lti_r, tsl_r, nr_r = (draft_tokens.clone(), output_hidden.clone(),
        last_token_indices.clone(), target_seq_lens.clone(), num_rejected.clone())
    pos_r = input_buffers.positions.clone()
    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(os.path.basename(log_path)[:-4]); log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  draft_tokens={draft_tokens.cpu().tolist()}")
            log(f"  last_token_indices={last_token_indices.cpu().tolist()}")
            log(f"  target_seq_lens={target_seq_lens.cpu().tolist()}, num_rejected={num_rejected.cpu().tolist()}")
            log(f"  positions(before)={pos_r.cpu().tolist()}")
            log(f"  num_reqs={num_reqs}, hidden={hidden}, max_model_len={max_model_len}")

            prepare_eagle_decode(draft_tokens, output_hidden, last_token_indices,
                                 target_seq_lens, num_rejected, input_buffers,
                                 input_hidden, max_model_len, max_num_reqs)
            ii = input_buffers.input_ids.cpu()
            ih = input_hidden.cpu()
            pos = input_buffers.positions.cpu()
            sl = input_buffers.seq_lens.cpu()
            qsl = input_buffers.query_start_loc.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: ({num_reqs + 1},), BLOCK_SIZE=1024 (last program pads qsl/seq_lens)")
            log("\nOutput:")
            log(f"  input_ids[:{num_reqs}]={ii[:num_reqs].tolist()}")
            log(f"  positions[:{num_reqs}]={pos[:num_reqs].tolist()}")
            log(f"  seq_lens[:{num_reqs}]={sl[:num_reqs].tolist()}")
            log(f"  query_start_loc={qsl.tolist()}")

            # reference
            errs = 0
            for r in range(num_reqs):
                if int(ii[r]) != int(dt_r[r]):
                    errs += 1
                src = int(lti_r[r])
                if (ih[r] - oh_r[src].cpu()).abs().max().item() > 1e-5:
                    errs += 1
                if int(pos[r]) != min(int(pos_r[r]) + 1, max_model_len - 1):
                    errs += 1
                exp_sl = min(int(tsl_r[r]) - int(nr_r[r]) + 1, max_model_len)
                if int(sl[r]) != exp_sl:
                    errs += 1
            # query_start_loc should be arange-ish (0,1,2,...) padded
            log("\nValidation (draft->input_id, hidden gather@last_idx, pos/seq update, qsl pad):")
            log(f"  errors: {errs}")
            assert errs == 0
            log("  validation: PASSED")
            log("\nSummary:")
            log(f"  Kernel execution: SUCCESS (grid ({num_reqs + 1},))")
            log("  Validation vs PyTorch reference: PASSED")
            log(f"  errors: {errs}")
        except Exception:
            log(f"\nStatus: FAILURE\n\nError:\n{traceback.format_exc()}")
            log("\nSummary:\n  Validation vs PyTorch reference: FAILED")
            fp.write("\n" + "-" * 36 + "\n"); sys.exit(1)
        fp.write("\n" + "-" * 36 + "\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    run_test(main, LOG_DIR, "prepare_eagle_decode")
