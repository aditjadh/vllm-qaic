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

from vllm.v1.worker.gpu.spec_decode.eagle import update_eagle_inputs

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEVICE = "qaic"
KERNEL_NAME = "_update_eagle_inputs_kernel"


def main(log_path):
    torch.manual_seed(42)
    num_reqs = 3
    hidden = 16
    max_model_len = 100
    max_tokens = 8

    draft_tokens = torch.tensor([11, 22, 33], dtype=torch.int64, device=DEVICE)
    output_hidden = torch.randn(num_reqs, hidden, dtype=torch.float32, device=DEVICE)

    input_buffers = SimpleNamespace(
        input_ids=torch.full((max_tokens,), -1, dtype=torch.int64, device=DEVICE),
        positions=torch.tensor([5, 50, 99, 0, 0, 0, 0, 0], dtype=torch.int64, device=DEVICE),
        seq_lens=torch.tensor([6, 51, 100, 0, 0, 0, 0, 0], dtype=torch.int32, device=DEVICE),
    )
    hidden_states = torch.zeros(max_tokens, hidden, dtype=torch.float32, device=DEVICE)

    dt_r, oh_r = draft_tokens.clone(), output_hidden.clone()
    pos_r = input_buffers.positions.clone()
    sl_r = input_buffers.seq_lens.clone()
    with open(log_path, "w") as fp:
        log = logger(fp)
        try:
            log(os.path.basename(log_path)[:-4]); log(f"Kernel: {KERNEL_NAME}")
            log("\nInputs:")
            log(f"  draft_tokens={draft_tokens.cpu().tolist()}")
            log(f"  output_hidden: shape={tuple(output_hidden.shape)}, device={output_hidden.device}")
            log(f"  positions(before)={pos_r.cpu().tolist()}, seq_lens(before)={sl_r.cpu().tolist()}")
            log(f"  num_reqs={num_reqs}, hidden={hidden}, max_model_len={max_model_len}")

            update_eagle_inputs(draft_tokens, output_hidden, input_buffers,
                                hidden_states, max_model_len)
            ii = input_buffers.input_ids.cpu()
            hs = hidden_states.cpu()
            pos = input_buffers.positions.cpu()
            sl = input_buffers.seq_lens.cpu()
            log("\nStatus: SUCCESS")
            log("\nGrid Configuration:")
            log(f"  grid: ({num_reqs},), BLOCK_SIZE=1024")
            log("\nOutput:")
            log(f"  input_ids[:{num_reqs}]={ii[:num_reqs].tolist()}")
            log(f"  positions[:{num_reqs}]={pos[:num_reqs].tolist()}")
            log(f"  seq_lens[:{num_reqs}]={sl[:num_reqs].tolist()}")

            # reference
            errs = 0
            for r in range(num_reqs):
                if int(ii[r]) != int(dt_r[r]):
                    errs += 1
                if (hs[r] - oh_r[r].cpu()).abs().max().item() > 1e-5:
                    errs += 1
                if int(pos[r]) != min(int(pos_r[r]) + 1, max_model_len - 1):
                    errs += 1
                if int(sl[r]) != min(int(sl_r[r]) + 1, max_model_len):
                    errs += 1
            log("\nValidation (draft->input_id, hidden copy, pos/seq +1 clamped):")
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
    run_test(main, LOG_DIR, "update_eagle_inputs")
