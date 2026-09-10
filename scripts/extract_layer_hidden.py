#!/usr/bin/env python3
"""Extract layer hidden states for the oracle-routing probe.

The PLE reader is installed as a *post-forward* hook on
``model.model.layers[layer_index]``, so the query it sees is the output of that
block.  In HF ``output_hidden_states`` terms that is ``hidden_states[layer+1]``
(``hidden_states[0]`` is the embedding output).

For each QA item we store two prompt-level summaries:

* ``last``: the last prompt-token hidden state;
* ``mean``: mean-pooled prompt hidden states.

The probe can then test whether "should the PLE reader be used?" is linearly
decodable from the same hidden state the reader itself sees.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch


def _load_run_phase0():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_phase0.py"
    spec = importlib.util.spec_from_file_location("run_phase0", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--qa-file", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument(
        "--hidden-state-index",
        type=int,
        default=None,
        help="HF hidden_states index; default layer + 1 (post-block output)",
    )
    parser.add_argument("--prompt-template", default=None)
    parser.add_argument("--boolq-prompt-template", default=None)
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--chat-enable-thinking", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    phase0 = _load_run_phase0()
    tokenizer, model = phase0._load_model(args.model, args.device)
    items = phase0._load_qa_file(args.qa_file)
    if args.limit:
        items = items[: args.limit]
    hidden_index = (
        int(args.hidden_state_index)
        if args.hidden_state_index is not None
        else int(args.layer) + 1
    )

    last_rows: list[np.ndarray] = []
    mean_rows: list[np.ndarray] = []
    tasks: list[str] = []
    questions: list[str] = []
    answers: list[str] = []
    prompt_lens: list[int] = []

    with torch.no_grad():
        for idx, item in enumerate(items):
            prompt_ids = phase0._qa_prompt_ids(
                tokenizer,
                item,
                args.prompt_template,
                args.boolq_prompt_template,
                chat_template=args.chat_template,
                chat_enable_thinking=args.chat_enable_thinking,
            )
            ids = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
            out = model(input_ids=ids, output_hidden_states=True)
            hidden = out.hidden_states[hidden_index][0]
            last_rows.append(hidden[-1].float().cpu().numpy())
            mean_rows.append(hidden.mean(dim=0).float().cpu().numpy())
            tasks.append(str(item.get("task", "unknown")))
            questions.append(str(item.get("question", "")))
            answers.append(str(item.get("answer", "")))
            prompt_lens.append(len(prompt_ids))
            if (idx + 1) % 200 == 0:
                print(f"  hidden {idx + 1}/{len(items)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        last=np.stack(last_rows).astype(np.float32),
        mean=np.stack(mean_rows).astype(np.float32),
        tasks=np.asarray(tasks),
        questions=np.asarray(questions),
        answers=np.asarray(answers),
        prompt_len=np.asarray(prompt_lens, dtype=np.int64),
        layer=np.asarray([args.layer], dtype=np.int64),
        hidden_state_index=np.asarray([hidden_index], dtype=np.int64),
        chat_template=np.asarray([int(bool(args.chat_template))], dtype=np.int64),
    )
    print(f"wrote {args.output} ({len(last_rows)} items, hidden={hidden_index})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
