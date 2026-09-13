#!/usr/bin/env python3
"""Round 168: does the evaluation template itself make the memory content-blind?

Round 167 found that at the position where an answer is generated, every item in
a task addresses a byte-identical row, because each QA template ends in a fixed
suffix.  Where that holds, a ``real``/``control`` comparison at that position is
a **null forced by the evaluation premise** -- not evidence about the
architecture.  That finding rested on one case; this script measures it over the
evaluation file we actually use, for every prompt convention in the queue.

For each item, two positions:

* **generation** -- the last token of the templated prompt, i.e. where the answer
  is produced (template-suffixed);
* **content** -- the last token of the question itself (item-specific, and the
  addressing that *should* have been used).

The row-id tuple is computed from the real PLE spec, so the measurement is about
the deployed geometry rather than an abstraction.  The general statement is not
empirical: a row id depends on at most three tokens, so a fixed suffix of >= 3
tokens forces a single row for the whole task; tokenizer merging across the
question/suffix boundary is the only escape, and the distinct-tuple count
measures whether it happened.

CPU only.  No GPU, no model weights -- only the tokenizer.

Usage::

    python scripts/round168_template_address_audit.py \
        --qa-file data/qa-standard/eval-600b.jsonl \
        --model /root/autodl-tmp/qwen35-ple/models/Qwen3.5-0.8B \
        --output outputs/round168/template-audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen35_ple.ple_hash import real_spec
from qwen35_ple.template_audit import (
    address_diversity,
    per_task_diversity,
    verdict_from_tasks,
)

PROMPT = "Question: {question}\nAnswer:"
BOOLQ_PROMPT = "Question: {question}\nAnswer with one word, Yes or No:"


def log(msg: str) -> None:
    print(f"[r168-tmpl] {msg}", flush=True)


def _last_row(spec, ids: list[int]) -> tuple[int, ...]:
    """Row-id tuple at the final token of ``ids``."""
    if len(ids) < 3:
        raise ValueError(f"need at least 3 tokens for a full window, got {len(ids)}")
    rows = spec.rowids_for_seq(tuple(int(x) for x in ids))
    return tuple(int(x) for x in rows[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--qa-file", default="data/qa-standard/eval-600b.jsonl")
    ap.add_argument("--model", required=True, help="tokenizer source")
    ap.add_argument("--max-items", type=int, default=0, help="0 = all")
    ap.add_argument("--output", default="outputs/round168/template-audit.json")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import run_phase0 as p0
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    spec = real_spec()

    items = [
        json.loads(line)
        for line in Path(args.qa_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.max_items:
        items = items[: args.max_items]
    if not items:
        raise SystemExit(f"no items in {args.qa_file}")
    log(f"{len(items)} items from {args.qa_file}")

    conventions: dict[str, dict[str, list]] = {
        "raw": {"gen": [], "content": [], "tasks": []},
        "chat": {"gen": [], "content": [], "tasks": []},
    }
    # Suffix variants actually used: the completion form sends BoolQ through a
    # different template, the native chat form uses one.
    template_variants = {"raw": 2, "chat": 1}
    for item in items:
        task = str(item.get("task", "unknown"))
        # The content position is the question ALONE.  Tokenizing it with the
        # template appended (the first draft of this script did) measures the
        # template again and reports a counterfactual that is not one.
        q_ids = tokenizer.encode(str(item["question"]), add_special_tokens=False)
        for name, chat in (("raw", False), ("chat", True)):
            ids = p0._qa_prompt_ids(tokenizer, item, PROMPT, BOOLQ_PROMPT, chat_template=chat)
            if len(ids) < 3:
                continue
            conventions[name]["gen"].append(_last_row(spec, ids))
            # The content position is the same for both conventions; it is the
            # last token of the question as the *completion* form renders it.
            conventions[name]["content"].append(_last_row(spec, q_ids))
            conventions[name]["tasks"].append(task)

    report: dict[str, Any] = {
        "qa_file": args.qa_file,
        "model": args.model,
        "n_items": len(items),
        "spec_total_rows": int(spec.total),
        "window_tokens": 3,
        "conventions": {},
        "assumptions": [
            (
                "A row id is a function of at most three tokens, so a template "
                "that appends a FIXED suffix of >= 3 tokens forces the same row "
                "for every item in a task.  Only tokenizer merging across the "
                "question/suffix boundary can break that, and the distinct-tuple "
                "count measures whether it did."
            ),
            (
                "The 'content' position is the last token of the question as the "
                "completion form renders it, tokenized with the same templates. "
                "It is the item-specific addressing the arms did NOT use."
            ),
            (
                "This is a statement about ADDRESSING only.  It says a "
                "real/control comparison at the generation position cannot "
                "distinguish memory content; it does not by itself say the memory "
                "is useless at other positions."
            ),
        ],
    }
    for name, payload in conventions.items():
        if not payload["gen"]:
            continue
        gen_rows = np.asarray(payload["gen"], dtype=np.int64)
        content_rows = np.asarray(payload["content"], dtype=np.int64)
        gen = address_diversity(gen_rows)
        content = address_diversity(content_rows)
        gen_tasks = per_task_diversity(gen_rows, payload["tasks"])
        content_tasks = per_task_diversity(content_rows, payload["tasks"])
        verdict = verdict_from_tasks(
            gen_tasks, content_tasks,
            n_template_variants=template_variants.get(name),
        )
        report["conventions"][name] = {
            "n": gen.n,
            "generation": gen.to_dict(),
            "content": content.to_dict(),
            "verdict": verdict,
            "per_task_generation": {k: v.to_dict() for k, v in gen_tasks.items()},
            "per_task_content": {k: v.to_dict() for k, v in content_tasks.items()},
        }
        log(f"{name}: per-task generation modal share "
            f"{verdict['worst_task_modal_share']:.3f} (worst {verdict['worst_task']}), "
            f"{gen.n_distinct} distinct rows / {gen.n} items | content modal share "
            f"max {max(d.modal_share for d in content_tasks.values()):.3f} -> "
            f"{verdict['label']}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
