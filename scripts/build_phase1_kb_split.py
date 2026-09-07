#!/usr/bin/env python3
"""Build Phase 1 knowledge-base/QA splits and a two-way contamination audit.

The split protocol follows ``docs/round-132-eval-contamination-principle.md``:

* KB.train: source documents used to build/read the external memory bank.
* KB.eval:  held-out source documents that are allowed in the memory *only*
  for an unseen-KB memory-reading experiment; they are never used to train the
  reader.
* QA.train: question/answer pairs used only to teach the reader prompt/format.
* QA.eval:  final held-out question/answer pairs used for reporting.

The script also emits a lightweight bidirectional contamination report:
``eval QA -> KB.train`` and ``eval QA -> KB.eval``, plus a reverse
``KB.eval -> QA.train`` check.

Usage::

    python scripts/build_phase1_kb_split.py \
      --source data/sources/wikitext.jsonl \
      --qa data/qa-expanded-150.json \
      --output-dir data/phase1/kb-wiki \
      --train-frac 0.8 --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")


def _normalize(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


def _doc_id(doc: dict | str) -> str:
    if isinstance(doc, str):
        text = doc
    else:
        text = str(doc.get("text") or doc.get("content") or doc)
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _doc_text(doc: dict | str) -> str:
    if isinstance(doc, str):
        return doc
    return str(doc.get("text") or doc.get("content") or doc)


def _load_qa(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]
    raise SystemExit("QA file must be a JSON list or {items: [...]}")


def _load_docs(path: Path, limit: int | None) -> list[dict | str]:
    docs: list[dict | str] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                valid = isinstance(obj, str) or (
                    isinstance(obj, dict) and (obj.get("text") or obj.get("content"))
                )
                if valid:
                    docs.append(obj)
                if limit is not None and len(docs) >= limit:
                    break
    else:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line:
                docs.append(line)
            if limit is not None and len(docs) >= limit:
                break
    return docs


def _qa_in_docs(qa: list[dict], docs: list[dict | str]) -> dict[str, int]:
    corpus_norm = "\n".join(_normalize(_doc_text(d)) for d in docs)
    n_answer = 0
    n_question = 0
    n_qa = 0
    for item in qa:
        q = _normalize(str(item.get("question", "")))
        a = _normalize(str(item.get("answer", "")))
        qa_norm = _normalize(
            str(item.get("question", "")) + " " + str(item.get("answer", ""))
        )
        if a and a in corpus_norm:
            n_answer += 1
        if q and q in corpus_norm:
            n_question += 1
        if qa_norm and qa_norm in corpus_norm:
            n_qa += 1
    return {
        "qa": len(qa),
        "answer_in_docs": n_answer,
        "question_in_docs": n_question,
        "qa_pair_in_docs": n_qa,
    }


def _write_jsonl(path: Path, rows: list[dict | str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            if isinstance(row, str):
                f.write(json.dumps({"text": row}, ensure_ascii=False) + "\n")
            else:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="KB source JSONL or text file")
    parser.add_argument("--qa", default="data/qa-expanded-150.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--qa-train-frac", type=float, default=0.5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    qa = _load_qa(Path(args.qa))
    docs = _load_docs(Path(args.source), args.limit_docs)
    if not docs:
        raise SystemExit("no source docs loaded")
    if not qa:
        raise SystemExit("no QA items loaded")

    # Deterministic doc split by content hash.
    train: list[dict | str] = []
    eval: list[dict | str] = []
    for doc in docs:
        bucket = int(_doc_id(doc), 16) % 1000
        if bucket < int(args.train_frac * 1000):
            train.append(doc)
        else:
            eval.append(doc)

    # Deterministic QA train/eval split.
    # Use a stable, seed-dependent hash so the split is reproducible but not
    # simply the document hash.
    qa_train: list[dict] = []
    qa_eval: list[dict] = []
    for item in qa:
        key = f"{args.seed}|{item.get('question','')}|{item.get('answer','')}"
        bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % 1000
        if bucket < int(args.qa_train_frac * 1000):
            qa_train.append(item)
        else:
            qa_eval.append(item)

    paths = {
        "kb.train": out / "kb.train.jsonl",
        "kb.eval": out / "kb.eval.jsonl",
        "qa.train": out / "qa.train.jsonl",
        "qa.eval": out / "qa.eval.jsonl",
    }
    if not args.force:
        for p in paths.values():
            if p.exists():
                raise SystemExit(f"output already exists: {p}; use --force to overwrite")

    _write_jsonl(paths["kb.train"], train)
    _write_jsonl(paths["kb.eval"], eval)
    _write_jsonl(paths["qa.train"], qa_train)
    _write_jsonl(paths["qa.eval"], qa_eval)

    audit = {
        "source": str(Path(args.source).resolve()),
        "qa": str(Path(args.qa).resolve()),
        "train_frac": args.train_frac,
        "qa_train_frac": args.qa_train_frac,
        "seed": args.seed,
        "docs_total": len(docs),
        "docs_train": len(train),
        "docs_eval": len(eval),
        "qa_total": len(qa),
        "qa_train": len(qa_train),
        "qa_eval": len(qa_eval),
        "eval_qa_in_kb_train": _qa_in_docs(qa_eval, train),
        "eval_qa_in_kb_eval": _qa_in_docs(qa_eval, eval),
        "train_qa_in_kb_eval": _qa_in_docs(qa_train, eval),
        "outputs": {k: str(v.resolve()) for k, v in paths.items()},
    }
    manifest = out / "manifest.json"
    manifest.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({k: v for k, v in audit.items() if k != "outputs"}, indent=2))
    print(f"[kb-split] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
