#!/usr/bin/env python3
"""Multi-source ablation for the 0.8B PLE/RAG/post-trained system.

Evaluated source combinations:

* base
* +RAG (BM25 retrieval on knowledge tasks)
* +PLE (calibrated/task-conditioned n-gram logit fusion)
* +MoRA (CAP-1 MoRA adapter)
* +RAG+MoRA
* +PLE+MoRA
* +all (RAG + PLE + MoRA)

The script uses a seeded item sampler so the same evaluation can be repeated as
a 3-seed ablation.  For PLE it builds an addressable n-gram memory from the
specified PLE corpus and applies it during teacher-forced answer log-probability
evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from qwen35_ple.addressable_memory import AddressableNgramMemory
from qwen35_ple.rag import BM25Index
from qwen35_ple.router import (
    CalibratedNgramLogitProcessor,
    TaskConditionedNgramLogitProcessor,
    load_fusion_router_config,
)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _load_model(model_path: str, adapter: str | None, device: str):
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tokenizer, model


def _load_items(args: argparse.Namespace) -> list[dict]:
    qa_path = Path(args.qa_file)
    data = json.loads(qa_path.read_text(encoding="utf-8"))
    items = data.get("items") if isinstance(data, dict) else data
    rng = random.Random(args.seed)
    if args.limit_kn is not None and args.limit_kn < len(items):
        items = rng.sample(items, args.limit_kn)

    out: list[dict] = []
    for i, x in enumerate(items):
        out.append(
            {
                "id": f"kb-{i}",
                "task": "knowledge",
                "question": str(x.get("question", "")),
                "answer": str(x.get("answer", "")),
                "is_rare": bool(x.get("is_rare", False)),
            }
        )

    # Arithmetic
    for i in range(args.n_arith):
        a = int(rng.randint(2, 99))
        b = int(rng.randint(2, 99))
        op = rng.choice([" + ", " - ", " * "])
        expr = f"{a}{op}{b}"
        ans = eval(expr, {"__builtins__": {}}, {})
        out.append(
            {
                "id": f"arith-{i}",
                "task": "arithmetic",
                "question": f"What is {expr}?",
                "answer": str(ans),
                "is_rare": False,
            }
        )

    # Code-output
    templates = [
        ("What does `1 + 2 * 3` evaluate to?", "7"),
        ("What does `len([1, 2, 3])` evaluate to?", "3"),
        ("What does `2 ** 3` evaluate to?", "8"),
        ("What does `10 // 3` evaluate to?", "3"),
        ("What does `10 % 3` evaluate to?", "1"),
        ("What does `1 + 2 == 3` evaluate to?", "True"),
        ("What does `3 > 2` evaluate to?", "True"),
        ("What does `'abc'[1]` evaluate to?", "b"),
        ("What does `[1, 2, 3][-1]` evaluate to?", "3"),
        ("What does `max(3, 7, 2)` evaluate to?", "7"),
    ]
    for i in range(args.n_code):
        q, ans = templates[i % len(templates)]
        out.append(
            {
                "id": f"code-{i}",
                "task": "code-output",
                "question": q,
                "answer": ans,
                "is_rare": False,
            }
        )
    return out


def _load_corpus(path: str, max_docs: int | None) -> list[str]:
    path = Path(path)
    docs: list[str] = []
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = str(obj.get("text") or obj.get("solution") or "")
                if text.strip():
                    docs.append(text.strip())
                if max_docs is not None and len(docs) >= max_docs:
                    break
    else:
        docs = [
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        ][:max_docs]
    return docs


def _load_ngram_texts(path: str, max_docs: int | None) -> list[str]:
    path = Path(path)
    texts: list[str] = []
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                problem = str(obj.get("problem") or "")
                solution = str(obj.get("solution") or "")
                text = (problem + "\n" + solution).strip()
                if text:
                    texts.append(text)
                if max_docs is not None and len(texts) >= max_docs:
                    break
    else:
        texts = [
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        ][:max_docs]
    return texts


def _build_ple_memory(tokenizer, corpora: list[str], max_docs_per_file: int) -> AddressableNgramMemory:
    mem = AddressableNgramMemory(min_order=2, max_order=4)
    value_id = 0
    for corpus in corpora:
        texts = _load_ngram_texts(corpus, max_docs_per_file)
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if ids:
                mem.add_document(ids, value_id=value_id)
                value_id += 1
    return mem


def _task_for_item(task: str) -> str:
    return {
        "knowledge": "semantic",
        "arithmetic": "number",
        "code-output": "code",
    }.get(task, "general")


def _answer_logprob(
    model,
    tokenizer,
    full_ids: list[int],
    answer_start: int,
    processor,
    device: str,
) -> tuple[float, bool]:
    ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
        logits = out.logits[0]
    total = 0.0
    n = 0
    first_hit = False
    first_t = max(0, answer_start - 1)
    for t in range(first_t, len(full_ids) - 1):
        cur = logits[t]
        if processor is not None:
            cur = processor(cur, full_ids[:t])
        log_probs = F.log_softmax(cur.float(), dim=-1)
        total += float(log_probs[int(full_ids[t + 1])])
        n += 1
        if t == first_t:
            first_hit = int(torch.argmax(cur.float())) == int(full_ids[t + 1])
    return total / max(1, n), first_hit


def _greedy_generate(
    model,
    tokenizer,
    input_ids: list[int],
    processor,
    *,
    max_new_tokens: int,
    device: str,
) -> str:
    generated = list(input_ids)
    for _ in range(max_new_tokens):
        ids = torch.tensor([generated], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=ids, use_cache=False).logits[0, -1]
            if processor is not None:
                logits = processor(logits, generated)
        nxt = int(torch.argmax(logits))
        if nxt == tokenizer.eos_token_id:
            break
        generated.append(nxt)
    return tokenizer.decode(generated[len(input_ids):], skip_special_tokens=True).strip()


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower().strip()).strip()


def _exact(pred: str, gold: str) -> bool:
    return _normalize(pred) == _normalize(gold)


def _make_processor(
    memory,
    config,
    tokenizer,
    ple_mode: str,
) -> CalibratedNgramLogitProcessor | TaskConditionedNgramLogitProcessor:
    cfg = load_fusion_router_config(config)
    fusion = cfg.get("fusion", {})
    if ple_mode == "plain":
        return CalibratedNgramLogitProcessor(
            memory,
            scale=fusion.get("scale", 1.0),
            bias=fusion.get("bias", 0.0),
            temperature=fusion.get("temperature", 1.0),
        )
    return TaskConditionedNgramLogitProcessor(
        memory,
        scale=fusion.get("scale", 1.0),
        bias=fusion.get("bias", 0.0),
        temperature=fusion.get("temperature", 1.0),
        semantic_tasks=["semantic", "knowledge", "qa"],
        ple_tasks=["code", "name", "number", "low_entropy"],
        default_task="general",
    )


def evaluate_combos(
    model,
    tokenizer,
    items,
    bm25,
    memory,
    config,
    *,
    top_k: int,
    ple_mode: str,
    device: str,
    greedy: bool,
    max_new_tokens: int,
    with_adapter_combos: bool = False,
):
    processor_cache: dict[str, object] = {}

    def get_processor(mode: str):
        key = mode
        if key not in processor_cache:
            processor_cache[key] = _make_processor(memory, config, tokenizer, mode)
        return processor_cache[key]

    combos = [
        ("base", False, False, None),
        ("rag", True, False, None),
        ("ple", False, True, None),
    ]
    if with_adapter_combos:
        combos = [
            ("mora", False, False, True),
            ("rag_mora", True, False, True),
            ("ple_mora", False, True, True),
            ("all", True, True, True),
        ]

    results: dict[str, dict] = {}
    for name, use_rag, use_ple, _ in combos:
        combo_results = []
        t0 = time.time()
        for item in items:
            question = item["question"]
            answer = item["answer"]
            qids = tokenizer.encode(question, add_special_tokens=False)
            ans_ids = tokenizer.encode(answer, add_special_tokens=False)
            if not qids or not ans_ids:
                continue

            context_ids: list[int] = []
            if use_rag and item["task"] == "knowledge" and bm25 is not None:
                hits = bm25.search(question, top_k=top_k)
                context = "\n\n".join(bm25.docs[i] for i in hits)
                if context:
                    context_ids = tokenizer.encode(context, add_special_tokens=False)

            full_ids = context_ids + qids + ans_ids
            answer_start = len(context_ids) + len(qids)

            processor = None
            if use_ple:
                processor = get_processor(ple_mode)
                if hasattr(processor, "set_task"):
                    processor.set_task(_task_for_item(item["task"]))

            lp, first_hit = _answer_logprob(
                model, tokenizer, full_ids, answer_start, processor, device
            )
            entry = {
                "id": item["id"],
                "task": item["task"],
                "answer_logprob": lp,
                "first_hit": first_hit,
            }
            if greedy:
                prompt_ids = context_ids + qids
                pred = _greedy_generate(
                    model,
                    tokenizer,
                    prompt_ids,
                    processor,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
                entry["greedy"] = pred
                entry["exact"] = _exact(pred, answer)
            combo_results.append(entry)

        summary = {}
        for task in ["knowledge", "arithmetic", "code-output"]:
            group = [r for r in combo_results if r["task"] == task]
            if not group:
                continue
            row = {
                "n": len(group),
                "answer_logprob": float(np.mean([r["answer_logprob"] for r in group])),
                "first_hit": float(np.mean([1.0 if r["first_hit"] else 0.0 for r in group])),
            }
            if greedy:
                row["exact"] = float(np.mean([1.0 if r["exact"] else 0.0 for r in group]))
            summary[task] = row

        results[name] = {
            "summary": summary,
            "runtime_seconds": time.time() - t0,
        }
        if greedy:
            results[name]["per_item"] = combo_results
        print(
            f"[ms-ab] combo={name} items={len(combo_results)} "
            f"time={results[name]['runtime_seconds']:.1f}s",
            flush=True,
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--adapter", default=None, help="MoRA/LoRA/QLoRA PEFT adapter")
    parser.add_argument("--qa-file", default="data/rare-kb-v1.json")
    parser.add_argument("--corpus", default="data/sources/wikitext.jsonl")
    parser.add_argument(
        "--ple-corpus",
        default="data/cap1-rag-distill-160.jsonl",
        help="comma-separated corpora used to build PLE n-gram memory",
    )
    parser.add_argument("--ple-config", default="configs/ngram-fusion-router.json")
    parser.add_argument("--ple-mode", choices=["task", "plain"], default="task")
    parser.add_argument("--limit-kn", type=int, default=30)
    parser.add_argument("--n-arith", type=int, default=10)
    parser.add_argument("--n-code", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-docs", type=int, default=20000)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--output", default="outputs/ms-ablation-seed0.json")
    args = parser.parse_args()

    t0 = time.time()
    tokenizer, model = _load_model(args.model, None, args.device)
    items = _load_items(args)
    print(f"[ms-ab] items={len(items)}", flush=True)

    bm25 = None
    if args.corpus:
        docs = _load_corpus(args.corpus, args.max_docs)
        bm25 = BM25Index(docs)
        print(f"[ms-ab] bm25 docs={len(docs)}", flush=True)

    ple_corpora = [x.strip() for x in args.ple_corpus.split(",") if x.strip()]
    memory = _build_ple_memory(tokenizer, ple_corpora, args.max_docs)
    print(f"[ms-ab] ple memory ngrams={memory.stats()}", flush=True)

    base_results = evaluate_combos(
        model,
        tokenizer,
        items,
        bm25,
        memory,
        args.ple_config,
        top_k=args.top_k,
        ple_mode=args.ple_mode,
        device=args.device,
        greedy=args.greedy,
        max_new_tokens=args.max_new_tokens,
        with_adapter_combos=False,
    )
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    adapter_results = {}
    if args.adapter:
        _, adapter_model = _load_model(args.model, args.adapter, args.device)
        adapter_results = evaluate_combos(
            adapter_model,
            tokenizer,
            items,
            bm25,
            memory,
            args.ple_config,
            top_k=args.top_k,
            ple_mode=args.ple_mode,
            device=args.device,
            greedy=args.greedy,
            max_new_tokens=args.max_new_tokens,
            with_adapter_combos=True,
        )
        # The adapter-model call only has adapter-bearing combos; merge.
        merged = {**base_results, **adapter_results}
    else:
        merged = base_results

    # Add per-combo adapter flag for readability.
    for name in merged:
        merged[name]["adapter"] = bool(args.adapter) and name not in {"base", "rag", "ple"}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "multisource-ablation-v1",
        "seed": args.seed,
        "config": vars(args),
        "combos": merged,
        "runtime_seconds": time.time() - t0,
    }
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v["summary"] for k, v in merged.items()}, indent=2, ensure_ascii=False))
    print(f"[ms-ab] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
