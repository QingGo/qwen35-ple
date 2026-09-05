#!/usr/bin/env python3
"""Small LoRA teacher-text distillation / SFT runner for the 0.8B model.

This is the simplest feasible teacher-distillation route with current resources:

* The "teacher" is offline text data (e.g. an existing CoT/solution corpus);
* The student is Qwen3.5-0.8B;
* We fine-tune a LoRA adapter to imitate the teacher-style answers.

For true logit-level distillation from a live teacher model, this script can be
extended later; here we establish the first trainable student path.

Usage::

    python scripts/run_lora_distill.py \
        --model data/models/Qwen3.5-0.8B \
        --data data/sources/distilled_corpus_400k_with_cot-filtered.jsonl \
        --steps 50 \
        --output outputs/lora-distill
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch


def _load_model(model_path: str, device: str, use_qlora: bool = False):
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if use_qlora:
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            quantization_config=bnb,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, dtype=torch.float32
        )
        model.to(device)
    model.config.use_cache = False
    model.train()
    return tokenizer, model


def _load_examples(path: str, limit: int | None) -> list[str]:
    path = Path(path)
    examples: list[str] = []
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
                text = _format_example(obj)
                if text:
                    examples.append(text)
                if limit is not None and len(examples) >= limit:
                    break
    elif path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data.get("items") if isinstance(data, dict) else data
        for obj in items:
            text = _format_example(obj)
            if text:
                examples.append(text)
            if limit is not None and len(examples) >= limit:
                break
    else:
        examples = [
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        ][:limit]
    return examples


def _format_example(obj: dict) -> str:
    if "instruction" in obj and "output" in obj:
        return f"Instruction: {obj['instruction']}\n\nResponse: {obj['output']}"
    if "problem" in obj:
        answer = obj.get("solution") or obj.get("answer") or ""
        context = obj.get("context")
        if context:
            return (
                f"Question: {obj['problem']}\n\n"
                f"Context:\n{context}\n\n"
                f"Answer: {answer}"
            )
        return f"Question: {obj['problem']}\n\nAnswer: {answer}"
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/Qwen3.5-0.8B")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="outputs/lora-distill")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-qlora", action="store_true", help="load base in 4-bit NF4 with bitsandbytes")
    parser.add_argument("--use-mora", action="store_true", help="use MoRA from peft-mora fork")
    parser.add_argument("--mora-type", type=int, default=1, help="MoRA type: 1/2/3/4/6 from peft-mora")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    t0 = time.time()

    tokenizer, model = _load_model(args.model, args.device, use_qlora=args.use_qlora)
    examples = _load_examples(args.data, args.limit)
    if not examples:
        print("[lora-distill] no examples loaded", flush=True)
        return 1
    print(f"[lora-distill] examples={len(examples)}", flush=True)

    from peft import LoraConfig, get_peft_model

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_kwargs = {
        "r": args.r,
        "lora_alpha": args.lora_alpha,
        "target_modules": target_modules,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    if args.use_mora:
        lora_kwargs.update(use_mora=True, mora_type=args.mora_type)
    lora_config = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora_config)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(
        f"[lora-distill] trainable params={sum(p.numel() for p in trainable)}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    losses: list[float] = []
    for step in range(args.steps):
        text = rng.choice(examples)
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        ).to(args.device)
        optimizer.zero_grad()
        out = model(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
            labels=enc["input_ids"],
        )
        loss = out.loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if (step + 1) % 5 == 0 or step == 0:
            print(
                f"  step {step + 1}/{args.steps}: loss={loss.item():.4f}",
                flush=True,
            )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    meta = {
        "config": vars(args),
        "examples": len(examples),
        "losses": losses,
        "final_loss": losses[-1] if losses else None,
        "runtime_seconds": time.time() - t0,
        "output": str(out_dir),
    }
    (out_dir / "distill-meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[lora-distill] saved adapter to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
