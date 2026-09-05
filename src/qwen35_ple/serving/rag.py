"""RAG serving adapter for the 0.8B product path.

This adapter wraps:

* a frozen Transformers model;
* a :class:`~qwen35_ple.rag.HybridRetriever` (BM25 + dense/rerank);
* prompt formatting and stopping controls.

It is engine-agnostic: vLLM/SGLang/CompileForge can call :meth:`answer` or use
the same retriever object directly.
"""

from __future__ import annotations

from typing import Any

import torch

from qwen35_ple.rag import Chunk, HybridRetriever, build_rag_prompt
from qwen35_ple.router import (
    build_task_conditioned_processor,
    build_task_router_from_config,
)


class RAGServingAdapter:
    """A small serving-facing wrapper for hybrid RAG + causal LM generation."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        retriever: HybridRetriever,
        *,
        max_new_tokens: int = 64,
        top_k: int = 3,
        candidate_pool: int = 50,
        concise: bool = True,
        device: str = "cpu",
        stop_sequences: list[str] | None = None,
        logit_processor=None,
        ngram_memory=None,
        fusion_config=None,
        task_router=None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.max_new_tokens = int(max_new_tokens)
        self.top_k = int(top_k)
        self.candidate_pool = int(candidate_pool)
        self.concise = bool(concise)
        self.device = device
        self.stop_sequences = stop_sequences or []
        if task_router is None and fusion_config is not None:
            task_router = build_task_router_from_config(fusion_config)
        self.task_router = task_router
        if logit_processor is None and ngram_memory is not None and fusion_config is not None:
            logit_processor = build_task_conditioned_processor(
                ngram_memory,
                fusion_config,
                tokenizer=tokenizer,
            )
        self.logit_processor = logit_processor

    def retrieve(self, question: str) -> list[Chunk | str]:
        return self.retriever.retrieve(
            question,
            top_k=self.top_k,
            candidate_pool=self.candidate_pool,
        )

    def build_prompt(self, context: str, question: str) -> str:
        prompt = build_rag_prompt(context, question)
        if self.concise:
            prompt = prompt.replace(
                "Answer:", "Answer briefly:"
            )
        return prompt

    def answer(self, question: str) -> dict[str, Any]:
        route_info: dict[str, Any] | None = None
        task: str | None = None
        if self.task_router is not None:
            if hasattr(self.task_router, "route"):
                route_info = self.task_router.route(question)
                task = str(route_info.get("task", "")) or None
            elif callable(self.task_router):
                task = str(self.task_router(question))
            elif hasattr(self.task_router, "classify"):
                task = str(self.task_router.classify(question))

        if self.logit_processor is not None and hasattr(self.logit_processor, "set_task"):
            if task is None and hasattr(self.logit_processor, "classifier"):
                task = self.logit_processor.classifier.classify(question)
            self.logit_processor.set_task(task)

        if route_info is not None and hasattr(self.retriever, "set_channel_weights"):
            self.retriever.set_channel_weights(
                **route_info.get("channel_weights", {})
            )

        chunks = self.retrieve(question)
        context = "\n\n".join(str(c) for c in chunks)
        prompt = self.build_prompt(context, question)
        answer_text = self._generate(prompt)
        result = {
            "question": question,
            "contexts": [str(c) for c in chunks],
            "prompt": prompt,
            "answer": answer_text,
        }
        result_task = task
        if result_task is None and getattr(self.logit_processor, "task", None) is not None:
            result_task = self.logit_processor.task
        if result_task is not None:
            result["task"] = result_task
        if getattr(self.logit_processor, "last_gate", None) is not None:
            gate = self.logit_processor.last_gate
            result["ple_gate"] = {
                "active": bool(gate.get("active", False)),
                "task": gate.get("task", result_task),
                "expected_log_density_ratio": gate.get("expected_log_density_ratio"),
                "mode": gate.get("mode"),
            }
        return result

    def _generate(self, prompt: str) -> str:
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        generated = list(ids)
        for _ in range(self.max_new_tokens):
            input_ids = torch.tensor([generated], dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits = self.model(input_ids=input_ids, use_cache=False).logits[0, -1]
                if self.logit_processor is not None:
                    logits = self.logit_processor(logits, generated)
            nxt = int(torch.argmax(logits))
            if nxt == self.tokenizer.eos_token_id:
                break
            generated.append(nxt)
            if self.stop_sequences:
                text_so_far = self.tokenizer.decode(
                    generated[len(ids):], skip_special_tokens=True
                )
                if any(stop in text_so_far for stop in self.stop_sequences):
                    break
        return self.tokenizer.decode(
            generated[len(ids):], skip_special_tokens=True
        ).strip()


__all__ = ["RAGServingAdapter"]
