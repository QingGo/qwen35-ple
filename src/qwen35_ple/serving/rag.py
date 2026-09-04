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
        chunks = self.retrieve(question)
        context = "\n\n".join(str(c) for c in chunks)
        prompt = self.build_prompt(context, question)
        answer_text = self._generate(prompt)
        return {
            "question": question,
            "contexts": [str(c) for c in chunks],
            "prompt": prompt,
            "answer": answer_text,
        }

    def _generate(self, prompt: str) -> str:
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        generated = list(ids)
        for _ in range(self.max_new_tokens):
            input_ids = torch.tensor([generated], dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits = self.model(input_ids=input_ids, use_cache=False).logits[0, -1]
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
