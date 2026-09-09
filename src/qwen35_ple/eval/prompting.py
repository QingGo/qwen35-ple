"""Prompt formatting helpers for Phase 0/2 QA evaluation.

The original Phase 0 harness used the raw question as a completion prompt.  A
base model then often continues with reasoning or an option list instead of a
short answer.  The diagnostic run uses an instruction-style template while
keeping the old behavior as the default for backward compatibility.
"""

from __future__ import annotations

DEFAULT_QA_PROMPT = "Question: {question}\nAnswer:"
DEFAULT_BOOLQ_PROMPT = "Question: {question}\nAnswer with one word, Yes or No:"


def format_qa_prompt(
    question: str,
    *,
    task: str | None = None,
    template: str | None = None,
    boolq_template: str | None = None,
) -> str:
    """Format a QA prompt.

    ``template`` applies to every task.  ``boolq_template`` overrides it for
    BoolQ.  An empty/``None`` template keeps the raw question (legacy Phase 0
    behavior).
    """
    chosen: str | None = None
    if task == "boolq" and boolq_template:
        chosen = boolq_template
    elif template:
        chosen = template
    if not chosen:
        return question
    if "{question}" not in chosen:
        raise ValueError("QA prompt template must contain {question!r}")
    return chosen.format(question=question)
