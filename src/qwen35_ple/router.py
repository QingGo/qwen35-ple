"""PLE-2 serving router: calibrated n-gram logit processor and task gates.

This module connects the calibrated n-gram fusion to actual generation.  A
``CalibratedNgramLogitProcessor`` is a drop-in logit processor for
``RAGServingAdapter``: it looks up the current n-gram distribution from the
addressable memory and applies the calibrated ``scale / bias / temperature``
transformation to the base model logits.

For the PLE-2c task-boundary work this module additionally provides:

* :class:`TaskClassifier` - lightweight, auditable query-task routing;
* :class:`LogDensityRatioGate` - a log-density-ratio gate using the same
  discriminant from round 89 (``E[log(p_m/p_b)] > 0``);
* :class:`TaskConditionedNgramLogitProcessor` - the production processor that
  combines task routing, the density gate, and calibrated fusion;
* JSON config persistence for calibrated fusion parameters.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from qwen35_ple.fusion import fuse_ngram_logits

_SCHEMA = "ngram-fusion-router-v1"

_DEFAULT_CODE_KEYWORDS = (
    "def ",
    "class ",
    "import ",
    "return ",
    "function",
    "lambda",
    "code",
    "program",
    "script",
    "snippet",
    "write ",
    "implement",
    "debug",
    "compile",
    "select ",
    "where ",
    "if ",
    "for ",
    "while ",
    "=>",
    "{",
    "}",
    ";",
    "//",
    "#include",
)

_DEFAULT_SEMANTIC_KEYWORDS = (
    "what",
    "who",
    "whom",
    "why",
    "how",
    "explain",
    "describe",
    "define",
    "definition",
    "meaning",
    "capital",
    "country",
    "history",
    "science",
    "knowledge",
    "answer",
    "which",
    "where",
)

_DEFAULT_NUMBER_KEYWORDS = (
    "number",
    "count",
    "sum",
    "calculate",
    "compute",
    "math",
    "arithmetic",
    "digit",
    "numeric",
    "integer",
    "prime",
    "fibonacci",
)

_DEFAULT_NAME_KEYWORDS = (
    "name",
    "called",
    "author",
    "person",
    "entity",
    "company",
    "organization",
    "city",
    "river",
)


def _to_numpy_logits(logits: Any) -> np.ndarray:
    """Convert torch/numpy logits to a float32 numpy array."""
    if hasattr(logits, "detach") and hasattr(logits, "to"):
        return logits.detach().cpu().float().numpy()
    return np.asarray(logits, dtype=np.float32)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    m = float(np.max(logits))
    z = logits - m
    return z - math.log(float(np.exp(z).sum()))


def log_density_ratio(
    base_logits: Any,
    ngram_probs: dict[int, float] | None,
) -> float:
    """Estimate ``E_{p_m}[log(p_m/p_b)]``.

    This is the round-89 discriminant: when the n-gram memory assigns more
    probability mass to tokens the base model considers unlikely, this value is
    large and positive.  It is also the KL divergence from the base to the
    memory distribution restricted to the memory support.

    Returns ``-inf`` when there is no memory distribution.
    """
    if not ngram_probs:
        return -math.inf
    logits = _to_numpy_logits(base_logits)
    log_pb = _log_softmax(logits)
    total = 0.0
    for tok, p in ngram_probs.items():
        if 0 <= tok < len(log_pb) and p > 0:
            total += float(p) * (math.log(float(p)) - float(log_pb[tok]))
    return float(total)


def pseudo_label_log_density_ratio(
    base_logits: Any,
    ngram_probs: dict[int, float] | None,
) -> float:
    """Log-density ratio at the base model's current argmax.

    This is a conservative runtime proxy for the true-label ratio: if the
    memory also favors the token the base already favors, the fusion is likely
    safe; if memory strongly disagrees on a semantic task, the gate can stop it.
    """
    if not ngram_probs:
        return -math.inf
    logits = _to_numpy_logits(base_logits)
    log_pb = _log_softmax(logits)
    base_argmax = int(np.argmax(logits))
    p_m = float(ngram_probs.get(base_argmax, 0.0))
    if p_m <= 0:
        return -math.inf
    return math.log(p_m) - float(log_pb[base_argmax])


class TaskClassifier:
    """Small, auditable query-task router.

    The classifier is intentionally rule-based so every routing decision can be
    traced back to an explicit keyword/pattern.  It is also straightforward to
    replace with a learned classifier later without changing callers.
    """

    def __init__(
        self,
        *,
        code_keywords: tuple[str, ...] | list[str] | None = None,
        semantic_keywords: tuple[str, ...] | list[str] | None = None,
        number_keywords: tuple[str, ...] | list[str] | None = None,
        name_keywords: tuple[str, ...] | list[str] | None = None,
        default_task: str = "general",
    ) -> None:
        self.code_keywords = tuple(code_keywords or _DEFAULT_CODE_KEYWORDS)
        self.semantic_keywords = tuple(semantic_keywords or _DEFAULT_SEMANTIC_KEYWORDS)
        self.number_keywords = tuple(number_keywords or _DEFAULT_NUMBER_KEYWORDS)
        self.name_keywords = tuple(name_keywords or _DEFAULT_NAME_KEYWORDS)
        self.default_task = default_task

    def classify(self, text: str) -> str:
        t = str(text).lower()
        if any(k in t for k in self.code_keywords):
            return "code"
        if any(k in t for k in self.semantic_keywords):
            return "semantic"
        if any(k in t for k in self.number_keywords):
            return "number"
        if any(k in t for k in self.name_keywords):
            return "name"
        # Very digit-heavy text is almost always a numeric/local task.
        digits = sum(ch.isdigit() for ch in t)
        if t and digits / max(1, len(t)) > 0.15:
            return "number"
        return self.default_task

    def state_dict(self) -> dict[str, Any]:
        return {
            "code_keywords": list(self.code_keywords),
            "semantic_keywords": list(self.semantic_keywords),
            "number_keywords": list(self.number_keywords),
            "name_keywords": list(self.name_keywords),
            "default_task": self.default_task,
        }


class TaskRouter:
    """Route a query to a task profile plus per-channel retrieval weights.

    This is the auditable query-level router used by
    :class:`~qwen35_ple.serving.rag.RAGServingAdapter`.  The channel weights
    are rule-based defaults and can be persisted in the fusion-router JSON so
    product deployments can inspect and tune every routing decision.
    """

    DEFAULT_CHANNEL_WEIGHTS: dict[str, dict[str, float]] = {
        "semantic": {"bm25_weight": 1.0, "dense_weight": 2.0, "ngram_weight": 0.0},
        "code": {"bm25_weight": 1.0, "dense_weight": 0.5, "ngram_weight": 2.0},
        "name": {"bm25_weight": 1.0, "dense_weight": 0.75, "ngram_weight": 1.5},
        "number": {"bm25_weight": 1.0, "dense_weight": 0.75, "ngram_weight": 1.5},
        "low_entropy": {"bm25_weight": 1.0, "dense_weight": 0.5, "ngram_weight": 2.0},
        "general": {"bm25_weight": 1.0, "dense_weight": 1.0, "ngram_weight": 1.0},
    }

    def __init__(
        self,
        *,
        classifier: TaskClassifier | None = None,
        channel_weights: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.classifier = classifier or TaskClassifier()
        self.channel_weights: dict[str, dict[str, float]] = {
            task: dict(weights)
            for task, weights in self.DEFAULT_CHANNEL_WEIGHTS.items()
        }
        if channel_weights:
            for task, weights in channel_weights.items():
                self.channel_weights[task] = {
                    **self.channel_weights.get(task, {}),
                    **dict(weights),
                }

    def classify(self, text: str) -> str:
        return self.classifier.classify(text)

    def route(self, text: str) -> dict[str, Any]:
        task = self.classify(text)
        weights = dict(
            self.channel_weights.get(
                task,
                self.channel_weights.get("general", {}),
            )
        )
        return {"task": task, "channel_weights": weights}

    def state_dict(self) -> dict[str, Any]:
        return {
            "channel_weights": {
                task: dict(weights) for task, weights in self.channel_weights.items()
            },
            "classifier": self.classifier.state_dict(),
        }


class LogDensityRatioGate:
    """Gate that decides when the calibrated n-gram fusion should be active.

    Supported modes:

    - ``expected_kl``: use ``E_{p_m}[log(p_m/p_b)] >= threshold`` (round 89).
    - ``pseudo_label``: use the log-density ratio at the base argmax.
    - ``memory_top1``: use the log-density advantage of the memory argmax.
    - ``hybrid``: require a strong expected-KL signal *and* agreement with the
      current base best token.
    """

    def __init__(
        self,
        *,
        mode: str = "expected_kl",
        threshold: float = 0.0,
    ) -> None:
        if mode not in {"expected_kl", "pseudo_label", "memory_top1", "hybrid"}:
            raise ValueError(f"unknown log-density gate mode: {mode}")
        self.mode = mode
        self.threshold = float(threshold)

    def evaluate(
        self,
        base_logits: Any,
        ngram_probs: dict[int, float] | None,
    ) -> dict[str, Any]:
        if not ngram_probs:
            return {
                "active": False,
                "mode": self.mode,
                "threshold": self.threshold,
                "expected_log_density_ratio": -math.inf,
                "pseudo_label_log_density_ratio": -math.inf,
                "memory_top1_log_density_ratio": -math.inf,
                "base_entropy": math.inf,
                "memory_entropy": math.inf,
                "base_top1_prob": 0.0,
                "memory_top1_prob": 0.0,
            }

        logits = _to_numpy_logits(base_logits)
        log_pb = _log_softmax(logits)
        pb = np.exp(log_pb)
        # Normalize the empirical memory distribution if needed.
        pm = {int(k): float(v) for k, v in ngram_probs.items() if int(k) >= 0}
        total = sum(pm.values())
        if total <= 0:
            pm = {}
        else:
            pm = {k: v / total for k, v in pm.items()}

        expected = log_density_ratio(logits, pm)
        pseudo = pseudo_label_log_density_ratio(logits, pm)
        mem_argmax = max(pm, key=pm.get) if pm else 0
        mem_top1 = (
            math.log(pm[mem_argmax]) - float(log_pb[mem_argmax])
            if pm and 0 <= mem_argmax < len(log_pb)
            else -math.inf
        )

        # Entropies are calculated over the full base distribution; for memory
        # they are over the sparse support (which is the distribution used).
        base_entropy = float(-np.sum(pb * np.log(np.maximum(pb, 1e-12))))
        memory_entropy = float(-sum(p * math.log(p) for p in pm.values() if p > 0))

        if self.mode == "expected_kl":
            active = expected >= self.threshold
        elif self.mode == "pseudo_label":
            active = pseudo >= self.threshold
        elif self.mode == "memory_top1":
            active = mem_top1 >= self.threshold
        else:  # hybrid
            active = (
                expected >= self.threshold
                and pseudo >= 0.0
            )

        return {
            "active": active,
            "mode": self.mode,
            "threshold": self.threshold,
            "expected_log_density_ratio": expected,
            "pseudo_label_log_density_ratio": pseudo,
            "memory_top1_log_density_ratio": mem_top1,
            "base_entropy": base_entropy,
            "memory_entropy": memory_entropy,
            "base_top1_prob": float(np.max(pb)) if len(pb) else 0.0,
            "memory_top1_prob": float(max(pm.values())) if pm else 0.0,
        }

    def decide(
        self,
        base_logits: Any,
        ngram_probs: dict[int, float] | None,
    ) -> bool:
        return bool(self.evaluate(base_logits, ngram_probs)["active"])

    def state_dict(self) -> dict[str, float | str]:
        return {"mode": self.mode, "threshold": self.threshold}


class CalibratedNgramLogitProcessor:
    """Apply a calibrated n-gram log-prior during generation.

    Parameters
    ----------
    memory
        Any object with ``continuation_distribution(context)`` returning either
        ``(dict, order)`` or ``None`` (for example ``AddressableNgramMemory``).
    scale, bias, temperature
        Calibrated fusion hyperparameters.
    """

    def __init__(
        self,
        memory,
        *,
        scale: float = 1.0,
        bias: float = 0.0,
        temperature: float = 1.0,
        enabled: bool = True,
    ) -> None:
        self.memory = memory
        self.scale = float(scale)
        self.bias = float(bias)
        self.temperature = float(temperature)
        self.enabled = bool(enabled)

    def __call__(self, logits: Any, context_ids: Any) -> Any:
        """Return fused logits (same type as input)."""
        if not self.enabled:
            return logits

        # Accept torch tensor or numpy array.
        is_torch = hasattr(logits, "detach") and hasattr(logits, "to")
        if is_torch:
            import torch

            logits_np = logits.detach().cpu().float().numpy()
            context = (
                context_ids.detach().cpu().tolist()
                if hasattr(context_ids, "detach")
                else list(context_ids)
            )
        else:
            logits_np = np.asarray(logits, dtype=np.float32)
            context = list(context_ids)

        result = self.memory.continuation_distribution(context)
        if result is None:
            dist = None
        else:
            dist = result[0]

        fused_np = fuse_ngram_logits(
            logits_np,
            dist,
            scale=self.scale,
            bias=self.bias,
            temperature=self.temperature,
        )

        if is_torch:
            return torch.as_tensor(fused_np, dtype=logits.dtype, device=logits.device)
        return fused_np

    def state_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "bias": self.bias,
            "temperature": self.temperature,
            "enabled": self.enabled,
        }


class TaskConditionedNgramLogitProcessor(CalibratedNgramLogitProcessor):
    """Calibrated n-gram fusion guarded by task routing and density ratio.

    The processor is a drop-in replacement for
    :class:`CalibratedNgramLogitProcessor`.  It can be told the current query
    task with :meth:`set_task` (typically from ``RAGServingAdapter.answer``),
    or it can classify on the fly if a ``context_decoder`` is provided.

    For semantic/knowledge tasks the processor disables PLE entirely.  For
    code/name/number/local tasks it additionally requires the density gate to
    pass, preventing noisy or over-confident n-gram priors from firing on
    generic contexts.
    """

    def __init__(
        self,
        memory,
        *,
        scale: float = 1.0,
        bias: float = 0.0,
        temperature: float = 1.0,
        enabled: bool = True,
        classifier: TaskClassifier | None = None,
        density_gate: LogDensityRatioGate | None = None,
        task: str | None = None,
        semantic_tasks: tuple[str, ...] | list[str] | None = None,
        ple_tasks: tuple[str, ...] | list[str] | None = None,
        default_task: str = "general",
        task_scale: dict[str, float] | None = None,
        context_decoder: Callable[[Any], str] | None = None,
    ) -> None:
        super().__init__(
            memory,
            scale=scale,
            bias=bias,
            temperature=temperature,
            enabled=enabled,
        )
        self.classifier = classifier or TaskClassifier(default_task=default_task)
        self.density_gate = density_gate or LogDensityRatioGate()
        self.task = task
        self.semantic_tasks = set(semantic_tasks or {"semantic", "knowledge", "qa"})
        self.ple_tasks = set(ple_tasks or {"code", "name", "number", "low_entropy"})
        self.default_task = default_task
        self.task_scale = dict(task_scale or {})
        self.context_decoder = context_decoder
        self.last_task: str | None = None
        self.last_gate: dict[str, Any] | None = None

    def set_task(self, task: str | None) -> None:
        self.task = task

    def _active_task(self, context_ids: Any) -> str:
        if self.task is not None:
            return self.task
        if self.context_decoder is not None:
            return self.classifier.classify(self.context_decoder(context_ids))
        return self.default_task

    def __call__(self, logits: Any, context_ids: Any) -> Any:
        if not self.enabled:
            return logits

        is_torch = hasattr(logits, "detach") and hasattr(logits, "to")
        if is_torch:
            import torch

            logits_np = logits.detach().cpu().float().numpy()
            context = (
                context_ids.detach().cpu().tolist()
                if hasattr(context_ids, "detach")
                else list(context_ids)
            )
        else:
            logits_np = np.asarray(logits, dtype=np.float32)
            context = list(context_ids)

        result = self.memory.continuation_distribution(context)
        if result is None:
            dist = None
        else:
            dist = result[0]
        if not dist:
            return logits

        task = self._active_task(context_ids)
        self.last_task = task
        if task in self.semantic_tasks:
            self.last_gate = {
                "active": False,
                "reason": "semantic_task",
                "task": task,
            }
            return logits

        gate = self.density_gate.evaluate(logits_np, dist)
        self.last_gate = gate
        if not gate["active"]:
            return logits

        factor = self.task_scale.get(task, 1.0)
        fused_np = fuse_ngram_logits(
            logits_np,
            dist,
            scale=self.scale * factor,
            bias=self.bias,
            temperature=self.temperature,
        )

        if is_torch:
            return torch.as_tensor(fused_np, dtype=logits.dtype, device=logits.device)
        return fused_np

    def state_dict(self) -> dict[str, Any]:
        out = super().state_dict()
        out.update(
            {
                "task": self.task,
                "default_task": self.default_task,
                "semantic_tasks": sorted(self.semantic_tasks),
                "ple_tasks": sorted(self.ple_tasks),
                "task_scale": dict(self.task_scale),
                "classifier": self.classifier.state_dict(),
                "density_gate": self.density_gate.state_dict(),
                "last_task": self.last_task,
            }
        )
        return out


DEFAULT_ROUTER_CONFIG: dict[str, Any] = {
    "schema": _SCHEMA,
    "fusion": {
        "scale": 1.0,
        "bias": -1.0,
        "temperature": 0.5,
        "enabled": True,
    },
    "router": {
        "mode": "expected_kl",
        "min_log_density_ratio": 0.5,
        "average_log_density_ratio": 0.0,
        "semantic_tasks": ["semantic", "knowledge", "qa"],
        "ple_tasks": ["code", "name", "number", "low_entropy"],
        "default_task": "general",
        "task_scale": {
            "code": 1.0,
            "name": 1.0,
            "number": 1.0,
            "low_entropy": 1.0,
            "general": 0.5,
        },
        "classifier": {
            "code_keywords": list(_DEFAULT_CODE_KEYWORDS),
            "semantic_keywords": list(_DEFAULT_SEMANTIC_KEYWORDS),
            "number_keywords": list(_DEFAULT_NUMBER_KEYWORDS),
            "name_keywords": list(_DEFAULT_NAME_KEYWORDS),
            "default_task": "general",
        },
        "channel_weights": {
            "semantic": {"bm25_weight": 1.0, "dense_weight": 2.0, "ngram_weight": 0.0},
            "code": {"bm25_weight": 1.0, "dense_weight": 0.5, "ngram_weight": 2.0},
            "name": {"bm25_weight": 1.0, "dense_weight": 0.75, "ngram_weight": 1.5},
            "number": {"bm25_weight": 1.0, "dense_weight": 0.75, "ngram_weight": 1.5},
            "low_entropy": {"bm25_weight": 1.0, "dense_weight": 0.5, "ngram_weight": 2.0},
            "general": {"bm25_weight": 1.0, "dense_weight": 1.0, "ngram_weight": 1.0},
        },
    },
}


def save_fusion_router_config(
    path: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save/update a fusion-router JSON config."""
    cfg = {
        "schema": _SCHEMA,
        "fusion": dict(DEFAULT_ROUTER_CONFIG["fusion"]),
        "router": dict(DEFAULT_ROUTER_CONFIG["router"]),
    }
    if config:
        cfg["schema"] = str(config.get("schema", _SCHEMA))
        cfg["fusion"].update(config.get("fusion", {}))
        cfg["router"].update(config.get("router", {}))
        # Deep-merge nested classifier/task_scale/channel_weights dicts.
        for key in ("task_scale", "classifier", "channel_weights"):
            if key in config.get("router", {}):
                cfg["router"][key] = {
                    **cfg["router"].get(key, {}),
                    **config["router"][key],
                }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return cfg


def load_fusion_router_config(path: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load and validate a fusion-router JSON config.

    A plain dict is returned (copied) so unit tests can avoid files.
    """
    if isinstance(path, dict):
        cfg = dict(path)
    else:
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("fusion router config must be a JSON object")
    if "fusion" not in cfg or "router" not in cfg:
        raise ValueError("fusion router config requires 'fusion' and 'router' sections")
    cfg.setdefault("schema", _SCHEMA)
    return cfg


def build_task_router_from_config(
    config: str | Path | dict[str, Any] | None = None,
) -> TaskRouter:
    """Build a query router from a persisted fusion-router config."""
    cfg = load_fusion_router_config(config) if config is not None else DEFAULT_ROUTER_CONFIG
    router = cfg.get("router", {})
    classifier_cfg = router.get("classifier", {})
    classifier = TaskClassifier(
        code_keywords=classifier_cfg.get("code_keywords"),
        semantic_keywords=classifier_cfg.get("semantic_keywords"),
        number_keywords=classifier_cfg.get("number_keywords"),
        name_keywords=classifier_cfg.get("name_keywords"),
        default_task=router.get("default_task", "general"),
    )
    return TaskRouter(
        classifier=classifier,
        channel_weights=router.get("channel_weights"),
    )


def build_task_conditioned_processor(
    memory,
    config: str | Path | dict[str, Any] | None = None,
    *,
    tokenizer=None,
) -> TaskConditionedNgramLogitProcessor:
    """Build a production processor from a persisted fusion-router config."""
    cfg = load_fusion_router_config(config) if config is not None else DEFAULT_ROUTER_CONFIG
    fusion = cfg.get("fusion", {})
    router = cfg.get("router", {})
    classifier_cfg = router.get("classifier", {})
    classifier = TaskClassifier(
        code_keywords=classifier_cfg.get("code_keywords"),
        semantic_keywords=classifier_cfg.get("semantic_keywords"),
        number_keywords=classifier_cfg.get("number_keywords"),
        name_keywords=classifier_cfg.get("name_keywords"),
        default_task=router.get("default_task", "general"),
    )
    gate = LogDensityRatioGate(
        mode=router.get("mode", "expected_kl"),
        threshold=float(router.get("min_log_density_ratio", 0.0)),
    )
    return TaskConditionedNgramLogitProcessor(
        memory,
        scale=fusion.get("scale", 1.0),
        bias=fusion.get("bias", 0.0),
        temperature=fusion.get("temperature", 1.0),
        enabled=fusion.get("enabled", True),
        classifier=classifier,
        density_gate=gate,
        semantic_tasks=router.get("semantic_tasks", ["semantic", "knowledge", "qa"]),
        ple_tasks=router.get("ple_tasks", ["code", "name", "number", "low_entropy"]),
        default_task=router.get("default_task", "general"),
        task_scale=router.get("task_scale", {}),
        context_decoder=None if tokenizer is None else lambda ids: tokenizer.decode(ids),
    )


__all__ = [
    "CalibratedNgramLogitProcessor",
    "DEFAULT_ROUTER_CONFIG",
    "LogDensityRatioGate",
    "TaskClassifier",
    "TaskConditionedNgramLogitProcessor",
    "TaskRouter",
    "build_task_conditioned_processor",
    "build_task_router_from_config",
    "load_fusion_router_config",
    "log_density_ratio",
    "pseudo_label_log_density_ratio",
    "save_fusion_router_config",
]
