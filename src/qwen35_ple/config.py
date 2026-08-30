"""Configuration loading and validation for qwen35-ple experiments.

This module is intentionally small: it parses the YAML samples in ``configs/``
and performs contract-level checks before handing the values to engram-peft.
Actual model construction remains in engram-peft; this repository only
orchestrates and validates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment should install pyyaml
    raise RuntimeError("qwen35-ple config loading requires PyYAML") from exc


ALLOWED_ENGINES = {"deepseek", "qwen_ple"}
ALLOWED_TABLE_SPECS = {"PLE_QWEN_V1", "ENG_DEEPSEEK_V1"}
ALLOWED_TABLE_SOURCES = {"memory", "engramdb:store", "engramdb:view"}
ALLOWED_TRAIN_MODES = {"engram_only", "preserve_trainable", "full_finetune"}


@dataclass
class EngineConfig:
    engine: str = "deepseek"
    table_spec: str | None = None
    table_source: str = "memory"
    view_path: str | None = None
    keys_path: str | None = None
    store_path: str | None = None
    model_dir: str | None = None
    shards: int | None = None
    rows_per_shard: int | None = None
    width: int | None = None
    scale: float | None = None
    cache_size: int = 4096

    def validate(self) -> None:
        if self.engine not in ALLOWED_ENGINES:
            raise ValueError(f"unsupported engine: {self.engine}")
        if self.table_spec is not None and self.table_spec not in ALLOWED_TABLE_SPECS:
            raise ValueError(f"unsupported table_spec: {self.table_spec}")
        if self.table_source not in ALLOWED_TABLE_SOURCES:
            raise ValueError(f"unsupported table_source: {self.table_source}")
        if self.table_source == "engramdb:view" and not (self.view_path or self.keys_path):
            raise ValueError("engramdb:view requires view_path and/or keys_path")
        if self.table_source == "engramdb:store" and not self.store_path:
            raise ValueError("engramdb:store requires store_path")


@dataclass
class EngramConfig:
    ngram_sizes: list[int] = field(default_factory=lambda: [2, 3])
    n_head_per_ngram: int = 8
    embedding_dim: int = 2560
    engram_vocab_size_per_ngram: list[int] = field(
        default_factory=lambda: [160_000_000, 160_000_000]
    )
    target_layers: list[int] = field(default_factory=lambda: [1])
    hc_mult: int = 1
    conv_kernel_size: int = 4
    conv_dilation: int = 3
    engram_dtype: str = "bfloat16"
    prime_sizes: list[int] | None = None
    use_sparse_embeddings: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.ngram_sizes != [2, 3]:
            raise ValueError("current PLE_QWEN_V1 orchestration expects ngram_sizes=[2,3]")
        if self.n_head_per_ngram != 8:
            raise ValueError("PLE_QWEN_V1 expects n_head_per_ngram=8")
        if self.prime_sizes is not None:
            if len(self.prime_sizes) != 16:
                raise ValueError("prime_sizes must contain 16 head prime sizes")
        elif self.embedding_dim != 2560:
            raise ValueError("PLE_QWEN_V1 e_t width is 16*160=2560")
        if not self.target_layers:
            raise ValueError("target_layers must not be empty")
        if self.hc_mult != 1:
            raise ValueError("Qwen3.5 PLE-lite variant currently expects hc_mult=1")


@dataclass
class TrainingConfig:
    train_mode: str = "engram_only"
    backbone_freeze_steps: int = 0
    learning_rate_multiplier: float = 5.0
    seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.train_mode not in ALLOWED_TRAIN_MODES:
            raise ValueError(f"unsupported train_mode: {self.train_mode}")
        if self.backbone_freeze_steps < 0:
            raise ValueError("backbone_freeze_steps must be >= 0")
        if self.learning_rate_multiplier <= 0:
            raise ValueError("learning_rate_multiplier must be > 0")


@dataclass
class Qwen35PleConfig:
    project: str
    base_model: str
    tokenizer: str
    engine: EngineConfig
    engram: EngramConfig
    training: TrainingConfig
    raw: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.project:
            raise ValueError("project must not be empty")
        if not self.base_model or not self.tokenizer:
            raise ValueError("base_model/tokenizer must not be empty")
        self.engine.validate()
        self.engram.validate()
        self.training.validate()

    def to_engram_config(
        self,
        *,
        hidden_size: int | None = None,
        compressed_vocab_size: int | None = None,
        pad_id: int | None = None,
        tokenizer_name_or_path: str | None = None,
        **overrides: Any,
    ):
        """Build an engram-peft ``EngramConfig`` from this YAML/contract config.

        This is the "configuration-only" bridge: after loading a qwen35-ple
        YAML, callers can hand the result directly to ``get_engram_model`` and
        ``table_source`` will be consumed automatically by engram-peft.
        """
        from engram_peft.config import EngramConfig as PeftEngramConfig

        data: dict[str, Any] = {
            "hidden_size": hidden_size,
            "compressed_vocab_size": compressed_vocab_size,
            "pad_id": pad_id,
            "tokenizer_name_or_path": tokenizer_name_or_path,
            "ngram_sizes": list(self.engram.ngram_sizes),
            "n_head_per_ngram": self.engram.n_head_per_ngram,
            "embedding_dim": self.engram.embedding_dim,
            "engram_vocab_size_per_ngram": list(
                self.engram.engram_vocab_size_per_ngram
            ),
            "target_layers": list(self.engram.target_layers),
            "hc_mult": self.engram.hc_mult,
            "conv_kernel_size": self.engram.conv_kernel_size,
            "conv_dilation": self.engram.conv_dilation,
            "engram_dtype": self.engram.engram_dtype,
            "prime_sizes": self.engram.prime_sizes,
            "use_sparse_embeddings": self.engram.use_sparse_embeddings,
            "engine": self.engine.engine,
            "table_spec": self.engine.table_spec,
            "table_source": self.engine.table_source,
            "table_store_path": self.engine.store_path,
            "table_model_dir": self.engine.model_dir,
            "table_shards": self.engine.shards,
            "table_rows_per_shard": self.engine.rows_per_shard,
            "table_width": self.engine.width,
            "table_scale": self.engine.scale,
            "table_cache_size": self.engine.cache_size,
        }
        data.update(self.engram.extra)
        data.update(overrides)
        return PeftEngramConfig(**{k: v for k, v in data.items() if v is not None})


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"config section '{key}' must be a mapping")
    return value


def load_config(path: str | Path) -> Qwen35PleConfig:
    """Load and validate a qwen35-ple YAML configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError("config root must be a mapping")

    engine_raw = _section(data, "engine")
    engram_raw = _section(data, "engram")
    training_raw = _section(data, "training")

    cfg = Qwen35PleConfig(
        project=str(data.get("project", "")),
        base_model=str(data.get("base_model", "")),
        tokenizer=str(data.get("tokenizer", "")),
        engine=EngineConfig(
            engine=str(engine_raw.get("engine", "deepseek")),
            table_spec=engine_raw.get("table_spec"),
            table_source=str(engine_raw.get("table_source", "memory")),
            view_path=engine_raw.get("view_path"),
            keys_path=engine_raw.get("keys_path"),
            store_path=engine_raw.get("store_path"),
            model_dir=engine_raw.get("model_dir"),
            shards=(
                int(engine_raw["shards"])
                if engine_raw.get("shards") is not None
                else None
            ),
            rows_per_shard=(
                int(engine_raw["rows_per_shard"])
                if engine_raw.get("rows_per_shard") is not None
                else None
            ),
            width=(
                int(engine_raw["width"])
                if engine_raw.get("width") is not None
                else None
            ),
            scale=(
                float(engine_raw["scale"])
                if engine_raw.get("scale") is not None
                else None
            ),
            cache_size=int(engine_raw.get("cache_size", 4096)),
        ),
        engram=EngramConfig(
            ngram_sizes=list(engram_raw.get("ngram_sizes", [2, 3])),
            n_head_per_ngram=int(engram_raw.get("n_head_per_ngram", 8)),
            embedding_dim=int(engram_raw.get("embedding_dim", 2560)),
            engram_vocab_size_per_ngram=list(
                engram_raw.get("engram_vocab_size_per_ngram", [160_000_000, 160_000_000])
            ),
            target_layers=list(engram_raw.get("target_layers", [1])),
            hc_mult=int(engram_raw.get("hc_mult", 1)),
            conv_kernel_size=int(engram_raw.get("conv_kernel_size", 4)),
            conv_dilation=int(engram_raw.get("conv_dilation", 3)),
            engram_dtype=str(engram_raw.get("engram_dtype", "bfloat16")),
            prime_sizes=(
                [int(x) for x in engram_raw["prime_sizes"]]
                if engram_raw.get("prime_sizes") is not None
                else None
            ),
            use_sparse_embeddings=bool(
                engram_raw.get("use_sparse_embeddings", True)
            ),
            extra={k: v for k, v in engram_raw.items() if k not in {
                "ngram_sizes", "n_head_per_ngram", "embedding_dim",
                "engram_vocab_size_per_ngram", "target_layers", "hc_mult",
                "conv_kernel_size", "conv_dilation", "engram_dtype",
                "prime_sizes", "use_sparse_embeddings",
            }},
        ),
        training=TrainingConfig(
            train_mode=str(training_raw.get("train_mode", "engram_only")),
            backbone_freeze_steps=int(training_raw.get("backbone_freeze_steps", 0)),
            learning_rate_multiplier=float(
                training_raw.get("learning_rate_multiplier", 5.0)
            ),
            seed=int(training_raw.get("seed", 0)),
            extra={k: v for k, v in training_raw.items() if k not in {
                "train_mode", "backbone_freeze_steps",
                "learning_rate_multiplier", "seed",
            }},
        ),
        raw=data,
    )
    cfg.validate()
    return cfg
