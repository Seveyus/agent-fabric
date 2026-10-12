"""
Experiment configuration: TOML file -> validated dataclasses.

Only the standard library is used (tomllib, dataclasses). Unknown keys are an
error, so a typo in a config cannot silently fall back to a default.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from core.dataset.temporal_graph import PRMergeTaskConfig


class ConfigError(ValueError):
    pass


@dataclass
class ExperimentSection:
    name: str
    seed: int = 0
    output_dir: str = "runs"


@dataclass
class SyntheticSection:
    n_users: int = 2000
    n_repos: int = 200
    days: int = 21
    seed: int = 0


@dataclass
class DataSection:
    source: str = "synthetic"            # "synthetic" | "gharchive"
    events: list[str] = field(default_factory=list)
    cache: str = ""                      # path of a cached TemporalTask (.pt); built if missing
    synthetic: SyntheticSection = field(default_factory=SyntheticSection)


@dataclass
class FeatureSection:
    use_history: bool = True             # include point-in-time history aggregates on PR nodes


@dataclass
class ModelSection:
    encoder: str = "hgt"                 # "mlp" | "sage" | "hgt"
    hidden: int = 64
    num_layers: int = 2
    heads: int = 2
    dropout: float = 0.1


@dataclass
class SamplerSection:
    batch_size: int = 512
    default_fanout: list[int] = field(default_factory=lambda: [8, 4])
    fanout: dict[str, list[int]] = field(default_factory=dict)   # "src/rel/dst" -> [hop1, hop2]
    temporal: bool = True


@dataclass
class TrainSection:
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 3
    max_batches_per_epoch: int = 0       # 0 = full epoch
    grad_clip: float = 1.0
    label_fraction: float = 1.0          # deterministic subsample of training seeds (label-efficiency runs)


@dataclass
class BackendSection:
    name: str = "pyg"                    # "pyg" | "cugraph"
    device: str = "cpu"
    num_workers: int = 0
    torch_threads: int = 0               # 0 = PyTorch default


@dataclass
class PretrainSection:
    checkpoint: str = ""                 # JEPA checkpoint whose online encoder initialises the model
    freeze_encoder: bool = False         # linear probe when True
    freeze_random_encoder: bool = False  # control: linear probe on a randomly initialised encoder


@dataclass
class JepaSection:
    """Temporal JEPA pretraining (training/pretrain_jepa.py)."""

    delta_hours: list[float] = field(default_factory=lambda: [12.0, 24.0, 48.0])
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    ema_tau: float = 0.996
    ema_tau_end: float = 1.0
    predictor_hidden: int = 128
    max_anchors: int = 0                 # 0 = all eligible PRs
    output: str = ""                     # checkpoint path; default <output_dir>/<name>/jepa.pt


@dataclass
class ExperimentConfig:
    experiment: ExperimentSection
    data: DataSection
    task: PRMergeTaskConfig
    features: FeatureSection = field(default_factory=FeatureSection)
    model: ModelSection = field(default_factory=ModelSection)
    sampler: SamplerSection = field(default_factory=SamplerSection)
    train: TrainSection = field(default_factory=TrainSection)
    backend: BackendSection = field(default_factory=BackendSection)
    pretrain: PretrainSection = field(default_factory=PretrainSection)
    jepa: JepaSection = field(default_factory=JepaSection)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.data.source not in ("synthetic", "gharchive"):
            raise ConfigError("data.source must be 'synthetic' or 'gharchive'")
        if self.data.source == "gharchive" and not self.data.events and not self.data.cache:
            raise ConfigError("data.events (or an existing data.cache) is required for gharchive")
        if self.model.encoder not in ("mlp", "sage", "hgt"):
            raise ConfigError("model.encoder must be one of mlp, sage, hgt")
        if self.model.hidden <= 0 or self.model.num_layers <= 0:
            raise ConfigError("model.hidden and model.num_layers must be > 0")
        if self.model.encoder == "hgt" and self.model.hidden % self.model.heads:
            raise ConfigError("model.hidden must be divisible by model.heads for HGT")
        if self.model.encoder != "mlp" and len(self.sampler.default_fanout) != self.model.num_layers:
            raise ConfigError("sampler.default_fanout needs one entry per message-passing layer")
        for k, v in self.sampler.fanout.items():
            if len(k.split("/")) != 3:
                raise ConfigError(f"sampler.fanout key {k!r} must be 'src/rel/dst'")
            if len(v) != len(self.sampler.default_fanout):
                raise ConfigError(f"sampler.fanout[{k!r}] must have {len(self.sampler.default_fanout)} hops")
        if self.sampler.batch_size <= 0:
            raise ConfigError("sampler.batch_size must be > 0")
        if not 0.0 <= self.model.dropout < 1.0:
            raise ConfigError("model.dropout must be in [0, 1)")
        if self.train.epochs <= 0 or self.train.lr <= 0:
            raise ConfigError("train.epochs and train.lr must be > 0")
        if not 0.0 < self.train.label_fraction <= 1.0:
            raise ConfigError("train.label_fraction must be in (0, 1]")
        j = self.jepa
        if not j.delta_hours or min(j.delta_hours) <= 0:
            raise ConfigError("jepa.delta_hours must be a non-empty list of positive values")
        if max(j.delta_hours) > self.task.horizon_hours:
            raise ConfigError("jepa.delta_hours must not exceed task.horizon_hours (embargo before validation)")
        if not 0.0 < j.ema_tau <= j.ema_tau_end <= 1.0:
            raise ConfigError("need 0 < jepa.ema_tau <= jepa.ema_tau_end <= 1")
        if self.backend.name not in ("pyg", "cugraph"):
            raise ConfigError("backend.name must be 'pyg' or 'cugraph'")
        if self.backend.name == "cugraph" and not self.backend.device.startswith("cuda"):
            raise ConfigError("the cugraph backend requires backend.device = 'cuda'")
        if self.pretrain.freeze_encoder and not self.pretrain.checkpoint:
            raise ConfigError("pretrain.freeze_encoder requires pretrain.checkpoint")
        if self.pretrain.freeze_random_encoder and self.pretrain.checkpoint:
            raise ConfigError("pretrain.freeze_random_encoder is a control without checkpoint")
        try:
            self.task.validate()
        except ValueError as exc:
            raise ConfigError(f"task: {exc}") from exc


def _build(cls, raw: dict[str, Any], where: str):
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ConfigError(f"unknown keys in [{where}]: {sorted(unknown)}")
    kwargs = {}
    for name, value in raw.items():
        sub = {"synthetic": SyntheticSection}.get(name) if cls is DataSection else None
        kwargs[name] = _build(sub, value, f"{where}.{name}") if sub else value
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"[{where}]: {exc}") from exc


_SECTIONS = {
    "experiment": ExperimentSection, "data": DataSection, "task": PRMergeTaskConfig,
    "features": FeatureSection, "model": ModelSection, "sampler": SamplerSection,
    "train": TrainSection, "backend": BackendSection, "pretrain": PretrainSection,
    "jepa": JepaSection,
}


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ConfigError(f"unknown sections: {sorted(unknown)}")
    for required in ("experiment", "data", "task"):
        if required not in raw:
            raise ConfigError(f"missing required section [{required}]")
    cfg = ExperimentConfig(**{k: _build(_SECTIONS[k], v, k) for k, v in raw.items()})
    cfg.validate()
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load a TOML config. ``overrides`` are 'section.key=value' strings (value parsed as TOML)."""
    raw = tomllib.loads(Path(path).read_text())
    for ov in overrides or []:
        key, _, value = ov.partition("=")
        if not value:
            raise ConfigError(f"override {ov!r} must look like section.key=value")
        *path_keys, leaf = key.strip().split(".")
        node = raw
        for k in path_keys:
            node = node.setdefault(k, {})
        try:
            node[leaf] = tomllib.loads(f"v = {value}")["v"]
        except tomllib.TOMLDecodeError:
            node[leaf] = value  # bare string
    return config_from_dict(raw)
