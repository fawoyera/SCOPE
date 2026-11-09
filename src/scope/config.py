"""Typed, layered experiment configuration.

A run's configuration is assembled by deep-merging YAML layers in a fixed order::

    configs/base.yaml
      -> configs/domains/<domain>.yaml
      -> configs/methods/<method>.yaml
      -> configs/models/<model>.yaml             (minus its ``method_overrides`` block)
      -> configs/models/<model>.yaml :: method_overrides[<method>]
      -> --set key.sub=value                     (command-line overrides)

The merged dict is converted into nested dataclasses.  Unknown keys raise an
error, so a typo in a YAML file fails loudly instead of being silently ignored.
"""
from __future__ import annotations

import copy
import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_args, get_origin, get_type_hints

import yaml

# Repository root: <root>/src/scope/config.py -> <root>
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs"


# ════════════════════════════════════════════════════════════════════════════
# Sections
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class RunConfig:
    domain: str = "atc"
    model: str = "llama31_8b"
    method: str = "scope"
    seed: int = 42
    output_root: str = "results"
    run_dir: str = ""              # filled in by `resolve_run_dir`
    init_from: str = ""            # checkpoint to initialise from ("" = model.hf_id)
    depot_path: str = ""           # optional persistent mirror (e.g. /depot/...)
    resume_from: str = ""          # step checkpoint to resume training from


@dataclass
class ModelConfig:
    key: str = ""
    hf_id: str = ""
    family: str = "generic"        # llama | qwen | gpt2 | gemma | mistral | generic
    prompt_format: str = "alpaca"  # alpaca | chat
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)
    merge_system_into_user: bool = False   # for chat templates without a system role
    dtype: str = "bfloat16"
    attn_implementation: str = "auto"      # auto | flash_attention_2 | sdpa | eager
    gradient_checkpointing: bool = False
    max_length: int = 1024
    trust_remote_code: bool = False
    device_map: str = "auto"               # auto (pipeline over all GPUs) | single


@dataclass
class DataConfig:
    path: str = "data/atc_pairs_enriched.jsonl"
    split_field: str = "split"
    train_split: str = "train"
    val_split: str = "valid"
    test_split: str = "test"
    drop_unintelligible: bool = True
    curriculum: str = "B"                  # A | B | C | none  (see data/datasets.py)
    tier_weights: Dict[str, float] = field(
        default_factory=lambda: {"gold": 3.0, "silver": 1.5, "bronze": 1.0})
    phase1_frac: float = 0.5               # strategy A only
    prompt_style: str = "rules"            # minimal | task | rules | fewshot  (prompt explicitness)
    fewshot_k: int = 3                     # demonstrations for prompt_style = fewshot


@dataclass
class ArtefactConfig:
    vocab: str = "artefacts/atc/vocab.json"
    phrases: str = "artefacts/atc/ngrams.json"
    grammar: str = "artefacts/atc/grammar.lark"
    system_prompt: str = "atc"             # key in data/prompts.py::SYSTEM_PROMPTS


@dataclass
class TrainConfig:
    enabled: bool = True                   # False for inference-only methods
    trainer: str = "objective"             # objective (SFT/SCOPE) | dpo
    epochs: int = 3
    batch_size: int = 2
    grad_accum: int = 8
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    optimizer: str = "auto"                # auto | adamw | adamw8bit
    early_stop_patience: int = 0           # 0 = disabled
    checkpoint_metric: str = "c_bar"
    hallucination_threshold: float = 0.10
    checkpoint_every_steps: int = 0        # crash-safe step checkpoints (0 = off)
    val_max_pairs: int = 200
    log_every: int = 10
    num_workers: int = 0
    keep_checkpoint: bool = True           # False: delete best ckpt after test eval


@dataclass
class GRPOConfig:
    num_samples: int = 4                   # M, group size
    temperature: float = 1.5
    top_p: float = 0.95
    max_new_tokens: int = 80
    every_n_steps: int = 2                 # run the RL branch every k micro-steps
    token_reduction: str = "mean"          # mean | sum   (log-prob over response tokens)
    length_penalty: bool = True            # damp rewards of degenerate short samples
    partial_credit: bool = True            # C_cfg reward with prefix partial credit (False: binary parse)
    kl_beta: float = 0.0                   # >0 loads a frozen reference model


@dataclass
class ObjectiveConfig:
    lambda_ce: float = 1.0
    lambda_tok: float = 0.0
    lambda_phr: float = 0.0
    lambda_cfg: float = 0.0
    loss_curriculum: bool = False
    curriculum_phase1: float = 1 / 3
    curriculum_phase2: float = 2 / 3
    curriculum_ramp_steps: int = 50
    gradnorm: bool = False
    gradnorm_alpha: float = 1.5
    gradnorm_every: int = 20
    gradnorm_lr: float = 1e-3
    grpo: GRPOConfig = field(default_factory=GRPOConfig)


@dataclass
class DPOConfig:
    beta: float = 0.1
    samples_per_prompt: int = 4
    sample_temperature: float = 1.0
    sample_top_p: float = 0.95
    max_prompts: int = 0                   # 0 = every training prompt
    gen_batch_size: int = 8


@dataclass
class GCDConfig:
    enabled: bool = False
    level: str = "bigram"                  # vocab | bigram (GCD†)  |  cfg (full G, Geng et al. 2023)
    successor_fallback: bool = True        # GCD†: if a word has no bigram successor, allow all of V
    num_beams: int = 1                     # beam search width under the constraint (Geng et al. use 1–10)


@dataclass
class BERTScoreConfig:
    enabled: bool = True
    domain_model: str = "models/bert-atc-mlm"
    general_model: str = "bert-base-uncased"
    num_layers: int = 9
    device: str = "cpu"
    batch_size: int = 64


@dataclass
class EvalConfig:
    batch_size: int = 8
    max_new_tokens: int = 80
    repetition_penalty: float = 1.3
    test_max_pairs: int = 0                # 0 = full test split
    grammar_workers: int = 4
    grammar_timeout: float = 3.0
    bertscore: BERTScoreConfig = field(default_factory=BERTScoreConfig)


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "scope"
    entity: str = ""
    mode: str = "online"                   # online | offline | disabled
    tags: List[str] = field(default_factory=list)


@dataclass
class LoggingConfig:
    tensorboard: bool = True
    csv: bool = True
    wandb: WandbConfig = field(default_factory=WandbConfig)
    log_examples: int = 20                 # sample generations logged per evaluation


@dataclass
class ExperimentConfig:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    artefacts: ArtefactConfig = field(default_factory=ArtefactConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    gcd: GCDConfig = field(default_factory=GCDConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # ── convenience ─────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: Union[str, Path]) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


# ════════════════════════════════════════════════════════════════════════════
# Dict <-> dataclass
# ════════════════════════════════════════════════════════════════════════════
def _from_dict(cls, data: Dict[str, Any], path: str = ""):
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"Unknown config key(s) under '{path or '<root>'}': {sorted(unknown)}")
    kwargs = {}
    for k, v in data.items():
        t = hints[k]
        if dataclasses.is_dataclass(t) and isinstance(v, dict):
            kwargs[k] = _from_dict(t, v, f"{path}.{k}".strip("."))
        else:
            kwargs[k] = _coerce(v, t)
    return cls(**kwargs)


def _coerce(v, t):
    """Light type coercion so that CLI strings like '1e-5' become floats."""
    if v is None:
        return v
    origin = get_origin(t)
    if origin is Union:
        args = [a for a in get_args(t) if a is not type(None)]
        return _coerce(v, args[0]) if args else v
    if t is float and isinstance(v, (int, str)):
        return float(v)
    if t is int and isinstance(v, str):
        return int(float(v))
    if t is bool and isinstance(v, str):
        return v.lower() in {"1", "true", "yes", "on"}
    if t is str and not isinstance(v, str):
        return str(v)
    return v


def deep_merge(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def parse_overrides(items: Optional[List[str]]) -> Dict[str, Any]:
    """['train.lr=1e-5', 'logging.wandb.mode=offline'] -> nested dict (YAML-typed values)."""
    out: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        key, raw = item.split("=", 1)
        val = yaml.safe_load(raw)
        node = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return out


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def available(kind: str) -> List[str]:
    """List config names in configs/<kind>/ (models, methods, domains)."""
    return sorted(p.stem for p in (CONFIG_ROOT / kind).glob("*.yaml"))


def load_config(model: str, method: str, domain: str = "atc", seed: int = 42,
                overrides: Optional[List[str]] = None,
                config_root: Path = CONFIG_ROOT) -> ExperimentConfig:
    merged = _read_yaml(config_root / "base.yaml")
    merged = deep_merge(merged, _read_yaml(config_root / "domains" / f"{domain}.yaml"))
    merged = deep_merge(merged, _read_yaml(config_root / "methods" / f"{method}.yaml"))

    model_yaml = _read_yaml(config_root / "models" / f"{model}.yaml")
    method_overrides = model_yaml.pop("method_overrides", {}) or {}
    merged = deep_merge(merged, model_yaml)
    merged = deep_merge(merged, method_overrides.get(method, {}))

    merged = deep_merge(merged, {"run": {"model": model, "method": method,
                                         "domain": domain, "seed": seed},
                                 "model": {"key": model}})
    merged = deep_merge(merged, parse_overrides(overrides))
    cfg = _from_dict(ExperimentConfig, merged)
    resolve_run_dir(cfg)
    return cfg


def load_saved_config(path: Union[str, Path]) -> ExperimentConfig:
    return _from_dict(ExperimentConfig, _read_yaml(Path(path)))


def run_dir_for(output_root: str, domain: str, model: str, method: str, seed: int) -> Path:
    return Path(output_root) / domain / model / method / f"s{seed}"


_SEED_DIR = re.compile(r"s(\d+)")


def iter_run_dirs(root: Union[str, Path], domain: str, filename: str):
    """Yield (model, method, seed, file) for *current* runs only: seed folders named exactly
    ``s<seed>``.  Archived attempts (``s<seed>.<timestamp>``) are skipped."""
    for f in sorted((Path(root) / domain).glob(f"*/*/s*/{filename}")):
        m = _SEED_DIR.fullmatch(f.parts[-2])
        if m:
            yield f.parts[-4], f.parts[-3], int(m.group(1)), f


def resolve_run_dir(cfg: ExperimentConfig) -> Path:
    if not cfg.run.run_dir:
        cfg.run.run_dir = str(run_dir_for(cfg.run.output_root, cfg.run.domain,
                                          cfg.run.model, cfg.run.method, cfg.run.seed))
    return Path(cfg.run.run_dir)


def resolve_path(p: str) -> str:
    """Relative artefact/data paths are resolved against CWD first, then the repo root."""
    if not p:
        return p
    path = Path(p)
    if path.is_absolute() or path.exists():
        return str(path)
    alt = REPO_ROOT / path
    return str(alt) if alt.exists() else str(path)
