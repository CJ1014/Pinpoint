"""Model and training configuration for the local PinPoint inference engine.

Deliberately torch-free so configs can be imported, validated and tested on a
machine that will never run training — the laptop that edits the config is not
usually the box that holds the GPU.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import os

# Preset sizes. Parameter counts are the standard GPT-2 family figures and
# include embeddings; they are recorded here so a config can be sanity-checked
# against available VRAM before a run starts rather than after it OOMs.
PRESETS = {
    "gpt2":        dict(n_layer=12, n_head=12, n_embd=768,  params=124_000_000),
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024, params=355_000_000),
    "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280, params=774_000_000),
}


@dataclass
class ModelConfig:
    """Architecture. Defaults are GPT-2 small (124M)."""

    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_positions: int = 1024          # context window
    vocab_size: int = 50257          # GPT-2 BPE
    dropout: float = 0.1
    init_from: str = "gpt2"          # HF checkpoint name, or "scratch"

    @classmethod
    def preset(cls, name: str, **overrides) -> "ModelConfig":
        if name not in PRESETS:
            raise ValueError(f"unknown preset {name!r}; have {sorted(PRESETS)}")
        p = {k: v for k, v in PRESETS[name].items() if k != "params"}
        p.update(overrides)
        return cls(init_from=name, **p)

    def validate(self) -> None:
        if self.n_embd % self.n_head:
            raise ValueError(
                f"n_embd ({self.n_embd}) must divide evenly by n_head "
                f"({self.n_head}); got remainder {self.n_embd % self.n_head}"
            )
        if self.n_positions > 1024 and self.init_from != "scratch":
            raise ValueError(
                f"n_positions={self.n_positions} exceeds the 1024 learned "
                f"position embeddings in pretrained {self.init_from}. Either "
                f"keep 1024, or set init_from='scratch'."
            )

    @property
    def approx_params(self) -> int:
        """Rough parameter count — enough to predict whether a run will fit."""
        emb = self.vocab_size * self.n_embd + self.n_positions * self.n_embd
        # per block: attn (4 * d^2) + mlp (8 * d^2), plus negligible norms
        per_block = 12 * self.n_embd ** 2
        return emb + self.n_layer * per_block


@dataclass
class TrainConfig:
    """One training run.

    ``lr`` defaults to the fine-tuning range. Pretraining from scratch wants
    something closer to 5e-4 with warmup; see ``for_scratch()``.
    """

    lr: float = 5e-5
    batch_size: int = 8
    grad_accum: int = 4              # effective batch = batch_size * grad_accum
    epochs: int = 3
    max_steps: Optional[int] = None  # overrides epochs when set
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    mixed_precision: bool = True

    eval_every: int = 500
    checkpoint_every: int = 1000
    keep_best_only: bool = False     # keep every checkpoint by default; disk is cheap, reruns are not

    seed: int = 1337
    out_dir: str = "checkpoints/pinpoint-v4"
    resume_from: Optional[str] = None

    @classmethod
    def for_scratch(cls, **overrides) -> "TrainConfig":
        base = dict(lr=5e-4, warmup_steps=2000, epochs=1, max_steps=100_000)
        base.update(overrides)
        return cls(**base)

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum


@dataclass
class RunConfig:
    """Model + training together — the thing that gets serialised next to a
    checkpoint so a run can be reproduced or explained six months later."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    notes: str = ""

    def validate(self) -> None:
        self.model.validate()
        if self.train.batch_size < 1 or self.train.grad_accum < 1:
            raise ValueError("batch_size and grad_accum must both be >= 1")

    def to_json(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)
        return path

    @classmethod
    def from_json(cls, path: str) -> "RunConfig":
        with open(path) as fh:
            raw = json.load(fh)
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
            notes=raw.get("notes", ""),
        )
