"""Local inference engine for PinPoint v4.

Replaces the Ollama / Anthropic round-trip in ``agent.chat_completion`` with a
locally-trained GPT-2 scale model.

Submodules import torch lazily: ``config``, ``confidence`` and ``persona`` work
anywhere, while ``data``, ``train`` and ``provider`` need torch only when their
functions are actually called.
"""

from .config import ModelConfig, TrainConfig, RunConfig, PRESETS
from .confidence import GenerationSignals, ConfidenceReport, assess, calibrate

__all__ = [
    "ModelConfig", "TrainConfig", "RunConfig", "PRESETS",
    "GenerationSignals", "ConfidenceReport", "assess", "calibrate",
]
