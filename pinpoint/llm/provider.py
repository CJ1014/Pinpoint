"""Local inference provider — the third branch of ``agent.chat_completion``.

Returns objects shaped like the OpenAI response the rest of the agent already
consumes (``.choices[0].message.content``), so existing call sites need no
change. Alongside that it captures per-token logprobs and hands them to
``pinpoint.llm.confidence``, which is what lets the agent state uncertainty
from measurement rather than from a number the model wrote about itself.

torch and transformers are imported on first load, not at module import.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import os

from .confidence import GenerationSignals, ConfidenceReport, assess
from .persona import FRAMINGS, ACCOUNTABILITY


@dataclass
class _Message:
    content: str
    role: str = "assistant"


@dataclass
class _Choice:
    message: _Message
    finish_reason: str = "stop"


@dataclass
class LocalResponse:
    """OpenAI-shaped, plus the confidence report the wrapper measured."""

    choices: List[_Choice]
    confidence: Optional[ConfidenceReport] = None
    usage: Optional[Dict[str, int]] = None

    @property
    def text(self) -> str:
        return self.choices[0].message.content if self.choices else ""


class LocalModelUnavailable(RuntimeError):
    """Raised when the checkpoint cannot be loaded.

    Deliberately fatal rather than silently falling back to another provider:
    a run that quietly answers from Ollama while you believe you are testing
    the local model produces evaluation numbers that mean nothing.
    """


class LocalProvider:
    """Wraps a trained checkpoint for use as PinPoint's inference engine."""

    def __init__(self, checkpoint: str, device: Optional[str] = None,
                 framing: str = ACCOUNTABILITY, max_context: int = 1024):
        self.checkpoint = checkpoint
        self.device_str = device
        self.framing = framing
        self.max_context = max_context
        self._model = None
        self._tok = None
        self._device = None

    # -- loading ----------------------------------------------------------

    def available(self) -> bool:
        if not os.path.isdir(self.checkpoint):
            return False
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def missing_config(self) -> str:
        if not os.path.isdir(self.checkpoint):
            return f"no checkpoint at {self.checkpoint}"
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            return f"missing dependency: {exc.name}"
        return ""

    def load(self):
        if self._model is not None:
            return self._model

        why = self.missing_config()
        if why:
            raise LocalModelUnavailable(
                f"cannot load local model — {why}. "
                f"Set LLM_PROVIDER to 'ollama' or 'claude' to use a remote engine."
            )

        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        self._device = torch.device(
            self.device_str or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._tok = GPT2TokenizerFast.from_pretrained(self.checkpoint)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._model = GPT2LMHeadModel.from_pretrained(self.checkpoint).to(self._device)
        self._model.eval()
        return self._model

    # -- prompting --------------------------------------------------------

    def format_prompt(self, messages: Sequence[Dict[str, str]], system: str = "") -> str:
        """Render chat messages in the same format persona.Turn trains on.

        Drift between this and ``persona.Turn.render`` is a silent quality
        killer: the model is conditioned on markers it never sees at inference.
        Change both together.
        """
        sys_text = system or FRAMINGS.get(self.framing, "")
        parts = [f"<|system|>\n{sys_text}"]
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                continue          # already placed at the front
            marker = "<|assistant|>" if role == "assistant" else "<|user|>"
            parts.append(f"{marker}\n{content}")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    def _truncate(self, ids: List[int], reserve: int) -> List[int]:
        """Keep the tail of an over-long prompt — recent turns matter most."""
        budget = self.max_context - reserve
        return ids[-budget:] if budget > 0 and len(ids) > budget else ids

    # -- generation -------------------------------------------------------

    def generate(self, messages: Sequence[Dict[str, str]], system: str = "",
                 max_tokens: int = 200, temperature: float = 0.8,
                 top_p: float = 0.95, stop: Optional[Sequence[str]] = None,
                 verified: Optional[bool] = None,
                 grounded: Optional[bool] = None) -> LocalResponse:
        """Generate one completion and measure its confidence.

        ``verified`` and ``grounded`` are passed through to the confidence
        assessment when the caller already knows them (for instance when the
        generation is about to be checked against a tool result).
        """
        import torch

        self.load()

        prompt = self.format_prompt(messages, system)
        ids = self._truncate(self._tok.encode(prompt), reserve=max_tokens)
        input_ids = torch.tensor([ids], device=self._device)

        with torch.no_grad():
            out = self._model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=self._tok.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,          # KV cache
            )

        new_ids = out.sequences[0][len(ids):]
        text = self._tok.decode(new_ids, skip_special_tokens=True)

        # Per-token logprob of what was actually sampled. This is the signal
        # the confidence layer needs; it cannot be recovered after the fact.
        logprobs: List[float] = []
        for score, tok_id in zip(out.scores, new_ids):
            lp = torch.log_softmax(score[0].float(), dim=-1)
            logprobs.append(lp[tok_id].item())

        finish = "length" if len(new_ids) >= max_tokens else "stop"
        if stop:
            cut = min((text.find(s) for s in stop if text.find(s) != -1), default=-1)
            if cut != -1:
                text = text[:cut]
                # Re-encode the kept text to learn how many tokens survived, so
                # the confidence signal describes the span actually returned
                # rather than the discarded tail as well.
                logprobs = logprobs[: len(self._tok.encode(text))]
                finish = "stop"

        report = assess(GenerationSignals(
            token_logprobs=logprobs,
            verified=verified,
            grounded_in_context=grounded,
            n_context_tokens=len(ids),
            truncated=(finish == "length"),
        ))

        return LocalResponse(
            choices=[_Choice(message=_Message(content=text.strip()), finish_reason=finish)],
            confidence=report,
            usage={"prompt_tokens": len(ids), "completion_tokens": len(new_ids),
                   "total_tokens": len(ids) + len(new_ids)},
        )

    def stream(self, messages: Sequence[Dict[str, str]], system: str = "",
               max_tokens: int = 200, temperature: float = 0.8):
        """Yield OpenAI-shaped delta chunks.

        Note: confidence is only available after the full generation, so a
        streamed call cannot label its own uncertainty mid-flight. Callers that
        need the label should use ``generate``.
        """
        import torch
        from transformers import TextIteratorStreamer
        from threading import Thread

        self.load()
        prompt = self.format_prompt(messages, system)
        ids = self._truncate(self._tok.encode(prompt), reserve=max_tokens)
        input_ids = torch.tensor([ids], device=self._device)

        streamer = TextIteratorStreamer(self._tok, skip_prompt=True,
                                        skip_special_tokens=True)
        kwargs = dict(inputs=input_ids, max_new_tokens=max_tokens,
                      do_sample=temperature > 0,
                      temperature=max(temperature, 1e-5),
                      pad_token_id=self._tok.eos_token_id,
                      streamer=streamer, use_cache=True)
        Thread(target=self._model.generate, kwargs=kwargs, daemon=True).start()

        class _Delta:
            def __init__(self, content):
                self.content = content

        class _StreamChoice:
            def __init__(self, content):
                self.delta = _Delta(content)

        class _Chunk:
            def __init__(self, content):
                self.choices = [_StreamChoice(content)]

        for piece in streamer:
            if piece:
                yield _Chunk(piece)


_PROVIDER: Optional[LocalProvider] = None


def get_provider(checkpoint: Optional[str] = None) -> LocalProvider:
    """Process-wide provider. Loading a 355M model per call is not viable."""
    global _PROVIDER
    path = checkpoint or os.environ.get("PINPOINT_MODEL_PATH", "checkpoints/pinpoint-v4/best")
    if _PROVIDER is None or _PROVIDER.checkpoint != path:
        _PROVIDER = LocalProvider(
            path,
            framing=os.environ.get("PINPOINT_FRAMING", ACCOUNTABILITY),
        )
    return _PROVIDER
