"""Confidence that is measured, not narrated.

The v4 spec asks for a model that flags uncertainty — "I'm about 87% confident
in this approach." There are two ways to get that number, and only one of them
is real:

1. Train the model to emit confidence tokens. The number then comes from the
   same distribution that produced the answer, and is fitted to the *shape* of
   confident writing in the training data. A model has no introspective access
   to its own error rate, so a trained-in "87%" is a fluent guess wearing a
   decimal point. It will read as calibrated and correlate with nothing.

2. Compute it from signals the runtime can actually observe — how sharp the
   token distribution was, whether the claim survived verification, whether it
   was grounded in retrieved context — and have the *wrapper* attach it.

This module does (2). It is the same rule ``communication/providers/base.py``
applies to delivery: a provider that cannot produce a confirmation must say so
rather than phrase an unconfirmed send as a success. An uncheckable confidence
number is an unconfirmed send.

Output is an epistemic label from ``pinpoint.epistemics``, because that is the
vocabulary the rest of the agent already reasons in.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence
import math

from pinpoint import epistemics as E

# Mean-logprob thresholds separating label bands. These are starting points,
# not discovered constants: they must be re-fit against real task outcomes
# before any of the numbers below deserve to be trusted. See calibrate().
DEFAULT_BANDS = {
    "observed": -0.25,   # very sharp distribution AND externally verified
    "inferred": -0.85,   # coherent, moderately sharp
    "assumed": -2.00,    # loose — the model is improvising
}


@dataclass
class GenerationSignals:
    """What the runtime observed while producing one span of text.

    Every field is something the inference loop can measure. Nothing here is
    reported by the model about itself.
    """

    token_logprobs: Sequence[float] = field(default_factory=list)
    verified: Optional[bool] = None      # did an external check pass?
    grounded_in_context: Optional[bool] = None   # claim traceable to retrieved text?
    n_context_tokens: int = 0
    truncated: bool = False              # hit max_tokens mid-thought

    @property
    def mean_logprob(self) -> Optional[float]:
        if not self.token_logprobs:
            return None
        return sum(self.token_logprobs) / len(self.token_logprobs)

    @property
    def min_logprob(self) -> Optional[float]:
        """The weakest token. A single very unlikely token often marks the exact
        place a generation went off the rails, which the mean hides."""
        return min(self.token_logprobs) if self.token_logprobs else None

    @property
    def perplexity(self) -> Optional[float]:
        m = self.mean_logprob
        return math.exp(-m) if m is not None else None


@dataclass
class ConfidenceReport:
    """The honest answer to 'how sure are you?'"""

    label: str                       # one of pinpoint.epistemics
    mean_logprob: Optional[float] = None
    perplexity: Optional[float] = None
    basis: List[str] = field(default_factory=list)   # why this label
    caveats: List[str] = field(default_factory=list)

    def render(self) -> str:
        """A sentence the agent can actually say out loud."""
        head = {
            E.OBSERVED: "Verified",
            E.INFERRED: "Reasoned, not verified",
            E.ASSUMED: "Working assumption",
            E.UNKNOWN: "Not established",
        }.get(self.label, self.label)

        parts = [head]
        if self.basis:
            parts.append("(" + "; ".join(self.basis) + ")")
        if self.caveats:
            parts.append("— " + "; ".join(self.caveats))
        return " ".join(parts)


def assess(signals: GenerationSignals, bands: Optional[dict] = None) -> ConfidenceReport:
    """Turn observed generation signals into an epistemic label.

    The ordering rule that matters: external verification outranks fluency.
    A sharp, low-perplexity generation that failed its check is FAILED, not
    OBSERVED — the model being sure is not evidence.
    """
    bands = bands or DEFAULT_BANDS
    basis: List[str] = []
    caveats: List[str] = []

    # Verification, when present, dominates everything else.
    if signals.verified is False:
        return ConfidenceReport(
            label=E.FAILED,
            mean_logprob=signals.mean_logprob,
            perplexity=signals.perplexity,
            basis=["external check failed"],
            caveats=["fluency of the output is not evidence against this"],
        )

    m = signals.mean_logprob
    if m is None:
        return ConfidenceReport(
            label=E.UNKNOWN,
            basis=["no token-level signal captured"],
            caveats=["confidence cannot be reported for this generation"],
        )

    if signals.verified is True:
        basis.append("external check passed")
        label = E.OBSERVED
    elif m >= bands["inferred"]:
        basis.append(f"mean logprob {m:.2f}")
        label = E.INFERRED
    elif m >= bands["assumed"]:
        basis.append(f"mean logprob {m:.2f} — loose")
        label = E.ASSUMED
    else:
        basis.append(f"mean logprob {m:.2f} — very loose")
        label = E.UNKNOWN

    # Grounding can only ever weaken an unverified claim, never strengthen it
    # past what a check would establish.
    if signals.grounded_in_context is False and label == E.INFERRED:
        label = E.ASSUMED
        caveats.append("not traceable to retrieved context")
    elif signals.grounded_in_context:
        basis.append("grounded in retrieved context")

    lo = signals.min_logprob
    if lo is not None and lo < -6.0:
        caveats.append(f"contains a very low-probability token ({lo:.1f}) — likely a fabricated span")

    if signals.truncated:
        caveats.append("output was truncated; the conclusion may be incomplete")

    if signals.n_context_tokens == 0 and label in (E.OBSERVED, E.INFERRED):
        caveats.append("produced without retrieved context — from parameters alone")

    return ConfidenceReport(
        label=label,
        mean_logprob=m,
        perplexity=signals.perplexity,
        basis=basis,
        caveats=caveats,
    )


def calibrate(samples: Sequence[tuple]) -> dict:
    """Re-fit the label bands against real outcomes.

    ``samples`` is a sequence of ``(mean_logprob, was_correct)``. Returns
    suggested band edges placed where observed accuracy crosses 0.9 and 0.6.

    Until this has been run on real task outcomes, DEFAULT_BANDS are guesses.
    Treat any confidence claim built on unfitted bands as ASSUMED, and say so.
    """
    scored = sorted((float(lp), bool(ok)) for lp, ok in samples)
    if len(scored) < 50:
        raise ValueError(
            f"need >= 50 samples to fit bands, got {len(scored)}; "
            f"fitting on fewer produces bands that encode noise"
        )

    def edge_at(target: float) -> float:
        # Walk from the confident end down until running accuracy drops below target.
        hits = 0
        for i, (lp, ok) in enumerate(reversed(scored), start=1):
            hits += ok
            if hits / i < target:
                return lp
        return scored[0][0]

    return {
        "observed": edge_at(0.90),
        "inferred": edge_at(0.90),
        "assumed": edge_at(0.60),
    }
