"""Epistemic status labels.

PinPoint must be able to say *how* it knows something. Conflating "I saw the
file appear" with "the tool didn't print an error" is the mechanism behind most
confident-sounding hallucinated state, so the two get different labels and the
difference is carried through the whole pipeline.
"""

OBSERVED = "OBSERVED"      # directly checked — the file is on disk, exit code was 0
INFERRED = "INFERRED"      # deduced from evidence that stops short of proof
ASSUMED = "ASSUMED"        # taken on faith to make progress; may be wrong
REQUESTED = "REQUESTED"    # asked for, outcome not yet known
COMPLETED = "COMPLETED"    # finished and verified
FAILED = "FAILED"          # known not to have worked
UNKNOWN = "UNKNOWN"        # genuinely not known — the honest default

ALL = (OBSERVED, INFERRED, ASSUMED, REQUESTED, COMPLETED, FAILED, UNKNOWN)

# How much weight a claim carries, for confidence arithmetic.
WEIGHT = {
    OBSERVED: 1.0,
    COMPLETED: 1.0,
    INFERRED: 0.6,
    ASSUMED: 0.3,
    REQUESTED: 0.2,
    FAILED: 0.0,
    UNKNOWN: 0.0,
}


def label(status: str, text: str) -> str:
    """Format a claim with its epistemic status, e.g. ``OBSERVED: exit code 0``."""
    status = status if status in ALL else UNKNOWN
    return f"{status}: {text}"


def weight(status: str) -> float:
    return WEIGHT.get(status, 0.0)
