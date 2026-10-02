"""Structured result of a single action.

Note the distinction between :data:`FAILED` and :data:`UNVERIFIED`. A tool that
ran without complaint but whose effect could not be confirmed is *not* a
success — it is unverified, and the agent is expected to say so rather than
report completion.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pinpoint import epistemics

SUCCESS = "SUCCESS"          # ran and the effect was confirmed
FAILED = "FAILED"            # ran and demonstrably did not work
UNVERIFIED = "UNVERIFIED"    # ran, no error, effect could not be confirmed
BLOCKED = "BLOCKED"          # policy refused it (RED, or emergency stop)
DENIED = "DENIED"            # a human declined it
UNAVAILABLE = "UNAVAILABLE"  # the capability does not exist in this environment
SKIPPED = "SKIPPED"          # deliberately not run

TERMINAL_FAILURES = (FAILED, BLOCKED, DENIED, UNAVAILABLE)


@dataclass
class ActionResult:
    """What happened when one action was attempted."""
    tool: str
    params: Dict[str, Any] = field(default_factory=dict)
    status: str = UNVERIFIED
    output: str = ""
    error: str = ""
    action_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_ms: float = 0.0
    confidence: float = 0.5
    reversible: bool = True
    side_effects: List[str] = field(default_factory=list)
    verification: Dict[str, Any] = field(default_factory=dict)
    permission_level: str = ""
    approval: str = ""
    epistemic: str = epistemics.UNKNOWN
    goal: str = ""
    attempt: int = 1

    @property
    def ok(self) -> bool:
        """True only when the effect was actually confirmed."""
        return self.status == SUCCESS

    @property
    def ran(self) -> bool:
        """True if the tool was actually invoked (as opposed to refused)."""
        return self.status in (SUCCESS, FAILED, UNVERIFIED)

    @property
    def retryable(self) -> bool:
        """Blocked and denied actions must not be retried unchanged."""
        return self.status in (FAILED, UNVERIFIED)

    def to_dict(self) -> dict:
        return {
            "action_id": self.action_id, "timestamp": self.timestamp,
            "tool": self.tool, "params": self.params, "status": self.status,
            "output": self.output[:2000], "error": self.error,
            "duration_ms": round(self.duration_ms, 2), "confidence": self.confidence,
            "reversible": self.reversible, "side_effects": self.side_effects,
            "verification": self.verification, "permission_level": self.permission_level,
            "approval": self.approval, "epistemic": self.epistemic,
            "goal": self.goal, "attempt": self.attempt,
        }

    def summary(self) -> str:
        """One line, honest about what is and isn't known."""
        head = f"[{self.status}] {self.tool}"
        if self.error:
            return f"{head} — {self.error[:160]}"
        if self.status == UNVERIFIED:
            detail = self.verification.get("detail", "effect not confirmed")
            return f"{head} — ran, but {detail}"
        return f"{head} — {epistemics.label(self.epistemic, self.output[:140].strip())}"

    def as_observation(self) -> str:
        """How this result should be described back to the model or the user."""
        if self.status == SUCCESS:
            return epistemics.label(epistemics.OBSERVED,
                                    f"{self.tool} succeeded: "
                                    f"{self.verification.get('detail', 'confirmed')}")
        if self.status == FAILED:
            return epistemics.label(epistemics.FAILED,
                                    f"{self.tool} failed: {self.error or self.output[:200]}")
        if self.status == UNVERIFIED:
            return epistemics.label(
                epistemics.UNKNOWN,
                f"{self.tool} ran but the result could not be confirmed "
                f"({self.verification.get('detail', 'no independent check')}). "
                f"Do not report this as done.")
        if self.status == BLOCKED:
            return epistemics.label(epistemics.FAILED, f"{self.tool} was blocked: {self.error}")
        if self.status == DENIED:
            return epistemics.label(epistemics.FAILED, f"{self.tool} was declined by the human.")
        if self.status == UNAVAILABLE:
            return epistemics.label(epistemics.FAILED,
                                    f"{self.tool} is unavailable here: {self.error}")
        return epistemics.label(epistemics.UNKNOWN, f"{self.tool}: {self.status}")


def blocked(tool: str, params: dict, reason: str, level: str = "") -> ActionResult:
    return ActionResult(tool=tool, params=params or {}, status=BLOCKED, error=reason,
                        permission_level=level, confidence=1.0,
                        epistemic=epistemics.OBSERVED)


def denied(tool: str, params: dict, reason: str, level: str = "") -> ActionResult:
    return ActionResult(tool=tool, params=params or {}, status=DENIED, error=reason,
                        permission_level=level, approval="denied", confidence=1.0,
                        epistemic=epistemics.OBSERVED)


def unavailable(tool: str, params: dict, reason: str) -> ActionResult:
    return ActionResult(tool=tool, params=params or {}, status=UNAVAILABLE, error=reason,
                        confidence=1.0, epistemic=epistemics.OBSERVED)


def from_exception(tool: str, params: dict, exc: Exception,
                   duration_ms: float = 0.0) -> ActionResult:
    return ActionResult(tool=tool, params=params or {}, status=FAILED,
                        error=f"{type(exc).__name__}: {exc}", duration_ms=duration_ms,
                        confidence=1.0, epistemic=epistemics.OBSERVED)
