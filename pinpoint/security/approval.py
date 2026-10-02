"""Human approval workflow.

The policy engine decides *whether* to ask; this module does the asking. It
fails closed: if no human is reachable, an action that needs approval does not
run. A hook may answer ``{"approved": True, "remember": True}`` to create a
standing grant so identical low-risk actions stop interrupting the user.

Nothing here consults the model. The hook is wired to a real human interface
(CLI prompt, PWA, keyboard listener) by ``main.py``.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from pinpoint.security import audit, permissions

# Hook signature: fn(ApprovalRequest) -> bool | dict
_hook: Optional[Callable] = None
_legacy_hook: Optional[Callable] = None
_default_timeout = 300.0

_history: List[dict] = []


@dataclass
class ApprovalRequest:
    """Everything a human needs to decide, and nothing they don't."""
    tool: str
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    risk: str = "medium"
    level: str = permissions.YELLOW
    preview: str = ""
    goal: str = ""
    reversible: bool = True
    external_side_effect: bool = False
    costs_money: bool = False

    def render(self) -> str:
        lines = [
            "═" * 56,
            "ACTION REQUIRES APPROVAL",
            "",
            f"Action : {self.tool}",
        ]
        if self.goal:
            lines.append(f"Goal   : {self.goal[:120]}")
        if self.preview:
            lines.append(f"Detail : {self.preview[:400]}")
        lines += [
            f"Reason : {self.reason}",
            f"Risk   : {self.risk}"
            + ("  (irreversible)" if not self.reversible else "")
            + ("  (external)" if self.external_side_effect else "")
            + ("  (costs money)" if self.costs_money else ""),
            "",
            "[Approve]   [Deny]",
            "═" * 56,
        ]
        return "\n".join(lines)


@dataclass
class ApprovalOutcome:
    approved: bool
    source: str = ""
    remembered: bool = False
    reason: str = ""

    def __bool__(self) -> bool:
        return self.approved


def set_hook(fn: Optional[Callable], timeout: Optional[float] = None) -> None:
    """Install the human-facing approval callback."""
    global _hook, _default_timeout
    _hook = fn
    if timeout is not None:
        _default_timeout = timeout


def set_legacy_hook(fn: Optional[Callable]) -> None:
    """Accept the v2 hook signature ``fn(desc, risk, preview, two_step) -> bool``."""
    global _legacy_hook
    _legacy_hook = fn


def has_hook() -> bool:
    return _hook is not None or _legacy_hook is not None


def _risk_of(decision: permissions.Decision) -> str:
    attrs = decision.attributes or {}
    if attrs.get("costs_money") or not attrs.get("reversible", True):
        return "high"
    if attrs.get("external_side_effect"):
        return "high"
    return "medium"


def _preview_for(tool: str, params: Dict[str, Any]) -> str:
    """A short, redacted, human-meaningful summary of what will happen."""
    safe = audit.redact(params or {})
    if tool == "run_shell":
        return str(safe.get("command", ""))[:400]
    if tool in ("write_file", "write_anywhere", "modify_own_source"):
        target = safe.get("filename") or safe.get("path") or "?"
        body = str(safe.get("content") or safe.get("new_content") or "")
        return f"{target}\n{body[:300]}"
    if tool == "delete_file":
        return f"delete {safe.get('path', '?')}"
    if tool in ("send_message", "send_email"):
        return f"to {safe.get('to', '?')}: {str(safe.get('body') or safe.get('message', ''))[:200]}"
    if tool == "make_call":
        return f"call {safe.get('to', '?')}"
    return str(safe)[:300]


def build_request(tool: str, params: Dict[str, Any],
                  decision: Optional[permissions.Decision] = None,
                  goal: str = "") -> ApprovalRequest:
    decision = decision or permissions.check(tool, params)
    attrs = decision.attributes or {}
    return ApprovalRequest(
        tool=tool,
        params=params or {},
        reason=decision.reason,
        risk=_risk_of(decision),
        level=decision.level,
        preview=_preview_for(tool, params or {}),
        goal=goal,
        reversible=bool(attrs.get("reversible", True)),
        external_side_effect=bool(attrs.get("external_side_effect", False)),
        costs_money=bool(attrs.get("costs_money", False)),
    )


def _invoke_hook(request: ApprovalRequest, timeout: float) -> ApprovalOutcome:
    """Call the hook off-thread so a hung UI cannot wedge the agent forever."""
    result: Dict[str, Any] = {}

    def _run():
        try:
            if _hook is not None:
                result["value"] = _hook(request)
            elif _legacy_hook is not None:
                result["value"] = _legacy_hook(
                    f"{request.tool}: {request.reason}", request.risk,
                    request.preview, not request.reversible)
        except Exception as exc:  # a broken UI must not approve by accident
            result["error"] = str(exc)

    thread = threading.Thread(target=_run, daemon=True, name="pinpoint-approval")
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        return ApprovalOutcome(False, source="timeout",
                               reason=f"no answer within {timeout:.0f}s")
    if "error" in result:
        return ApprovalOutcome(False, source="error", reason=result["error"])

    value = result.get("value")
    if isinstance(value, dict):
        return ApprovalOutcome(
            approved=bool(value.get("approved")),
            source=str(value.get("source", "human")),
            remembered=bool(value.get("remember")),
            reason=str(value.get("reason", "")),
        )
    return ApprovalOutcome(approved=bool(value), source="human")


def request(tool: str, params: Optional[Dict[str, Any]] = None,
            decision: Optional[permissions.Decision] = None,
            goal: str = "", timeout: Optional[float] = None) -> ApprovalOutcome:
    """Ask the human. Denies when nobody is reachable."""
    params = params or {}
    decision = decision or permissions.check(tool, params)
    req = build_request(tool, params, decision, goal)

    if not has_hook():
        outcome = ApprovalOutcome(False, source="none",
                                  reason="approval required but no human is available")
    else:
        outcome = _invoke_hook(req, timeout if timeout is not None else _default_timeout)

    if outcome.approved and outcome.remembered:
        # Persist the standing grant through the policy engine, as a human.
        permissions.grant(tool, source=permissions.HUMAN, scope="session",
                          note="approved with 'don't ask again'")

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "approved": outcome.approved,
        "source": outcome.source,
        "reason": outcome.reason or decision.reason,
        "remembered": outcome.remembered,
    }
    _history.append(entry)
    audit.record(
        "approval_decision", tool=tool, params=params, goal=goal,
        permission=decision.level,
        approval="approved" if outcome.approved else "denied",
        error="" if outcome.approved else (outcome.reason or "denied by human"),
    )
    return outcome


def history(limit: int = 50) -> List[dict]:
    return _history[-limit:]


def stats() -> dict:
    granted = sum(1 for h in _history if h["approved"])
    return {
        "requests": len(_history),
        "approved": granted,
        "denied": len(_history) - granted,
        "standing_grants": list(permissions.list_grants().keys()),
    }


def reset_for_tests() -> None:
    global _hook, _legacy_hook
    _hook = None
    _legacy_hook = None
    _history.clear()
