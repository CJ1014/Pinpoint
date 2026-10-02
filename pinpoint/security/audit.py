"""Structured audit trail.

Every consequential action appends one JSON object to a daily JSONL file, so
what PinPoint actually did can be reconstructed after the fact — including the
permission decision, whether a human approved it, what the verification said,
and how a failure was recovered from.

Secrets are redacted before anything is written. An audit log that leaks the
credentials it was meant to protect is worse than no audit log.
"""

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pinpoint import jsonstore, paths

# Parameter names whose values are never written to disk.
_SECRET_KEYS = {
    "password", "passwd", "secret", "token", "api_key", "apikey", "auth",
    "authorization", "credential", "credentials", "access_token", "private_key",
    "session_key", "account_sid", "auth_token",
}

# Values that look like credentials even under an innocent key name.
_SECRET_VALUE_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),          # OpenAI / Anthropic style
    re.compile(r"AC[0-9a-fA-F]{30,}"),               # Twilio account SID
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),             # GitHub token
    re.compile(r"AKIA[0-9A-Z]{12,}"),                # AWS access key id
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{16,}"),    # bearer header
]

_REDACTED = "[REDACTED]"
_MAX_VALUE_LEN = 400


def _redact_value(value: Any) -> Any:
    """Recursively strip anything that looks like a credential."""
    if isinstance(value, dict):
        return {k: (_REDACTED if k.lower() in _SECRET_KEYS else _redact_value(v))
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_value(v) for v in value]
    if isinstance(value, str):
        out = value
        for pattern in _SECRET_VALUE_PATTERNS:
            out = pattern.sub(_REDACTED, out)
        if len(out) > _MAX_VALUE_LEN:
            out = out[:_MAX_VALUE_LEN] + "…"
        return out
    return value


def redact(params: Any) -> Any:
    """Public entry point for redaction (used by approval previews too)."""
    return _redact_value(params)


def _log_path(day: Optional[str] = None) -> str:
    day = day or datetime.now(timezone.utc).strftime("%Y%m%d")
    return paths.output_dir("audit", f"audit-{day}.jsonl")


def record(
    action: str,
    *,
    tool: str = "",
    params: Any = None,
    result: str = "",
    error: str = "",
    goal: str = "",
    session: Any = "",
    permission: str = "",
    approval: str = "",
    verification: Any = None,
    recovery: str = "",
    duration_ms: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> dict:
    """Append one audit entry. Never raises; returns the entry written."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "tool": tool,
        "session": session,
        "goal": goal[:200] if isinstance(goal, str) else goal,
        "params": redact(params if params is not None else {}),
        "result": redact(result)[:600] if isinstance(result, str) else result,
        "error": redact(error)[:400] if isinstance(error, str) else error,
        "permission": permission,
        "approval": approval,
        "verification": verification,
        "recovery": recovery,
        "duration_ms": duration_ms,
    }
    if extra:
        entry.update(redact(extra))
    jsonstore.append_jsonl(_log_path(), entry)
    return entry


def query(
    *,
    day: Optional[str] = None,
    tool: str = "",
    action: str = "",
    session: Any = None,
    limit: int = 0,
) -> List[dict]:
    """Read back audit entries, optionally filtered. For reconstruction."""
    records = jsonstore.read_jsonl(_log_path(day))
    if tool:
        records = [r for r in records if r.get("tool") == tool]
    if action:
        records = [r for r in records if r.get("action") == action]
    if session is not None:
        records = [r for r in records if r.get("session") == session]
    return records[-limit:] if limit > 0 else records


def summarize(day: Optional[str] = None) -> dict:
    """Aggregate counts for a day — used by the dashboard and reports."""
    records = query(day=day)
    tools: Dict[str, int] = {}
    for r in records:
        name = r.get("tool") or r.get("action") or "?"
        tools[name] = tools.get(name, 0) + 1
    return {
        "total": len(records),
        "errors": sum(1 for r in records if r.get("error")),
        "approvals": sum(1 for r in records if r.get("approval") == "approved"),
        "denials": sum(1 for r in records if r.get("approval") == "denied"),
        "by_tool": dict(sorted(tools.items(), key=lambda kv: -kv[1])),
    }
