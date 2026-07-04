"""
human_oversight.py — Structured human-in-the-loop oversight for PinPoint.

Tracks which actions needed approval, logs corrections, and provides
stats so PinPoint can learn from human feedback over time.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

RISKY_TOOLS = frozenset({
    "modify_own_source", "delete_file", "run_shell",
    "write_anywhere", "pip_install",
})

LOW_CONFIDENCE_THRESHOLD = 0.60


class OversightManager:
    def __init__(self):
        self.interventions: list[dict] = []
        self.load()

    def load(self) -> None:
        try:
            from tools import _load_memory  # lazy
            mem = _load_memory()
            self.interventions = mem.get("human_interventions", [])
        except Exception:
            pass

    def save(self) -> None:
        try:
            from tools import _load_memory, _save_memory_file  # lazy
            mem = _load_memory()
            mem["human_interventions"] = self.interventions[-50:]
            _save_memory_file(mem)
        except Exception as e:
            logger.warning("OversightManager.save failed: %s", e)

    def requires_approval(self, tool_name: str, confidence: float = 1.0) -> bool:
        return tool_name in RISKY_TOOLS or confidence < LOW_CONFIDENCE_THRESHOLD

    def format_approval_request(self, tool_name: str, tool_input: dict,
                                 confidence: float, context: str = "") -> str:
        preview = json.dumps(tool_input)[:200]
        lines = [
            "═" * 50,
            "[APPROVAL REQUIRED]",
            f"Action    : {tool_name}",
            f"Input     : {preview}",
            f"Confidence: {confidence:.0%}",
        ]
        if context:
            lines.append(f"Context   : {context[:120]}")
        lines.append("═" * 50)
        return "\n".join(lines)

    def log_approval(self, tool_name: str, tool_input: dict,
                     approved: bool, user_reason: str = "") -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "approval_requested",
            "tool": tool_name,
            "input_preview": json.dumps(tool_input)[:150],
            "approved": approved,
            "user_reason": user_reason[:200],
        }
        self.interventions.append(entry)
        self.save()

    def log_correction(self, what_agent_did: str, correction: str,
                       impact: str = "") -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "correction",
            "agent_action": what_agent_did[:200],
            "correction": correction[:200],
            "impact": impact[:200],
        }
        self.interventions.append(entry)
        self.save()

    def get_summary(self) -> str:
        if not self.interventions:
            return "No human interventions recorded yet."
        recent = self.interventions[-5:]
        lines = [f"Recent interventions ({len(self.interventions)} total):"]
        for i in recent:
            ts = i.get("timestamp", "")[:16]
            evt = i.get("event_type", "")
            tool = i.get("tool", i.get("agent_action", "?"))[:40]
            approved = i.get("approved", None)
            status = "" if approved is None else (" ✓" if approved else " ✗")
            lines.append(f"  [{ts}] {evt}: {tool}{status}")
        return "\n".join(lines)

    def get_stats(self) -> dict:
        approvals = sum(1 for i in self.interventions if i.get("event_type") == "approval_requested" and i.get("approved"))
        rejections = sum(1 for i in self.interventions if i.get("event_type") == "approval_requested" and not i.get("approved"))
        corrections = sum(1 for i in self.interventions if i.get("event_type") == "correction")
        tools = [i.get("tool", "") for i in self.interventions if i.get("tool")]
        most_flagged = max(set(tools), key=tools.count) if tools else ""
        return {
            "total_interventions": len(self.interventions),
            "approvals": approvals,
            "rejections": rejections,
            "corrections": corrections,
            "most_flagged_tool": most_flagged,
        }


# Module-level singleton
_manager = OversightManager()


def requires_approval(tool_name: str, confidence: float = 1.0) -> bool:
    return _manager.requires_approval(tool_name, confidence)


def format_request(tool_name: str, tool_input: dict,
                   confidence: float, context: str = "") -> str:
    return _manager.format_approval_request(tool_name, tool_input, confidence, context)


def log_approval(tool_name: str, tool_input: dict,
                 approved: bool, reason: str = "") -> None:
    _manager.log_approval(tool_name, tool_input, approved, reason)


def get_summary() -> str:
    return _manager.get_summary()
