# human_oversight.py — Phase 7: Human-in-the-Loop Safety

import json
from typing import Dict, List, Optional, Callable
from datetime import datetime

class OversightManager:
    """Track and control risky actions."""

    def __init__(self):
        # Tools that always require approval
        self.risky_tools = {
            "modify_own_source",
            "delete_file",
            "run_shell",
            "pip_install",
        }

        # Log of all user interventions
        self.interventions: List[Dict] = []
        self.approvals_granted: int = 0
        self.approvals_denied: int = 0

    def requires_approval(self, tool_name: str, confidence: float) -> bool:
        """Does this action need human approval?"""
        # Risky tools always need approval
        if tool_name in self.risky_tools:
            return True

        # Low-confidence actions need approval
        if confidence < 0.6:
            return True

        return False

    def request_approval(self, tool_name: str, tool_input: Dict, confidence: float,
                         approval_callback: Optional[Callable] = None) -> bool:
        """Ask user for approval. Return True if approved.

        Prefers approval_callback (routed through PinPoint's input queue —
        a raw input() here would fight the keyboard-listener thread for stdin).
        Falls back to input() when no callback is available (e.g. headless test).
        """
        preview = json.dumps(tool_input, indent=2)[:300]

        if approval_callback is not None:
            risk = "high" if tool_name in ("modify_own_source", "delete_file") else "medium"
            try:
                is_approved = bool(approval_callback(
                    f"{tool_name} (confidence {confidence:.0%})", risk, preview,
                    tool_name == "modify_own_source"))
            except Exception:
                is_approved = False
        else:
            print(f"\n{'='*60}")
            print(f"[APPROVAL REQUIRED]")
            print(f"Tool: {tool_name}")
            print(f"Confidence: {confidence:.0%}")
            print(f"Input: {preview}...")
            print(f"{'='*60}\n")
            try:
                response = input("Approve? (y/n): ").strip().lower()
            except Exception:
                response = "n"
            is_approved = response == 'y'

        # Log the decision
        self.log_intervention(
            "approval_request",
            {
                "tool": tool_name,
                "confidence": confidence,
                "approved": is_approved,
            }
        )

        if is_approved:
            self.approvals_granted += 1
        else:
            self.approvals_denied += 1

        return is_approved

    def log_intervention(self, event_type: str, details: Dict):
        """Log a user intervention or decision."""
        self.interventions.append({
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details,
        })

    def get_oversight_summary(self) -> str:
        """Summary of human oversight."""
        text = f"""HUMAN OVERSIGHT SUMMARY

Total Interventions: {len(self.interventions)}
Approvals Granted: {self.approvals_granted}
Approvals Denied: {self.approvals_denied}

Recent Interventions:
"""
        for intervention in self.interventions[-5:]:
            text += f"  [{intervention['event_type']}] {intervention['details']}\n"

        return text

    def to_dict(self) -> dict:
        return {
            "total_interventions": len(self.interventions),
            "approvals_granted": self.approvals_granted,
            "approvals_denied": self.approvals_denied,
            "recent_interventions": self.interventions[-20:],
        }


# ── Compatibility layer ───────────────────────────────────────────────────────
# Existing integration (agent.py audit hook, tools dispatch) imports module-level
# functions. Shares one manager and persists interventions to memory.json in the
# shape the live viewer panel reads.

_manager = OversightManager()


def _persist_interventions() -> None:
    try:
        from tools import _load_memory, _save_memory_file  # lazy — avoids circular import
        mem = _load_memory()
        # Viewer expects flat entries: {timestamp, event_type, tool, approved}
        mem["human_interventions"] = [
            {
                "timestamp": i.get("timestamp", ""),
                "event_type": i.get("event_type", ""),
                "tool": i.get("details", {}).get("tool", ""),
                "approved": i.get("details", {}).get("approved"),
            }
            for i in _manager.interventions[-50:]
        ]
        _save_memory_file(mem)
    except Exception:
        pass


def requires_approval(tool_name: str, confidence: float = 1.0) -> bool:
    return _manager.requires_approval(tool_name, confidence)


def format_request(tool_name: str, tool_input: dict, confidence: float, context: str = "") -> str:
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


def log_approval(tool_name: str, tool_input: dict, approved: bool, reason: str = "") -> None:
    _manager.log_intervention("approval_request", {
        "tool": tool_name,
        "approved": approved,
        "reason": reason[:200],
    })
    if approved:
        _manager.approvals_granted += 1
    else:
        _manager.approvals_denied += 1
    _persist_interventions()


def get_summary() -> str:
    return _manager.get_oversight_summary()
