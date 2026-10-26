# reality_check.py — Anti-Hallucination Foundation

import json
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class RealityAnchor:
    """Ground truth about what's actually happened in this session and before."""

    def __init__(self, memory_file: str = "memory.json"):
        self.memory_file = memory_file
        self.current_session_id = None
        self.current_session_goal = None
        self.current_session_actions = []
        self.current_session_files_created = []
        self.current_session_tools_called = []
        self.current_session_start_time = None

    def initialize_session(self, session_num: int, goal: str):
        """At session start, create a real session record."""
        self.current_session_start_time = datetime.now()
        self.current_session_id = f"session_{session_num}_{self.current_session_start_time.strftime('%Y%m%d_%H%M%S')}"
        self.current_session_goal = goal
        self.current_session_actions = []
        self.current_session_files_created = []
        self.current_session_tools_called = []

        # Log it to memory immediately
        self._save_to_memory({
            "event": "session_start",
            "session_id": self.current_session_id,
            "goal": goal,
            "timestamp": datetime.now().isoformat(),
        })

    def log_action(self, action_description: str):
        """Log every real action taken."""
        self.current_session_actions.append({
            "action": action_description,
            "timestamp": datetime.now().isoformat(),
        })
        self._save_to_memory({
            "event": "action",
            "session_id": self.current_session_id,
            "action": action_description,
            "timestamp": datetime.now().isoformat(),
        })

    def log_file_created(self, filename: str, content_preview: str = ""):
        """Log every file created — this is verifiable."""
        exists = os.path.exists(filename)
        self.current_session_files_created.append({
            "filename": filename,
            "timestamp": datetime.now().isoformat(),
            "exists": exists,
            "content_preview": content_preview[:50] if content_preview else "",
        })
        self._save_to_memory({
            "event": "file_created",
            "session_id": self.current_session_id,
            "filename": filename,
            "exists": exists,
            "timestamp": datetime.now().isoformat(),
        })

    def log_tool_call(self, tool_name: str, tool_input: dict, result: str):
        """Log every tool call — input, output, timestamp."""
        result_summary = result.strip()[:150] if result else ""
        success = not any(err in result.lower() for err in ["error:", "failed", "traceback"])

        self.current_session_tools_called.append({
            "tool": tool_name,
            "input_keys": list(tool_input.keys()),
            "result_preview": result_summary,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        })
        self._save_to_memory({
            "event": "tool_called",
            "session_id": self.current_session_id,
            "tool": tool_name,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        })

    def get_session_summary(self) -> str:
        """What actually happened this session? Return verifiable facts only."""
        elapsed = (datetime.now() - self.current_session_start_time).total_seconds() if self.current_session_start_time else 0

        summary = f"""
╔════════════════════════════════════════════════════════════╗
║           SESSION REALITY SUMMARY (VERIFIED FACTS)         ║
╚════════════════════════════════════════════════════════════╝

Session ID: {self.current_session_id}
Goal: {self.current_session_goal}
Duration: {elapsed:.0f} seconds

ACTIONS TAKEN (Verified): {len(self.current_session_actions)}
"""
        for i, a in enumerate(self.current_session_actions[:15], 1):
            summary += f"  {i}. {a['action']}\n"
        if len(self.current_session_actions) > 15:
            summary += f"  ... and {len(self.current_session_actions) - 15} more\n"

        summary += f"""
FILES CREATED (VERIFIED TO EXIST): {len(self.current_session_files_created)}
"""
        for f in self.current_session_files_created:
            status = "✓ EXISTS" if f['exists'] else "✗ NOT FOUND"
            summary += f"  {f['filename']} [{status}]\n"

        summary += f"""
TOOLS CALLED: {len(self.current_session_tools_called)}
"""
        for t in self.current_session_tools_called[:10]:
            status = "✓" if t['success'] else "✗"
            summary += f"  {status} {t['tool']}\n"
        if len(self.current_session_tools_called) > 10:
            summary += f"  ... and {len(self.current_session_tools_called) - 10} more\n"

        summary += """
╔════════════════════════════════════════════════════════════╗
║ CRITICAL: ANYTHING NOT IN THIS LIST NEVER HAPPENED        ║
║ If you're about to reference something not listed above,   ║
║ it's a hallucination. STOP. Call verify_claim() first.     ║
╚════════════════════════════════════════════════════════════╝
"""
        return summary

    def verify_claim(self, claim: str) -> Tuple[bool, str]:
        """Check if a claim matches reality.

        Returns: (is_true, evidence_or_contradiction)
        """
        claim_lower = claim.lower()

        # Check against actual session data
        action_descriptions = [a['action'].lower() for a in self.current_session_actions]
        file_names = [f['filename'].lower() for f in self.current_session_files_created]
        tool_names = [t['tool'].lower() for t in self.current_session_tools_called]

        # Exact matches first
        for action in action_descriptions:
            if action == claim_lower:
                return (True, f"✓ VERIFIED: Exact match in action log")

        for filename in file_names:
            if filename in claim_lower or claim_lower in filename:
                return (True, f"✓ VERIFIED: File '{filename}' created")

        for tool in tool_names:
            if tool in claim_lower:
                return (True, f"✓ VERIFIED: Tool '{tool}' called")

        # Partial matches
        for action in action_descriptions:
            if len(action) > 5 and action in claim_lower:
                return (True, f"✓ LIKELY VERIFIED: '{action}' in action log")

        # Claim doesn't match any reality
        summary = (
            f"✗ NOT VERIFIED: This claim doesn't match session history.\n"
            f"  Session has:\n"
            f"    • {len(self.current_session_actions)} actions logged\n"
            f"    • {len(self.current_session_files_created)} files created\n"
            f"    • {len(self.current_session_tools_called)} tools called\n\n"
            f"  Before proceeding, call verify_claim() on anything you're not sure about."
        )
        return (False, summary)

    def get_previous_sessions_summary(self) -> str:
        """What did we accomplish in PREVIOUS sessions?"""
        memory = self._load_memory()
        reality_log = memory.get("reality_log", [])

        # Group by session
        sessions = {}
        for event in reality_log:
            sid = event.get("session_id", "unknown")
            if sid not in sessions:
                sessions[sid] = []
            sessions[sid].append(event)

        if not sessions:
            return "No previous session data found."

        summary = "═══ PREVIOUS SESSIONS ═══\n"
        for sid in sorted(sessions.keys())[-5:]:  # Last 5 sessions
            events = sessions[sid]
            summary += f"\n{sid}:\n"
            for e in events[:5]:  # First 5 events per session
                if e.get("event") == "session_start":
                    summary += f"  Started: {e.get('goal', 'unknown')}\n"
                elif e.get("event") == "file_created":
                    summary += f"  Created: {e.get('filename', 'unknown')}\n"
            if len(events) > 5:
                summary += f"  ... and {len(events) - 5} more events\n"

        return summary

    def _save_to_memory(self, event: dict):
        """Append event to memory.json's reality log."""
        try:
            memory = self._load_memory()
            if "reality_log" not in memory:
                memory["reality_log"] = []
            memory["reality_log"].append(event)
            self._save_memory(memory)
        except Exception as e:
            pass  # Silently fail if memory save fails

    def _load_memory(self) -> dict:
        """Load memory.json."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_memory(self, data: dict):
        """Save memory.json."""
        try:
            with open(self.memory_file, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def to_dict(self) -> dict:
        """Export current state."""
        return {
            "session_id": self.current_session_id,
            "goal": self.current_session_goal,
            "actions_count": len(self.current_session_actions),
            "files_created_count": len(self.current_session_files_created),
            "tools_called_count": len(self.current_session_tools_called),
            "actions": self.current_session_actions[:20],  # Last 20
            "files": self.current_session_files_created,
            "tools": self.current_session_tools_called[:20],  # Last 20
        }


# ── Shared singleton ──────────────────────────────────────────────────────────
# agent.py initializes sessions on it; tools.py dispatch reads it for
# verify_claim / get_session_reality. Lives here so neither imports the other.
# Uses its own log file (not PinPoint's main memory.json) so concurrent writes
# from the chat thread can never clobber the reality log or vice versa.
_ANCHOR_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "reality_log.json")
_anchor: Optional[RealityAnchor] = None


def get_anchor() -> RealityAnchor:
    global _anchor
    if _anchor is None:
        try:
            os.makedirs(os.path.dirname(_ANCHOR_FILE), exist_ok=True)
        except Exception:
            pass
        _anchor = RealityAnchor(_ANCHOR_FILE)
    return _anchor
