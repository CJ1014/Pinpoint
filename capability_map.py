# capability_map.py — Phase 4: Capability Boundaries

import json
from typing import Dict, List, Tuple
from datetime import datetime

# Extra trigger words per domain/boundary — the raw names ("email_access")
# rarely appear verbatim in natural-language tasks ("check my email account").
KEYWORD_HINTS = {
    "email_access": ["email", "inbox", "mailbox", "gmail"],
    "camera_access": ["camera", "webcam", "video feed"],
    "microphone_raw": ["microphone", "raw audio"],
    "future_events": ["predict", "forecast", "future", "will happen"],
    "future_prediction": ["predict", "forecast", "prophecy"],
    "user_location": ["location", "locate", "where i am", "gps"],
    "user_identity": ["identify me", "who am i by"],
    "user_emotions": ["how do i feel", "my emotions", "my mood"],
    "real_world_events": ["news", "current events", "happening now"],
    "code_writing": ["code", "python", "script", "program", "function"],
    "file_operations": ["file", "save", "write", "folder", "directory"],
    "web_search": ["search", "look up", "google", "research online"],
    "math_computation": ["calculate", "math", "compute", "sum"],
    "text_analysis": ["analyze text", "summarize", "parse text"],
}


class CapabilityMap:
    """Know what you can and cannot do. Explicit boundaries."""

    def __init__(self):
        # Domain confidence: 0.0 = impossible, 1.0 = certain
        self.domains = {
            "code_writing": 0.95,
            "file_operations": 0.92,
            "reasoning": 0.88,
            "web_search": 0.80,
            "math_computation": 0.90,
            "text_analysis": 0.85,
            "system_commands": 0.70,
            "self_modification": 0.65,
            "user_emotions": 0.15,
            "real_world_events": 0.20,
            "future_prediction": 0.10,
            "email_access": 0.05,
            "camera_access": 0.00,
            "user_location": 0.10,
        }

        # Hard boundaries: absolutely cannot do these
        self.hard_boundaries = {
            "camera_access": "No camera available — cannot access visual feeds",
            "microphone_raw": "Cannot access raw microphone — only text input possible",
            "email_access": "Cannot access user's email — not available as a tool",
            "future_events": "Cannot predict the future or know unrevealed information",
            "user_location": "Cannot determine user's exact location without explicit permission",
            "user_identity": "Cannot identify users by appearance or personal data",
        }

        self.failures: Dict[str, List[Dict]] = {}
        self.successes: Dict[str, List[Dict]] = {}

    def _matches(self, name: str, task_lower: str) -> bool:
        """Match a domain/boundary name against a task using name forms + hints."""
        search_terms = [
            name,
            name.replace("_", " "),
            name.replace("_", "-"),
        ] + KEYWORD_HINTS.get(name, [])
        return any(term in task_lower for term in search_terms)

    def can_do(self, task: str) -> Tuple[bool, float, str]:
        """Can we do this task? Returns (yes/no, confidence, reason)."""
        task_lower = task.lower()

        # Check hard boundaries first
        for boundary_name, boundary_reason in self.hard_boundaries.items():
            if self._matches(boundary_name, task_lower):
                return (False, 0.0, boundary_reason)

        # Check domain confidence
        for domain, confidence in self.domains.items():
            if self._matches(domain, task_lower):
                if confidence < 0.5:
                    return (
                        False,
                        confidence,
                        f"Low-confidence domain '{domain}' ({confidence:.0%}). Risky to attempt."
                    )
                return (
                    True,
                    confidence,
                    f"Capable domain '{domain}' ({confidence:.0%})"
                )

        # Default: cautiously yes
        return (True, 0.7, "General task — moderate confidence")

    def record_success(self, task: str, domain: str):
        """Learn from successes."""
        if domain not in self.successes:
            self.successes[domain] = []
        self.successes[domain].append({
            "task": task,
            "timestamp": datetime.now().isoformat()
        })
        # Slightly increase confidence
        if domain in self.domains:
            self.domains[domain] = min(1.0, self.domains[domain] + 0.02)

    def record_failure(self, task: str, domain: str, reason: str):
        """Learn from failures."""
        if domain not in self.failures:
            self.failures[domain] = []
        self.failures[domain].append({
            "task": task,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })
        # Decrease confidence
        if domain in self.domains:
            self.domains[domain] = max(0.0, self.domains[domain] - 0.05)

    def suggest_workaround(self, task: str) -> str:
        """If we can't do X, what CAN we do?"""
        task_lower = task.lower()

        if "email" in task_lower:
            return "I cannot access email, but you could: copy/paste email content, or I can draft text you send"
        elif "camera" in task_lower or "video" in task_lower:
            return "No camera access, but you could: describe what you see, take a screenshot, use a separate tool"
        elif "future" in task_lower or "predict" in task_lower:
            return "Cannot predict the future, but I can: analyze trends, model scenarios, plan contingencies"
        elif "locate" in task_lower or "location" in task_lower:
            return "No location data, but you could: tell me your location, or I can help find nearby services"
        elif "see" in task_lower or "look" in task_lower or "view" in task_lower:
            return "I cannot see, but you could: describe what you're looking at, share a screenshot"
        else:
            return "This is outside my capabilities. Can you break it into steps I can assist with?"

    def get_summary(self) -> str:
        """Capability summary for display."""
        summary = "CAPABILITY SUMMARY\n\n"

        # High-confidence domains
        high_conf = [(d, c) for d, c in self.domains.items() if c >= 0.8]
        summary += f"HIGH CONFIDENCE ({len(high_conf)}):\n"
        for domain, conf in sorted(high_conf, key=lambda x: x[1], reverse=True):
            summary += f"  ✓ {domain}: {conf:.0%}\n"

        # Low-confidence domains
        low_conf = [(d, c) for d, c in self.domains.items() if c < 0.5]
        summary += f"\nLOW/NO CONFIDENCE ({len(low_conf)}):\n"
        for domain, conf in sorted(low_conf, key=lambda x: x[1]):
            summary += f"  ✗ {domain}: {conf:.0%}\n"

        # Hard boundaries
        if self.hard_boundaries:
            summary += f"\nHARD BOUNDARIES ({len(self.hard_boundaries)}):\n"
            for boundary in self.hard_boundaries:
                summary += f"  ✗ {boundary}\n"

        return summary

    def to_dict(self) -> dict:
        return {
            "domains": self.domains,
            "hard_boundaries": list(self.hard_boundaries.keys()),
            "failures": {k: len(v) for k, v in self.failures.items()},
            "successes": {k: len(v) for k, v in self.successes.items()},
        }


# ── Compatibility layer ───────────────────────────────────────────────────────
# Existing integration (agent.py order-check, tools.check_capability) imports
# module-level can_do(task) and check(task). Shares one map instance and
# persists it to memory.json for the live viewer panel.

_map = CapabilityMap()


def _persist_map() -> None:
    try:
        from tools import _load_memory, _save_memory_file  # lazy — avoids circular import
        mem = _load_memory()
        mem["capability_map"] = _map.to_dict()
        _save_memory_file(mem)
    except Exception:
        pass


def can_do(task: str) -> Tuple[bool, float, str]:
    return _map.can_do(task)


def check(task: str) -> str:
    ok, confidence, reason = _map.can_do(task)
    _persist_map()
    return json.dumps({
        "task": task,
        "can_do": ok,
        "confidence": round(confidence, 2),
        "reason": reason,
        "workaround": "" if ok else _map.suggest_workaround(task),
    }, indent=2)


def record_success(task: str, domain: str) -> None:
    _map.record_success(task, domain)
    _persist_map()


def record_failure(task: str, domain: str, reason: str = "") -> None:
    _map.record_failure(task, domain, reason)
    _persist_map()
