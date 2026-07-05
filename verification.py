# verification.py — Phase 2: Self-Verification Loops

import json
from typing import Dict, List, Tuple
from datetime import datetime

class VerificationResult:
    """Result of verifying an action or output."""
    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        self.passed: bool = True
        self.confidence: float = 1.0
        self.issues: List[str] = []
        self.suggestions: List[str] = []
        self.timestamp = datetime.now().isoformat()

    @property
    def suggestion(self) -> str:
        """First suggestion, for callers expecting a single string."""
        return self.suggestions[0] if self.suggestions else ""

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "tool": self.tool_name,  # viewer panel reads .tool
            "passed": self.passed,
            "confidence": self.confidence,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "timestamp": self.timestamp,
        }

class Verifier:
    """Verify actions before and after execution."""

    def __init__(self):
        self.verification_log: List[VerificationResult] = []
        self.domain_confidence = {
            "write_file": 0.95,
            "read_file": 0.95,
            "code_writing": 0.92,
            "run_python": 0.85,
            "file_operations": 0.92,
            "reasoning": 0.88,
            "system_commands": 0.65,
            "self_modification": 0.70,
            "web_search": 0.75,
            "web_fetch": 0.75,
        }
        self.tool_failure_rates = {}  # track what fails

    def verify_tool_result(self, tool_name: str, result: str) -> VerificationResult:
        """Check if a tool result looks successful. Catches errors."""
        vr = VerificationResult(tool_name)

        # Check for error patterns
        error_markers = [
            "error:", "error,", "failed", "traceback", "exception:",
            "syntax error", "permission denied", "not found", "could not",
            "exception", "failed to", "unable to", "invalid"
        ]

        result_lower = result.lower()
        found_errors = [m for m in error_markers if m in result_lower]

        if found_errors:
            vr.passed = False
            vr.confidence = 0.2
            vr.issues.append(f"Error markers found: {found_errors[0]}")
            vr.suggestions.append("Review the error message. Root cause is likely in the output above.")

        # Trust level by domain
        elif tool_name in self.domain_confidence:
            vr.confidence = self.domain_confidence[tool_name]
        else:
            vr.confidence = 0.75

        # Check for suspiciously short results
        if len(result.strip()) < 5 and tool_name not in ("done", "think", "speak"):
            vr.confidence -= 0.15
            vr.issues.append("Result is unexpectedly brief")
            vr.suggestions.append("Check if the tool succeeded or if output was truncated")

        # Check for common failure patterns
        if "no such file" in result_lower or "does not exist" in result_lower:
            vr.confidence -= 0.3
            vr.issues.append("File path may be incorrect")

        if "permission" in result_lower or "denied" in result_lower:
            vr.confidence -= 0.2
            vr.issues.append("Permission issue detected")

        # Log it
        self.verification_log.append(vr)

        # Track failure rates
        if tool_name not in self.tool_failure_rates:
            self.tool_failure_rates[tool_name] = {"passed": 0, "failed": 0}

        if vr.passed:
            self.tool_failure_rates[tool_name]["passed"] += 1
        else:
            self.tool_failure_rates[tool_name]["failed"] += 1

        return vr

    def get_verification_stats(self) -> dict:
        """Summary of verification performance."""
        if not self.verification_log:
            return {"total_verifications": 0}

        passed = sum(1 for vr in self.verification_log if vr.passed)
        total = len(self.verification_log)
        avg_confidence = sum(vr.confidence for vr in self.verification_log) / total

        return {
            "total_verifications": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": (passed / total * 100) if total > 0 else 0,
            "average_confidence": avg_confidence,
            "most_trusted_tools": sorted(
                self.domain_confidence.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }

    def get_tool_reliability(self, tool_name: str) -> Tuple[float, int]:
        """Get reliability percentage for a specific tool based on history."""
        if tool_name not in self.tool_failure_rates:
            return (self.domain_confidence.get(tool_name, 0.75), 0)

        stats = self.tool_failure_rates[tool_name]
        total = stats["passed"] + stats["failed"]
        if total == 0:
            return (self.domain_confidence.get(tool_name, 0.75), 0)

        reliability = (stats["passed"] / total)
        return (reliability, total)

    def to_memory_format(self) -> dict:
        """Export for saving to memory.json."""
        return {
            "verification_log": [vr.to_dict() for vr in self.verification_log[-50:]],  # Last 50
            "stats": self.get_verification_stats(),
            "tool_reliability": {
                t: {"reliability": r, "samples": s}
                for t, (r, s) in [
                    (tn, self.get_tool_reliability(tn))
                    for tn in self.tool_failure_rates.keys()
                ]
            }
        }

    def to_markdown(self) -> str:
        """Render stats as markdown."""
        stats = self.get_verification_stats()

        md = f"""# Verification Statistics

Total Checks: {stats['total_verifications']}
Passed: {stats.get('passed', 0)}
Failed: {stats.get('failed', 0)}
Success Rate: {stats.get('success_rate', 0):.1f}%
Average Confidence: {stats.get('average_confidence', 0):.0%}

## Tool Reliability
"""
        for tool, conf in stats.get('most_trusted_tools', []):
            md += f"- {tool}: {conf:.0%}\n"

        return md


# ── Compatibility layer ───────────────────────────────────────────────────────
# Existing integration (agent.py post-dispatch hook, tools.verify_last_action)
# imports verify(tool_name, tool_input, result, context). Delegates to a shared
# Verifier and persists the log to memory.json for the live viewer panel.

_verifier = Verifier()


def _persist_log() -> None:
    try:
        from tools import _load_memory, _save_memory_file  # lazy — avoids circular import
        mem = _load_memory()
        mem["verification_log"] = [vr.to_dict() for vr in _verifier.verification_log[-100:]]
        stats = _verifier.get_verification_stats()
        mem["verification_patterns"] = {
            "verification_accuracy": stats.get("average_confidence", 0),
            "success_rate": stats.get("success_rate", 0),
            "tools_with_high_failure_rates": [
                t for t, s in _verifier.tool_failure_rates.items()
                if (s["passed"] + s["failed"]) >= 3 and s["failed"] / (s["passed"] + s["failed"]) > 0.3
            ],
        }
        _save_memory_file(mem)
    except Exception:
        pass


def verify(tool_name: str, tool_input: dict, result: str, context: str = "") -> VerificationResult:
    """Verify a tool result and log it. Returns VerificationResult."""
    vr = _verifier.verify_tool_result(tool_name, result or "")
    _persist_log()
    return vr


def get_verification_stats() -> dict:
    return _verifier.get_verification_stats()
