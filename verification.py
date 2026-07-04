"""
verification.py — PinPoint Verifier module.

Provides lightweight LLM-based sanity-checking for tool calls and code snippets.
All LLM / memory imports are lazy (inside methods) to avoid circular imports.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Tools whose results are trusted implicitly — no LLM round-trip needed.
_SKIP_VERIFICATION_TOOLS = frozenset({"think", "speak", "done", "brainstorm"})


@dataclass
class VerificationResult:
    """Result of a single verification pass."""
    passed: bool
    confidence: float          # 0.0 – 1.0
    issues: list[str] = field(default_factory=list)
    suggestion: str = ""

    def __post_init__(self) -> None:
        # Clamp confidence to valid range.
        self.confidence = max(0.0, min(1.0, float(self.confidence)))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_llm_json(raw: str) -> Optional[dict]:
    """
    Extract the first JSON object from an LLM response string.
    Returns None if nothing parseable is found.
    """
    raw = raw.strip()
    # Try the whole string first (common case: model returns bare JSON).
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Fall back: find the outermost {...} block.
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


def _result_from_dict(data: dict) -> VerificationResult:
    """Build a VerificationResult from the parsed LLM JSON payload."""
    passed = bool(data.get("passed", True))
    raw_conf = data.get("confidence", 50)
    # Accept both 0-100 and 0-1 scales.
    confidence = float(raw_conf) / 100.0 if float(raw_conf) > 1.0 else float(raw_conf)
    issues_raw = data.get("issues", [])
    if isinstance(issues_raw, str):
        issues = [issues_raw] if issues_raw else []
    else:
        issues = [str(i) for i in issues_raw if i]
    suggestion = str(data.get("suggestion", "") or "")
    return VerificationResult(passed=passed, confidence=confidence, issues=issues, suggestion=suggestion)


_FALLBACK_PASS = VerificationResult(passed=True, confidence=0.5, issues=[], suggestion="")


# ---------------------------------------------------------------------------
# Verifier class
# ---------------------------------------------------------------------------

class Verifier:
    """
    Lightweight wrapper around the agent's LLM that checks tool results and
    code snippets for obvious problems.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_action(
        self,
        tool_name: str,
        tool_input: dict,
        result: str,
        context: str = "",
    ) -> VerificationResult:
        """
        Ask the LLM whether a tool call succeeded and if the result looks correct.

        Tools in _SKIP_VERIFICATION_TOOLS get an automatic high-confidence pass
        to avoid unnecessary LLM round-trips for purely conversational actions.
        """
        if tool_name in _SKIP_VERIFICATION_TOOLS:
            return VerificationResult(passed=True, confidence=0.95, issues=[], suggestion="")

        prompt = self._build_action_prompt(tool_name, tool_input, result, context)
        raw = self._call_llm(prompt, max_tokens=300)
        if raw is None:
            logger.warning("verify_action: LLM call failed for tool=%s", tool_name)
            return _FALLBACK_PASS

        data = _parse_llm_json(raw)
        if data is None:
            logger.debug("verify_action: could not parse JSON from LLM response for tool=%s", tool_name)
            return _FALLBACK_PASS

        vr = _result_from_dict(data)
        self.log_result(tool_name, vr)
        return vr

    def verify_code(self, code: str, language: str = "python") -> VerificationResult:
        """
        Ask the LLM to review a code snippet for syntax errors, bad imports,
        logic bugs, and edge-case problems.
        """
        prompt = (
            f"Review the following {language} code snippet for issues.\n"
            "Check for: syntax errors, missing or incorrect imports, logic bugs, "
            "and potential edge-case problems.\n\n"
            f"```{language}\n{code}\n```\n\n"
            "Respond ONLY with a JSON object in this exact format:\n"
            '{"passed": <true|false>, "confidence": <0-100>, '
            '"issues": ["<issue1>", ...], "suggestion": "<one-line fix or empty>"}\n'
            "Do not include any text outside the JSON object."
        )
        raw = self._call_llm(prompt, max_tokens=400)
        if raw is None:
            logger.warning("verify_code: LLM call failed")
            return _FALLBACK_PASS

        data = _parse_llm_json(raw)
        if data is None:
            logger.debug("verify_code: could not parse JSON from LLM response")
            return _FALLBACK_PASS

        vr = _result_from_dict(data)
        self.log_result(f"verify_code[{language}]", vr)
        return vr

    def log_result(self, tool_name: str, result: VerificationResult) -> None:
        """
        Append an entry to memory["verification_log"], keeping only the last 100.
        Silently swallows any persistence errors so verification never crashes the agent.
        """
        try:
            from tools import _load_memory, _save_memory_file  # lazy import

            memory = _load_memory()
            log: list = memory.setdefault("verification_log", [])

            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "tool": tool_name,
                "passed": result.passed,
                "confidence": round(result.confidence, 3),
                "issues": result.issues,
                "suggestion": result.suggestion,
            }
            log.append(entry)

            # Keep only the most recent 100 entries.
            if len(log) > 100:
                memory["verification_log"] = log[-100:]

            _save_memory_file(memory)
        except Exception as exc:  # noqa: BLE001
            logger.debug("log_result: failed to persist verification log: %s", exc)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_action_prompt(
        self,
        tool_name: str,
        tool_input: dict,
        result: str,
        context: str,
    ) -> str:
        """Construct the short verification prompt for a tool action."""
        input_summary = json.dumps(tool_input, ensure_ascii=False)
        if len(input_summary) > 400:
            input_summary = input_summary[:400] + "…"
        result_summary = result if isinstance(result, str) else str(result)
        if len(result_summary) > 600:
            result_summary = result_summary[:600] + "…"

        context_line = f"\nContext: {context}" if context else ""

        return (
            f"A tool was just called. Did it succeed and does the result look correct?{context_line}\n\n"
            f"Tool: {tool_name}\n"
            f"Input: {input_summary}\n"
            f"Result: {result_summary}\n\n"
            "Respond ONLY with a JSON object in this exact format:\n"
            '{"passed": <true|false>, "confidence": <0-100>, '
            '"issues": ["<issue1>", ...], "suggestion": "<one-line suggestion or empty>"}\n'
            "Do not include any text outside the JSON object."
        )

    def _call_llm(self, prompt: str, max_tokens: int = 300) -> Optional[str]:
        """
        Send a single user message to the LLM and return the raw text response.
        Returns None on any error so callers can fall back gracefully.
        """
        try:
            from agent import AGENT_MODEL, get_llm_client, chat_completion  # lazy import

            client = get_llm_client(timeout=30.0)
            messages = [{"role": "user", "content": prompt}]
            response = chat_completion(
                client,
                messages,
                temperature=0.0,
                max_tokens=max_tokens,
                model=AGENT_MODEL,
            )
            # chat_completion returns an OpenAI-compatible object.
            content = response.choices[0].message.content
            return content.strip() if content else None
        except Exception as exc:  # noqa: BLE001
            logger.debug("_call_llm error: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Module-level singleton + convenience function
# ---------------------------------------------------------------------------

_verifier = Verifier()


def verify(
    tool_name: str,
    tool_input: dict,
    result: str,
    context: str = "",
) -> VerificationResult:
    """
    Module-level convenience wrapper around the shared Verifier singleton.

    Usage:
        from verification import verify
        vr = verify("read_file", {"path": "/tmp/foo.txt"}, file_contents)
        if not vr.passed:
            print(vr.issues)
    """
    return _verifier.verify_action(tool_name, tool_input, result, context)
