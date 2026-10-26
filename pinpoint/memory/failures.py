"""Failure memory and postmortems.

Every failure is classified into a root-cause category, recorded with the
recovery that was attempted, and turned into a lesson.

Lessons are *hypotheses*, not facts. Each carries a confidence that rises when
it proves useful and falls when it turns out to be wrong, and advice is
rendered with that confidence attached. A single bad diagnosis must not calcify
into something the planner treats as settled.
"""

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from pinpoint import jsonstore, paths

# ── Root-cause categories ─────────────────────────────────────────────────────
MISSING_DEPENDENCY = "missing_dependency"
FILE_NOT_FOUND = "file_not_found"
PERMISSION = "permission"
SYNTAX = "syntax"
TYPE_ERROR = "type_error"
ASSERTION = "assertion"
NETWORK = "network"
PORT_IN_USE = "port_in_use"
AUTH = "auth"
TIMEOUT = "timeout"
DISK = "disk"
CONFIG = "config"
BLOCKED_BY_POLICY = "blocked_by_policy"
CAPABILITY_MISSING = "capability_missing"
UNVERIFIED_EFFECT = "unverified_effect"
UNKNOWN = "unknown"


@dataclass
class CategoryProfile:
    """What a category means, and what usually gets you out of it."""
    category: str
    root_cause: str
    strategies: List[str]
    lesson: str
    preventability: str          # "high" | "medium" | "low"
    prerequisite: str = ""       # work to insert before retrying
    escalate: bool = False       # needs a human rather than another attempt


PROFILES: Dict[str, CategoryProfile] = {
    MISSING_DEPENDENCY: CategoryProfile(
        MISSING_DEPENDENCY, "a required package or command is not installed",
        ["install the missing dependency, then retry",
         "use an alternative that is already available",
         "vendor the functionality instead of depending on it"],
        "check dependencies during planning, before implementation",
        "high", prerequisite="install the missing dependency"),
    FILE_NOT_FOUND: CategoryProfile(
        FILE_NOT_FOUND, "the path referenced does not exist",
        ["list the directory and correct the path",
         "create the missing file or directory first",
         "search the project for the file's real location"],
        "confirm a path exists before acting on it", "high",
        prerequisite="locate the correct path"),
    PERMISSION: CategoryProfile(
        PERMISSION, "the operating system refused access",
        ["work inside a directory that is writable",
         "ask the human to grant the access"],
        "prefer paths PinPoint owns over system locations", "medium",
        escalate=True),
    SYNTAX: CategoryProfile(
        SYNTAX, "the code as written is not valid",
        ["read the reported line and fix the syntax",
         "rewrite the offending section from scratch"],
        "syntax-check generated code before running it", "high"),
    TYPE_ERROR: CategoryProfile(
        TYPE_ERROR, "a value was not the type the code expected",
        ["inspect the actual value and correct the assumption",
         "add a guard for the unexpected type"],
        "verify data shape before consuming it", "high"),
    ASSERTION: CategoryProfile(
        ASSERTION, "a test asserted something the implementation does not do",
        ["read the failing assertion and fix the implementation",
         "check whether the test itself encodes the wrong expectation"],
        "a failing test may be the test's fault — read it before changing code",
        "medium"),
    NETWORK: CategoryProfile(
        NETWORK, "a network request did not complete",
        ["retry once in case it was transient",
         "use a cached or local source instead",
         "continue without the network-dependent part"],
        "network steps need a fallback that does not need the network", "low"),
    PORT_IN_USE: CategoryProfile(
        PORT_IN_USE, "the port is already bound by another process",
        ["use a different port", "identify and stop the process holding it"],
        "check port availability before binding", "high"),
    AUTH: CategoryProfile(
        AUTH, "credentials were missing, invalid, or rejected",
        ["check the credential environment variables are set",
         "report the missing credential to the human"],
        "verify credentials exist before attempting an authenticated call",
        "high", escalate=True),
    TIMEOUT: CategoryProfile(
        TIMEOUT, "the operation did not finish within its budget",
        ["retry with a longer timeout",
         "break the work into smaller pieces"],
        "long operations need a bigger budget or smaller chunks", "medium"),
    DISK: CategoryProfile(
        DISK, "there is no space left on the device",
        ["free space or write somewhere else"],
        "check free space before writing large files", "medium", escalate=True),
    CONFIG: CategoryProfile(
        CONFIG, "configuration is missing or malformed",
        ["read the config and correct it", "regenerate it from a known-good default"],
        "validate configuration before depending on it", "high"),
    BLOCKED_BY_POLICY: CategoryProfile(
        BLOCKED_BY_POLICY, "the action is not permitted",
        ["achieve the goal a way the policy allows",
         "ask the human to authorise it"],
        "check the policy before planning around a restricted action", "high",
        escalate=True),
    CAPABILITY_MISSING: CategoryProfile(
        CAPABILITY_MISSING, "this environment does not have that capability",
        ["use the documented workaround",
         "report the limitation instead of attempting it"],
        "check capability before planning work that depends on it", "high",
        escalate=True),
    UNVERIFIED_EFFECT: CategoryProfile(
        UNVERIFIED_EFFECT, "the action ran but its effect could not be confirmed",
        ["check the effect directly a different way",
         "redo the action and observe it more closely"],
        "an unconfirmed effect is not a completed effect", "medium"),
    UNKNOWN: CategoryProfile(
        UNKNOWN, "the cause is not established",
        ["gather more evidence before trying again",
         "try a different approach entirely"],
        "when the cause is unclear, investigate before retrying", "low"),
}

# Ordered — the first match wins, so specific patterns precede generic ones.
_SIGNATURES = [
    (MISSING_DEPENDENCY, re.compile(
        r"modulenotfounderror|no module named|command not found|"
        r"is not recognized as an internal|cannot find module|"
        r"unable to locate package|importerror", re.I)),
    (PORT_IN_USE, re.compile(r"address already in use|port .{0,10}in use|eaddrinuse", re.I)),
    (PERMISSION, re.compile(r"permission denied|eacces|operation not permitted|"
                            r"access is denied", re.I)),
    (DISK, re.compile(r"no space left|enospc|disk full", re.I)),
    (AUTH, re.compile(r"\b401\b|\b403\b|unauthorized|forbidden|invalid api key|"
                      r"authentication failed|invalid credentials", re.I)),
    (FILE_NOT_FOUND, re.compile(r"no such file|filenotfounderror|does not exist|"
                                r"\b404\b|not found", re.I)),
    (SYNTAX, re.compile(r"syntaxerror|indentationerror|unexpected token|parse error", re.I)),
    (TYPE_ERROR, re.compile(r"typeerror|attributeerror|keyerror|valueerror|"
                            r"nonetype", re.I)),
    (ASSERTION, re.compile(r"assertionerror|assert\b|test(s)? failed|\d+ failed", re.I)),
    (TIMEOUT, re.compile(r"timed out|timeout", re.I)),
    (NETWORK, re.compile(r"connection refused|connection reset|network is unreachable|"
                         r"name or service not known|dns|ssl|econnrefused", re.I)),
    (CONFIG, re.compile(r"invalid config|malformed|could not parse .*(yaml|json|toml)", re.I)),
    (BLOCKED_BY_POLICY, re.compile(r"blocked by guardrail|emergency stop|"
                                   r"declined this action|not permitted", re.I)),
    (CAPABILITY_MISSING, re.compile(r"is not installed|no graphical display|"
                                    r"no .* provider is configured|unavailable here", re.I)),
]


def classify(text: str) -> str:
    """Map an error message to a root-cause category. Deterministic."""
    blob = text or ""
    for category, pattern in _SIGNATURES:
        if pattern.search(blob):
            return category
    return UNKNOWN


def profile_for(category: str) -> CategoryProfile:
    return PROFILES.get(category, PROFILES[UNKNOWN])


@dataclass
class Failure:
    """One postmortem record."""
    goal: str = ""
    task: str = ""
    action: str = ""
    tool: str = ""
    observation: str = ""
    error: str = ""
    category: str = UNKNOWN
    root_cause: str = ""
    recovery: str = ""
    lesson: str = ""
    preventability: str = "low"
    confidence: float = 0.5
    resolved: bool = False
    disputed: int = 0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict) -> "Failure":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def render(self) -> str:
        return (f"Failure     : {self.error[:160]}\n"
                f"Root cause  : {self.root_cause}\n"
                f"Recovery    : {self.recovery or 'none yet'}\n"
                f"Lesson      : {self.lesson}\n"
                f"Confidence  : {self.confidence:.0%}   "
                f"Preventable: {self.preventability}")


class FailureMemory:
    """Persistent store of failures and the lessons drawn from them."""

    # Below this, a lesson has been contradicted too often to keep offering.
    MIN_CONFIDENCE = 0.25

    def __init__(self, path: str = ""):
        self._path = path or paths.output_dir("failures.jsonl")

    # ── writing ──────────────────────────────────────────────────────────────

    def record(self, *, error: str, goal: str = "", task: str = "", tool: str = "",
               action: str = "", observation: str = "",
               category: str = "", recovery: str = "") -> Failure:
        category = category or classify(f"{error} {observation}")
        profile = profile_for(category)
        failure = Failure(
            goal=goal, task=task, action=action, tool=tool,
            observation=observation[:500], error=(error or "")[:500],
            category=category, root_cause=profile.root_cause,
            recovery=recovery, lesson=profile.lesson,
            preventability=profile.preventability,
            confidence=0.5 if category == UNKNOWN else 0.65,
        )
        jsonstore.append_jsonl(self._path, failure.to_dict())
        return failure

    def _rewrite(self, failures: List[Failure]) -> None:
        """Rewrite the log in place — used when a record's confidence changes."""
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        lines = "".join(json.dumps(f.to_dict(), default=str) + "\n" for f in failures)
        temp = self._path + ".rewrite"
        with open(temp, "w", encoding="utf-8") as handle:
            handle.write(lines)
        os.replace(temp, self._path)

    def mark_resolved(self, failure_id: str, recovery: str) -> Optional[Failure]:
        """The recovery worked — the lesson earns confidence."""
        failures = self.all()
        target = next((f for f in failures if f.id == failure_id), None)
        if target is None:
            return None
        target.resolved = True
        target.recovery = recovery
        target.confidence = min(1.0, target.confidence + 0.2)
        self._rewrite(failures)
        return target

    def dispute(self, failure_id: str, reason: str = "") -> Optional[Failure]:
        """The lesson turned out not to hold. Lower its confidence."""
        failures = self.all()
        target = next((f for f in failures if f.id == failure_id), None)
        if target is None:
            return None
        target.disputed += 1
        target.confidence = max(0.0, target.confidence - 0.25)
        if reason:
            target.observation = (target.observation + f" | disputed: {reason}")[:500]
        self._rewrite(failures)
        return target

    # ── reading ──────────────────────────────────────────────────────────────

    def all(self) -> List[Failure]:
        return [Failure.from_dict(d) for d in jsonstore.read_jsonl(self._path)]

    def recent(self, limit: int = 10) -> List[Failure]:
        return self.all()[-limit:]

    def by_category(self, category: str) -> List[Failure]:
        return [f for f in self.all() if f.category == category]

    def similar(self, text: str, limit: int = 5) -> List[Failure]:
        """Past failures whose category matches, most recent first."""
        category = classify(text or "")
        matches = [f for f in self.all() if f.category == category]
        return list(reversed(matches))[:limit]

    def credible_lessons(self, limit: int = 5) -> List[Failure]:
        """Lessons still worth listening to, strongest first."""
        seen = set()
        out: List[Failure] = []
        for failure in sorted(self.all(), key=lambda f: -f.confidence):
            if failure.confidence < self.MIN_CONFIDENCE:
                continue
            if failure.lesson in seen:
                continue
            seen.add(failure.lesson)
            out.append(failure)
            if len(out) >= limit:
                break
        return out

    def advice_for(self, context: str, limit: int = 4) -> List[str]:
        """Lessons to bring into planning — phrased as hypotheses, not facts."""
        blob = (context or "").lower()
        scored: List[tuple] = []
        for failure in self.all():
            if failure.confidence < self.MIN_CONFIDENCE:
                continue
            overlap = 0
            for word in set(re.findall(r"[a-z]{4,}", failure.goal.lower())):
                if word in blob:
                    overlap += 1
            scored.append((overlap, failure.confidence, failure))

        scored.sort(key=lambda item: (-item[0], -item[1]))
        out: List[str] = []
        seen = set()
        for _, _, failure in scored:
            if failure.lesson in seen:
                continue
            seen.add(failure.lesson)
            out.append(f"{failure.lesson} (from a past {failure.category} failure, "
                       f"confidence {failure.confidence:.0%} — check it still applies)")
            if len(out) >= limit:
                break
        return out

    def stats(self) -> dict:
        failures = self.all()
        by_category: Dict[str, int] = {}
        for failure in failures:
            by_category[failure.category] = by_category.get(failure.category, 0) + 1
        return {
            "total": len(failures),
            "resolved": sum(1 for f in failures if f.resolved),
            "disputed": sum(1 for f in failures if f.disputed),
            "by_category": dict(sorted(by_category.items(), key=lambda kv: -kv[1])),
            "most_common": max(by_category, key=by_category.get) if by_category else "",
        }


_default: Optional[FailureMemory] = None


def memory() -> FailureMemory:
    """Shared instance, rebuilt when PINPOINT_HOME changes (tests)."""
    global _default
    expected = paths.output_dir("failures.jsonl")
    if _default is None or _default._path != expected:
        _default = FailureMemory(expected)
    return _default
