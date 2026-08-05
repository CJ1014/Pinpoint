"""Intent engine — natural language in, structured objective out.

Parsing is deterministic: no LLM call happens here, because turning "get my
server running again" into a goal record should not cost a round trip or vary
between runs. An optional refiner can enrich the result when a model is
already in the loop, but the structure never depends on it.

The hard rule is that unknowns stay unknown. If the user says "text John" and
there is no John on file, the objective records ``John`` as an unresolved
entity — it does not invent a phone number, and it does not quietly pick the
first contact that starts with J.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ── Objective kinds ───────────────────────────────────────────────────────────
FIX = "fix"
BUILD = "build"
INVESTIGATE = "investigate"
RESEARCH = "research"
MONITOR = "monitor"
COMMUNICATE = "communicate"
SCHEDULE = "schedule"
MAINTAIN = "maintain"
UNKNOWN = "unknown"

PRIORITY_LOW, PRIORITY_NORMAL, PRIORITY_HIGH, PRIORITY_URGENT = (
    "low", "normal", "high", "urgent")
AUTONOMY_LOW, AUTONOMY_NORMAL, AUTONOMY_HIGH = "low", "normal", "high"


@dataclass
class Objective:
    """A user request, made explicit enough to plan against."""
    goal: str
    raw: str = ""
    kind: str = UNKNOWN
    priority: str = PRIORITY_NORMAL
    autonomy: str = AUTONOMY_NORMAL
    constraints: List[str] = field(default_factory=list)
    success_conditions: List[str] = field(default_factory=list)
    failure_conditions: List[str] = field(default_factory=list)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    unknowns: List[str] = field(default_factory=list)
    deadline: str = ""
    long_running: bool = False
    until_condition: str = ""

    def to_dict(self) -> dict:
        return {
            "goal": self.goal, "raw": self.raw, "kind": self.kind,
            "priority": self.priority, "autonomy": self.autonomy,
            "constraints": self.constraints,
            "success_conditions": self.success_conditions,
            "failure_conditions": self.failure_conditions,
            "entities": self.entities, "unknowns": self.unknowns,
            "deadline": self.deadline, "long_running": self.long_running,
            "until_condition": self.until_condition,
        }

    def render(self) -> str:
        lines = [f"Objective : {self.goal}",
                 f"Kind      : {self.kind}",
                 f"Priority  : {self.priority}   Autonomy: {self.autonomy}"]
        if self.deadline:
            lines.append(f"Deadline  : {self.deadline}")
        if self.constraints:
            lines.append("Constraints:")
            lines += [f"  - {c}" for c in self.constraints]
        if self.success_conditions:
            lines.append("Success when:")
            lines += [f"  - {c}" for c in self.success_conditions]
        if self.unknowns:
            lines.append("Unknown (must not be guessed):")
            lines += [f"  - {u}" for u in self.unknowns]
        return "\n".join(lines)


# ── Keyword tables ────────────────────────────────────────────────────────────

_KIND_PATTERNS = [
    (COMMUNICATE, r"\b(text|message|sms|call|phone|email|e-mail|dm|reply to|"
                  r"tell\s+\w+\s+that|let\s+\w+\s+know)\b"),
    (MONITOR, r"\b(monitor|keep an eye|watch|stay on top of|alert me|notify me|"
              r"if it crashes|when it goes down)\b"),
    (SCHEDULE, r"\b(remind me|schedule|at \d{1,2}\s*(am|pm)|tomorrow|every day|"
               r"each morning|later today)\b"),
    (FIX, r"\b(fix|repair|broken|won'?t (build|start|run|compile)|failing|"
          r"crash(ed|ing)?|debug|get .* (running|working) again|restore|"
          r"resolve|unbreak)\b"),
    (INVESTIGATE, r"\b(why|investigate|diagnose|figure out|find out why|"
                  r"what'?s wrong|root cause|troubleshoot)\b"),
    (BUILD, r"\b(build|create|make|write|implement|add|generate|design|"
            r"set up|scaffold)\b"),
    (RESEARCH, r"\b(research|look up|find the latest|search for|read up on|"
               r"documentation for|compare)\b"),
    (MAINTAIN, r"\b(update|upgrade|refactor|clean up|migrate|bump|tidy)\b"),
]

_SUCCESS_BY_KIND = {
    FIX: ["the failure is reproduced", "the root cause is identified",
          "a fix is applied", "the original failure no longer occurs"],
    BUILD: ["the artifact exists on disk", "it runs without errors",
            "its behaviour matches the request"],
    INVESTIGATE: ["evidence is gathered from the actual system",
                  "the explanation is supported by that evidence",
                  "findings are reported"],
    RESEARCH: ["relevant sources are read", "findings are synthesised",
               "sources are cited"],
    MONITOR: ["the monitor is active", "relevant events trigger a response",
              "the user is told what happened"],
    COMMUNICATE: ["the recipient is unambiguously resolved",
                  "the message content is confirmed",
                  "the provider confirms delivery"],
    SCHEDULE: ["the reminder or task is scheduled",
               "it is confirmed back to the user"],
    MAINTAIN: ["the change is applied", "existing tests still pass"],
    UNKNOWN: ["the requested outcome is achieved and verified"],
}

_FAILURE_BY_KIND = {
    FIX: ["the root cause cannot be identified after the attempt budget",
          "the fix requires access PinPoint does not have"],
    COMMUNICATE: ["the recipient is ambiguous or unknown",
                  "no authorised provider is configured",
                  "the provider rejects the send"],
    MONITOR: ["the target cannot be observed"],
    UNKNOWN: ["the attempt budget is exhausted without progress"],
}

_URGENT = re.compile(r"\b(urgent|asap|right now|immediately|emergency|critical)\b", re.I)
_HIGH = re.compile(r"\b(soon|today|quickly|priority|important)\b", re.I)
_LOW = re.compile(r"\b(when you (can|get a chance)|no rush|sometime|eventually|"
                  r"whenever)\b", re.I)

_HIGH_AUTONOMY = re.compile(
    r"\b(keep (working|going)|don'?t ask me|without asking|on your own|"
    r"autonomous(ly)?|just do it|until (it works|the tests pass|it'?s done))\b", re.I)
_LOW_AUTONOMY = re.compile(
    r"\b(ask me (first|before)|check with me|let me know before|confirm with me|"
    r"run it by me)\b", re.I)

_UNTIL = re.compile(r"\buntil\s+(.+?)(?:[.;]|$)", re.I)
_DEADLINE = re.compile(
    r"\b(by|before|at)\s+((?:\d{1,2}(?::\d{2})?\s*(?:am|pm))|tomorrow[^.,;]*|"
    r"tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"\d{4}-\d{2}-\d{2})", re.I)

_CONSTRAINT_PATTERNS = [
    re.compile(r"\b(?:don'?t|do not|never)\s+([^.,;]{3,80})", re.I),
    re.compile(r"\bwithout\s+([^.,;]{3,60})", re.I),
    re.compile(r"\bonly\s+(?:use|if)\s+([^.,;]{3,60})", re.I),
    re.compile(r"\bmake sure (?:you|to)\s+([^.,;]{3,80})", re.I),
    re.compile(r"\balways\s+(?:use\s+)?([^.,;]{3,60})", re.I),
]

_PATH = re.compile(r"(?:^|\s)((?:[A-Za-z]:\\|~?/|\./)[\w./\\-]{2,}|[\w-]+\.[a-z]{2,4})(?=\s|$|[.,;])")
_URL = re.compile(r"https?://[^\s]+")
_PORT = re.compile(r"\bport\s+(\d{2,5})\b", re.I)
# A capitalised name after a communication verb — the intended recipient.
# The verb is matched case-insensitively; the name still has to be capitalised,
# so "text john" stays an unknown recipient rather than a confident guess.
_RECIPIENT = re.compile(
    r"\b(?i:text|message|call|phone|email|e-mail|dm|remind|tell|ask)\s+"
    r"(?!me\b|him\b|her\b|them\b|us\b)([A-Z][a-zA-Z]{1,20}(?:\s+[A-Z][a-zA-Z]{1,20})?)")
_MY_THING = re.compile(r"\bmy\s+([a-z][\w -]{2,30}?)(?=\s|$|[.,;])", re.I)


def _detect_kind(text: str) -> str:
    for kind, pattern in _KIND_PATTERNS:
        if re.search(pattern, text, re.I):
            return kind
    return UNKNOWN


def _detect_priority(text: str) -> str:
    if _URGENT.search(text):
        return PRIORITY_URGENT
    if _LOW.search(text):
        return PRIORITY_LOW
    if _HIGH.search(text):
        return PRIORITY_HIGH
    return PRIORITY_NORMAL


def _detect_autonomy(text: str) -> str:
    # An explicit "ask me first" wins over a general "keep going".
    if _LOW_AUTONOMY.search(text):
        return AUTONOMY_LOW
    if _HIGH_AUTONOMY.search(text):
        return AUTONOMY_HIGH
    return AUTONOMY_NORMAL


def _extract_constraints(text: str) -> List[str]:
    found: List[str] = []
    for pattern in _CONSTRAINT_PATTERNS:
        for match in pattern.finditer(text):
            phrase = match.group(0).strip()
            phrase = re.sub(r"\s+", " ", phrase)
            if phrase.lower() not in (f.lower() for f in found):
                found.append(phrase)
    return found[:8]


def _extract_entities(text: str) -> Dict[str, List[str]]:
    entities: Dict[str, List[str]] = {}
    urls = _URL.findall(text)
    if urls:
        entities["urls"] = urls
    paths = [p for p in _PATH.findall(text) if p not in urls]
    if paths:
        entities["paths"] = paths[:6]
    ports = _PORT.findall(text)
    if ports:
        entities["ports"] = ports
    recipients = [r.strip() for r in _RECIPIENT.findall(text)]
    if recipients:
        entities["recipients"] = recipients[:4]
    targets = [t.strip() for t in _MY_THING.findall(text)]
    if targets:
        entities["targets"] = targets[:4]
    return entities


def _extract_unknowns(text: str, kind: str, entities: Dict[str, List[str]]) -> List[str]:
    """Name what the request does not pin down, so it can't be filled in by guesswork."""
    unknowns: List[str] = []
    if kind == COMMUNICATE:
        for name in entities.get("recipients", []):
            unknowns.append(f"which contact '{name}' refers to, and their number/address")
        if not entities.get("recipients"):
            unknowns.append("who the recipient is")
        if not re.search(r"\b(that|saying|tell (?:him|her|them)|:)\b", text, re.I):
            unknowns.append("the exact message content")
    if kind in (FIX, INVESTIGATE) and not entities.get("paths") and not entities.get("targets"):
        unknowns.append("which project or component is affected")
    if kind == MONITOR and not entities.get("targets") and not entities.get("ports"):
        unknowns.append("exactly what should be monitored")
    return unknowns


def parse(text: str) -> Objective:
    """Turn a natural-language request into a structured objective."""
    raw = (text or "").strip()
    goal = re.sub(r"^\s*(pinpoint|hey pinpoint|ok pinpoint)[,:]?\s*", "", raw, flags=re.I)
    goal = goal.strip() or raw

    kind = _detect_kind(goal)
    entities = _extract_entities(goal)
    objective = Objective(
        goal=goal,
        raw=raw,
        kind=kind,
        priority=_detect_priority(goal),
        autonomy=_detect_autonomy(goal),
        constraints=_extract_constraints(goal),
        success_conditions=list(_SUCCESS_BY_KIND.get(kind, _SUCCESS_BY_KIND[UNKNOWN])),
        failure_conditions=list(_FAILURE_BY_KIND.get(kind, _FAILURE_BY_KIND[UNKNOWN])),
        entities=entities,
        unknowns=_extract_unknowns(goal, kind, entities),
    )

    until = _UNTIL.search(goal)
    if until:
        objective.until_condition = until.group(1).strip()
        objective.success_conditions.insert(0, objective.until_condition)
        objective.long_running = True

    deadline = _DEADLINE.search(goal)
    if deadline:
        objective.deadline = deadline.group(2).strip()

    if kind == MONITOR or _HIGH_AUTONOMY.search(goal):
        objective.long_running = objective.long_running or kind == MONITOR

    return objective


def needs_clarification(objective: Objective) -> Optional[str]:
    """The question to ask before starting, or None if it can proceed.

    Only genuinely blocking gaps qualify. Sending a message to an unidentified
    person is blocking; not knowing which file is broken is something the agent
    can go and find out for itself.
    """
    if objective.kind == COMMUNICATE:
        for unknown in objective.unknowns:
            if unknown.startswith("who the recipient"):
                return "Who should I send this to?"
            if unknown.startswith("which contact"):
                name = unknown.split("'")[1] if "'" in unknown else "them"
                return f"Which {name} do you mean?"
        if "the exact message content" in objective.unknowns:
            return "What exactly do you want me to say?"
    return None
