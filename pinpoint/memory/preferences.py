"""User preference learning.

Preferences are learned from what the user *says*, not from what the agent
notices. "Always use Modrinth" is a preference; "CJ built three games in a row
so he must prefer games" is a guess, and guesses about a person do not belong
in durable memory.

Every preference records source, confidence, timestamp and scope. Only
preferences whose source is the user can translate into a permission grant,
and that grant still goes through the policy engine's human path — the agent
cannot manufacture one by writing the right sentence into its own transcript.
"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from pinpoint import jsonstore, paths

USER = "user"          # said by the human — carries authority
AGENT = "agent"        # noticed by the agent — a suggestion at best

# Scopes a preference can apply to.
SCOPE_GLOBAL = "global"
SCOPE_TOOL = "tool"
SCOPE_TOPIC = "topic"


@dataclass
class Preference:
    """One stated preference."""
    statement: str
    key: str = ""
    value: str = ""
    scope: str = SCOPE_GLOBAL
    subject: str = ""              # the tool or topic it applies to
    source: str = USER
    confidence: float = 0.8
    grants_autonomy: bool = False  # "don't ask before X"
    requires_approval: bool = False  # "always ask before X"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict) -> "Preference":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def render(self) -> str:
        return (f"{self.statement} "
                f"[{self.scope}{'/' + self.subject if self.subject else ''}, "
                f"{self.source}, {self.confidence:.0%}]")


# Tools a natural-language phrase may refer to.
_TOOL_WORDS = {
    "test": "run_tests", "tests": "run_tests", "testing": "run_tests",
    "install": "pip_install", "installing": "pip_install",
    "message": "send_message", "messages": "send_message",
    "text": "send_message", "texting": "send_message", "texts": "send_message",
    "call": "make_call", "calls": "make_call", "calling": "make_call",
    "email": "send_email", "emails": "send_email",
    "delete": "delete_file", "deleting": "delete_file",
    "shell": "run_shell", "commands": "run_shell",
    "commit": "git_commit", "committing": "git_commit",
}

# "don't ask me before running tests" / "just run the tests without asking"
_AUTONOMY = re.compile(
    r"\b(?:don'?t|do not|no need to|never)\s+ask(?:\s+me)?\s+"
    r"(?:before|about|for permission (?:to|before))?\s*([\w\s]{2,40})", re.I)
_AUTONOMY_ALT = re.compile(r"\b(?:just|feel free to)\s+([\w\s]{2,40}?)\s+"
                           r"without asking\b", re.I)

# "never send messages without asking" / "always ask before you delete anything"
_APPROVAL = re.compile(
    r"\b(?:always ask|ask me|check with me|confirm with me)\s+"
    r"(?:first\s+)?(?:before|about)\s+([\w\s]{2,40})", re.I)
_APPROVAL_ALT = re.compile(
    r"\bnever\s+([\w\s]{2,40}?)\s+without\s+(?:asking|checking|permission)\b", re.I)

# "always use Modrinth when possible" / "prefer TypeScript"
_GENERAL = [
    re.compile(r"\balways\s+(use|prefer|start with|check)\s+([\w .+#-]{2,40})", re.I),
    re.compile(r"\bi\s+prefer\s+([\w .+#-]{2,40})", re.I),
    re.compile(r"\bnever\s+use\s+([\w .+#-]{2,40})", re.I),
    re.compile(r"\bstop\s+(?:using|doing)\s+([\w .+#-]{2,40})", re.I),
]


def _subject_tool(phrase: str) -> str:
    """Map a phrase like 'running installs' to the tool it constrains."""
    for word in re.findall(r"[a-z]+", (phrase or "").lower()):
        if word in _TOOL_WORDS:
            return _TOOL_WORDS[word]
        if word.endswith("s") and word[:-1] in _TOOL_WORDS:
            return _TOOL_WORDS[word[:-1]]
    return ""


class PreferenceStore:
    """Learned preferences, persisted with full provenance."""

    def __init__(self, path: str = ""):
        self._path = path or paths.output_dir("preferences.json")
        self._items: List[Preference] = []
        self._loaded = False

    def load(self) -> "PreferenceStore":
        if not self._loaded:
            data = jsonstore.read_json(self._path, {"preferences": []})
            self._items = [Preference.from_dict(d) for d in data.get("preferences", [])]
            self._loaded = True
        return self

    def save(self) -> bool:
        return jsonstore.write_json(
            self._path, {"preferences": [p.to_dict() for p in self._items]})

    # ── learning ─────────────────────────────────────────────────────────────

    def learn(self, text: str, source: str = USER) -> List[Preference]:
        """Extract explicitly stated preferences. Nothing is inferred."""
        found: List[Preference] = []
        blob = text or ""
        confidence = 0.9 if source == USER else 0.4

        # "don't ask me before running tests" literally contains "ask me before
        # running tests", so autonomy phrases are matched first and masked out
        # before the approval patterns get a look at the text.
        remaining = blob
        for pattern in (_AUTONOMY, _AUTONOMY_ALT):
            for match in pattern.finditer(remaining):
                phrase = match.group(1).strip()
                subject = _subject_tool(phrase)
                found.append(Preference(
                    statement=match.group(0).strip(), key="grants_autonomy",
                    value=phrase, scope=SCOPE_TOOL if subject else SCOPE_GLOBAL,
                    subject=subject, source=source, confidence=confidence,
                    grants_autonomy=True))
            remaining = pattern.sub(" ", remaining)

        for pattern in (_APPROVAL, _APPROVAL_ALT):
            for match in pattern.finditer(remaining):
                phrase = match.group(1).strip()
                subject = _subject_tool(phrase)
                found.append(Preference(
                    statement=match.group(0).strip(), key="requires_approval",
                    value=phrase, scope=SCOPE_TOOL if subject else SCOPE_GLOBAL,
                    subject=subject, source=source, confidence=confidence,
                    requires_approval=True))

        for pattern in _GENERAL:
            for match in pattern.finditer(blob):
                value = match.groups()[-1].strip()
                found.append(Preference(
                    statement=match.group(0).strip(), key="style", value=value,
                    scope=SCOPE_TOPIC, subject=value.lower()[:40], source=source,
                    confidence=confidence - 0.05))

        self.load()
        for preference in found:
            self._replace_conflicting(preference)
            self._items.append(preference)
        if found:
            self.save()
        return found

    def _replace_conflicting(self, new: Preference) -> None:
        """A newer statement about the same subject supersedes the older one."""
        self._items = [
            p for p in self._items
            if not (p.subject == new.subject and p.scope == new.scope
                    and (p.grants_autonomy or p.requires_approval)
                    == (new.grants_autonomy or new.requires_approval)
                    and p.key == new.key)
        ]

    # ── reading ──────────────────────────────────────────────────────────────

    def all(self) -> List[Preference]:
        self.load()
        return list(self._items)

    def for_tool(self, tool: str) -> List[Preference]:
        return [p for p in self.all() if p.subject == tool]

    def wants_approval_for(self, tool: str) -> bool:
        return any(p.requires_approval and p.subject == tool for p in self.all())

    def grants_autonomy_for(self, tool: str) -> bool:
        """True only when the user themselves said so, and didn't later retract it."""
        relevant = [p for p in self.all() if p.subject == tool and p.source == USER]
        if not relevant:
            return False
        latest = max(relevant, key=lambda p: p.timestamp)
        return latest.grants_autonomy

    def style_notes(self, limit: int = 6) -> List[str]:
        return [p.statement for p in self.all() if p.key == "style"][-limit:]

    def context_block(self) -> str:
        items = self.all()
        if not items:
            return ""
        lines = ["Preferences CJ has stated:"]
        for preference in items[-8:]:
            lines.append(f"  - {preference.render()}")
        return "\n".join(lines)

    # ── acting on preferences ────────────────────────────────────────────────

    def apply_permissions(self) -> Dict[str, str]:
        """Turn user-stated autonomy into real grants, and asks into real rules.

        Only ``source == USER`` preferences do anything here: text the user
        typed is human authority, text the agent produced is not.
        """
        from pinpoint.security import permissions

        applied: Dict[str, str] = {}
        for preference in self.all():
            if preference.source != USER or not preference.subject:
                continue
            if preference.grants_autonomy and self.grants_autonomy_for(preference.subject):
                if permissions.grant(preference.subject, source=permissions.HUMAN,
                                     scope="always",
                                     note=f"stated preference: {preference.statement[:80]}"):
                    applied[preference.subject] = "granted"
            elif preference.requires_approval:
                permissions.revoke(preference.subject)
                permissions.set_custom_level(preference.subject, permissions.YELLOW,
                                             source=permissions.HUMAN)
                applied[preference.subject] = "always ask"
        return applied

    def forget(self, preference_id: str) -> bool:
        self.load()
        before = len(self._items)
        self._items = [p for p in self._items if p.id != preference_id]
        if len(self._items) != before:
            self.save()
            return True
        return False


_default: Optional[PreferenceStore] = None


def preferences() -> PreferenceStore:
    """Shared instance, rebuilt when PINPOINT_HOME changes (tests)."""
    global _default
    expected = paths.output_dir("preferences.json")
    if _default is None or _default._path != expected:
        _default = PreferenceStore(expected)
    return _default
