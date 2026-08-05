"""Autonomy policy engine.

Three levels:

* **GREEN**  — run autonomously (reads, inspections, reversible local work).
* **YELLOW** — needs human approval, unless the active profile or a standing
  human grant covers it.
* **RED**    — hard blocked. No profile, no grant, and no amount of agent
  reasoning unblocks a RED action.

Classification is a pure function of ``(tool_name, params)`` plus stored human
grants. The model's own words never enter into it, so "I've decided this is
fine" changes nothing. Profile changes and grants require ``source=HUMAN``.
"""

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pinpoint import jsonstore, paths

GREEN = "GREEN"
YELLOW = "YELLOW"
RED = "RED"

SAFE = "SAFE"
ASSISTED = "ASSISTED"
AUTONOMOUS = "AUTONOMOUS"
CUSTOM = "CUSTOM"
PROFILES = (SAFE, ASSISTED, AUTONOMOUS, CUSTOM)

HUMAN = "human"

DEFAULT_PROFILE = os.environ.get("PINPOINT_PROFILE", ASSISTED).strip().upper()


# ── RED: things that are never allowed ────────────────────────────────────────

# Files that implement PinPoint's own oversight. Editing them from inside the
# agent is how a system removes its own brakes, so it is refused outright.
PROTECTED_PATH_PARTS = (
    os.path.join("pinpoint", "security"),
    "emergency_stop.py",
    "permissions.py",
    "approval.py",
    "audit.py",
    "EMERGENCY_STOP.json",
)

# Shell patterns that are destructive-without-authorization or bypass controls.
_RED_SHELL_PATTERNS = [
    (re.compile(r"\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+(/|/\*|~|\$HOME)(\s|$)"),
     "recursive delete of a filesystem root or home directory"),
    (re.compile(r"\bmkfs(\.[a-z0-9]+)?\b"), "filesystem format"),
    (re.compile(r"\bdd\b[^\n]*\bof=/dev/[sn]"), "raw write to a block device"),
    (re.compile(r":\(\)\s*\{.*\|.*&.*\}\s*;?\s*:"), "fork bomb"),
    (re.compile(r"\bchmod\s+-R\s+777\s+/(\s|$)"), "world-writable filesystem root"),
    (re.compile(r">\s*/dev/[sn]d[a-z]"), "overwriting a block device"),
    (re.compile(r"\b(shutdown|reboot|halt)\b\s+-"), "host shutdown"),
    # Credential theft
    (re.compile(r"(cat|cp|scp|curl|less|head|tail)[^\n]*(\.ssh/id_|\.aws/credentials|"
                r"\.netrc|id_rsa|id_ed25519)"), "reading private credentials"),
    (re.compile(r"(curl|wget)[^\n]*\b(-d|--data|--upload-file|-T)\b[^\n]*(\.env|"
                r"credentials|id_rsa|secret)"), "exfiltrating credentials"),
    # Disabling PinPoint's own controls
    (re.compile(r"PINPOINT_APPROVAL\s*=\s*allow"), "disabling the approval policy"),
    (re.compile(r"rm[^\n]*EMERGENCY_STOP"), "removing the emergency stop"),
    (re.compile(r"(chattr|chmod)[^\n]*pinpoint/security"), "tampering with the security package"),
]

# Parameter values that indicate an attempt to bypass authentication.
_RED_INTENT_PATTERNS = [
    (re.compile(r"\b(bypass|disable|circumvent|defeat)\b[^\n]{0,40}"
                r"\b(auth|authentication|login|2fa|mfa|security|guardrail|oversight|approval)\b"),
     "bypassing authentication or oversight"),
    (re.compile(r"\b(brute[- ]?force|crack|keylog(ger)?|exfiltrat)\w*\b"),
     "credential attack tooling"),
]


# ── Fallback tool attributes (the registry supersedes these in Stage 3) ───────

_FALLBACK_LEVELS: Dict[str, str] = {
    # GREEN — read-only or reversible local work
    "read_file": GREEN, "read_anywhere": GREEN, "list_files": GREEN,
    "read_own_source": GREEN, "search_web": GREEN, "fetch_url": GREEN,
    "deep_research": GREEN, "get_news": GREEN, "get_system_info": GREEN,
    "recall_memories": GREEN, "list_memory_categories": GREEN, "list_goals": GREEN,
    "think": GREEN, "deep_think": GREEN, "brainstorm": GREEN, "critique": GREEN,
    "decompose": GREEN, "validate_html": GREEN, "check_js": GREEN,
    "run_tests": GREEN, "take_screenshot": GREEN, "see_screen": GREEN,
    "capture_screen": GREEN, "show_dashboard": GREEN, "review_own_work": GREEN,
    "dictionary_lookup": GREEN, "list_experiments": GREEN,
    # YELLOW — writes, execution, external effects
    "write_file": YELLOW, "write_anywhere": YELLOW, "delete_file": YELLOW,
    "run_python": YELLOW, "run_shell": YELLOW, "pip_install": YELLOW,
    "run_gui": YELLOW, "start_server": YELLOW, "open_app": YELLOW,
    "open_url": YELLOW, "type_text": YELLOW, "press_key": YELLOW,
    "git_commit": YELLOW, "modify_own_source": YELLOW,
    "send_message": YELLOW, "make_call": YELLOW, "send_email": YELLOW,
}

# (reversible, external_side_effect, costs_money)
_FALLBACK_ATTRS: Dict[str, tuple] = {
    "write_file": (True, False, False),
    "write_anywhere": (True, False, False),
    "delete_file": (False, False, False),
    "run_python": (True, False, False),
    "run_shell": (False, False, False),
    "pip_install": (False, False, False),
    "modify_own_source": (True, False, False),
    "git_commit": (True, False, False),
    "open_app": (True, False, False),
    "type_text": (False, False, False),
    "press_key": (False, False, False),
    "send_message": (False, True, True),
    "make_call": (False, True, True),
    "send_email": (False, True, False),
}


@dataclass
class Decision:
    """The outcome of a policy check."""
    tool: str
    level: str
    allowed: bool
    requires_approval: bool
    reason: str
    rule: str = ""
    profile: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return not self.allowed and not self.requires_approval

    def to_dict(self) -> dict:
        return {
            "tool": self.tool, "level": self.level, "allowed": self.allowed,
            "requires_approval": self.requires_approval, "reason": self.reason,
            "rule": self.rule, "profile": self.profile,
        }


# ── Persistent state: profile + standing grants ───────────────────────────────

def _state_file() -> str:
    return paths.state_path("pinpoint_permissions.json")


def _load_state() -> dict:
    state = jsonstore.read_json(_state_file(), {})
    state.setdefault("profile", DEFAULT_PROFILE)
    state.setdefault("grants", {})
    state.setdefault("custom_levels", {})
    return state


def _save_state(state: dict) -> None:
    jsonstore.write_json(_state_file(), state)


def get_profile() -> str:
    profile = _load_state().get("profile", DEFAULT_PROFILE)
    return profile if profile in PROFILES else ASSISTED


def set_profile(profile: str, source: str = "") -> bool:
    """Change the autonomy profile. Humans only — the agent cannot promote itself."""
    profile = (profile or "").strip().upper()
    if profile not in PROFILES:
        return False
    if source != HUMAN:
        _audit("profile_change_refused", {"profile": profile, "source": source},
               error="only a human may change the autonomy profile")
        return False
    state = _load_state()
    previous = state.get("profile")
    state["profile"] = profile
    _save_state(state)
    _audit("profile_changed", {"from": previous, "to": profile, "source": source})
    return True


def grant(tool: str, source: str = "", scope: str = "session", note: str = "") -> bool:
    """Record a standing human permission so identical low-risk actions stop asking.

    ``scope`` is ``"session"`` (until cleared) or ``"always"`` (persists).
    """
    if source != HUMAN:
        _audit("grant_refused", {"tool": tool, "source": source},
               error="only a human may grant a standing permission")
        return False
    state = _load_state()
    state["grants"][tool] = {
        "scope": scope,
        "note": note,
        "granted_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_state(state)
    _audit("grant_created", {"tool": tool, "scope": scope, "note": note})
    return True


def revoke(tool: str, source: str = "") -> bool:
    """Withdraw a standing grant. Revocation is always permitted."""
    state = _load_state()
    if tool in state.get("grants", {}):
        state["grants"].pop(tool, None)
        _save_state(state)
        _audit("grant_revoked", {"tool": tool, "source": source})
        return True
    return False


def list_grants() -> Dict[str, dict]:
    return _load_state().get("grants", {})


def clear_session_grants() -> None:
    """Drop session-scoped grants. Called at session start."""
    state = _load_state()
    state["grants"] = {t: g for t, g in state.get("grants", {}).items()
                       if g.get("scope") == "always"}
    _save_state(state)


def set_custom_level(tool: str, level: str, source: str = "") -> bool:
    """Per-tool override used by the CUSTOM profile. Humans only.

    A custom level can make a tool *stricter*, never looser than RED: RED is
    computed from the action itself and is not overridable here.
    """
    if source != HUMAN or level not in (GREEN, YELLOW, RED):
        return False
    state = _load_state()
    state["custom_levels"][tool] = level
    _save_state(state)
    _audit("custom_level_set", {"tool": tool, "level": level})
    return True


# ── Classification ────────────────────────────────────────────────────────────

def _audit(action: str, params: Any = None, error: str = "") -> None:
    try:
        from pinpoint.security import audit as _audit_mod
        _audit_mod.record(action, params=params, error=error)
    except Exception:
        pass


def _spec_for(tool: str) -> Optional[Any]:
    """Look the tool up in the registry if Stage 3 is installed."""
    try:
        from pinpoint.tools import registry
        return registry.get(tool)
    except Exception:
        return None


def _attributes(tool: str) -> Dict[str, bool]:
    spec = _spec_for(tool)
    if spec is not None:
        return {
            "reversible": bool(getattr(spec, "reversible", True)),
            "external_side_effect": bool(getattr(spec, "external_side_effect", False)),
            "costs_money": bool(getattr(spec, "costs_money", False)),
        }
    reversible, external, money = _FALLBACK_ATTRS.get(tool, (True, False, False))
    return {"reversible": reversible, "external_side_effect": external,
            "costs_money": money}


def _base_level(tool: str) -> str:
    spec = _spec_for(tool)
    if spec is not None and getattr(spec, "level", None):
        return spec.level
    return _FALLBACK_LEVELS.get(tool, YELLOW)


def _param_text(params: Any) -> str:
    if isinstance(params, dict):
        return " ".join(str(v) for v in params.values())
    return str(params or "")


def _targets_protected_path(params: Any) -> Optional[str]:
    if not isinstance(params, dict):
        return None
    for key in ("filename", "path", "file", "target", "destination"):
        raw = params.get(key)
        if not raw:
            continue
        normalized = str(raw).replace("\\", "/")
        for part in PROTECTED_PATH_PARTS:
            if part.replace("\\", "/") in normalized:
                return str(raw)
    return None


def _red_reason(tool: str, params: Any) -> Optional[tuple]:
    """Return ``(reason, rule)`` if this action is hard blocked."""
    # 1. Tampering with PinPoint's own oversight machinery.
    if tool in ("write_file", "write_anywhere", "delete_file", "modify_own_source"):
        hit = _targets_protected_path(params)
        if hit:
            return (f"'{hit}' is part of PinPoint's safety machinery and cannot be "
                    f"modified by the agent", "protected_path")

    text = _param_text(params)

    # 2. Destructive or control-bypassing shell.
    if tool in ("run_shell", "pip_install", "run_python"):
        for pattern, why in _RED_SHELL_PATTERNS:
            if pattern.search(text):
                return (f"blocked: {why}", "red_shell")

    # 3. Stated intent to bypass authentication or oversight, in any tool.
    for pattern, why in _RED_INTENT_PATTERNS:
        if pattern.search(text.lower()):
            return (f"blocked: {why}", "red_intent")

    return None


def check(tool: str, params: Any = None, profile: Optional[str] = None) -> Decision:
    """Classify an action. This is the single authority on what may run."""
    params = params if isinstance(params, dict) else {}
    active = (profile or get_profile()).upper()
    attrs = _attributes(tool)

    red = _red_reason(tool, params)
    if red:
        return Decision(tool=tool, level=RED, allowed=False, requires_approval=False,
                        reason=red[0], rule=red[1], profile=active, attributes=attrs)

    state = _load_state()
    level = state.get("custom_levels", {}).get(tool) or _base_level(tool)

    if level == RED:
        return Decision(tool=tool, level=RED, allowed=False, requires_approval=False,
                        reason=f"'{tool}' is classified RED", rule="tool_level",
                        profile=active, attributes=attrs)

    if level == GREEN:
        return Decision(tool=tool, level=GREEN, allowed=True, requires_approval=False,
                        reason="reversible / read-only action", rule="green",
                        profile=active, attributes=attrs)

    # YELLOW from here. A standing human grant short-circuits the ask.
    if tool in state.get("grants", {}):
        note = state["grants"][tool].get("note", "")
        return Decision(tool=tool, level=YELLOW, allowed=True, requires_approval=False,
                        reason=f"covered by a standing human grant{': ' + note if note else ''}",
                        rule="grant", profile=active, attributes=attrs)

    if active == AUTONOMOUS:
        # Broad authorization still stops at money and the outside world.
        if attrs["costs_money"] or attrs["external_side_effect"]:
            return Decision(tool=tool, level=YELLOW, allowed=False, requires_approval=True,
                            reason="external or billable action — approval required even "
                                   "in AUTONOMOUS",
                            rule="autonomous_external", profile=active, attributes=attrs)
        return Decision(tool=tool, level=YELLOW, allowed=True, requires_approval=False,
                        reason="AUTONOMOUS profile covers local actions", rule="profile",
                        profile=active, attributes=attrs)

    if active == ASSISTED:
        if attrs["reversible"] and not attrs["external_side_effect"] and not attrs["costs_money"]:
            return Decision(tool=tool, level=YELLOW, allowed=True, requires_approval=False,
                            reason="reversible local action under ASSISTED", rule="profile",
                            profile=active, attributes=attrs)
        return Decision(tool=tool, level=YELLOW, allowed=False, requires_approval=True,
                        reason="irreversible or external action needs approval",
                        rule="assisted_risky", profile=active, attributes=attrs)

    # SAFE and CUSTOM default: ask about everything YELLOW.
    return Decision(tool=tool, level=YELLOW, allowed=False, requires_approval=True,
                    reason=f"{active} profile requires approval for '{tool}'",
                    rule="profile", profile=active, attributes=attrs)


def explain(tool: str, params: Any = None) -> str:
    """Human-readable rationale — used in approval prompts and reports."""
    d = check(tool, params)
    verdict = ("BLOCKED" if d.blocked else
               "NEEDS APPROVAL" if d.requires_approval else "ALLOWED")
    return f"{tool}: {d.level} → {verdict} ({d.reason})"
