"""Runtime capability probes.

Before an action runs, the executor asks whether the capability it needs
actually exists here. A missing capability produces an honest UNAVAILABLE
result with a concrete alternative — never a hopeful attempt that gets reported
as success.

A probe returns ``True`` (present), ``False`` (definitely absent), or ``None``
(can't tell). Only a definite ``False`` blocks an action.
"""

import importlib
import os
import platform
import shutil
from typing import Dict, Optional, Tuple

# Suggested fallback when a capability is missing — the "closest possible
# alternative" the agent should offer instead of pretending.
WORKAROUNDS: Dict[str, str] = {
    "pyautogui": "no keyboard/mouse control here — drive the task through shell "
                 "commands or files instead",
    "display": "no graphical display — run the program headless and inspect its "
               "output instead of its window",
    "desktop": "no desktop session — open URLs and files by path rather than "
               "launching applications",
    "screen": "no screen capture available — rely on program output and logs",
    "vision_model": "no vision model reachable — describe state from logs and "
                    "file contents instead",
    "audio_out": "no audio output device — return the text instead of speaking it",
    "sms_provider": "no messaging provider configured — set PINPOINT_SMS_PROVIDER "
                    "and its credentials, or draft the message for the user to send",
    "email_provider": "no email provider configured — set PINPOINT_EMAIL_PROVIDER, "
                      "or draft the email for the user to send",
    "voice_provider": "no telephony provider configured — set "
                      "PINPOINT_VOICE_PROVIDER, or draft what to say on the call",
}


def _module_present(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def _has_display() -> Optional[bool]:
    system = platform.system()
    if system in ("Windows", "Darwin"):
        return True
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return True
    return False


def _provider_configured(kind: str) -> Optional[bool]:
    try:
        from pinpoint.communication import manager
        return manager.provider_available(kind)
    except Exception:
        # Stage 8 not installed yet — fall back to the env-var contract.
        return bool(os.environ.get(f"PINPOINT_{kind.upper()}_PROVIDER"))


def probe(requirement: str) -> Tuple[Optional[bool], str]:
    """Return ``(available, detail)`` for one requirement."""
    if requirement == "pyautogui":
        present = _module_present("pyautogui")
        return present, ("pyautogui available" if present else
                         "pyautogui is not installed")
    if requirement in ("display", "desktop"):
        present = _has_display()
        return present, ("graphical session available" if present else
                         "no graphical display (headless environment)")
    if requirement == "screen":
        present = _module_present("PIL") or _module_present("mss")
        return present, ("screen capture available" if present else
                         "neither Pillow nor mss is installed")
    if requirement == "vision_model":
        return None, "vision model availability unknown until called"
    if requirement == "audio_out":
        present = bool(shutil.which("espeak") or shutil.which("aplay")
                       or shutil.which("afplay") or platform.system() == "Windows"
                       or _module_present("edge_tts"))
        return present, ("audio output available" if present else
                         "no audio playback path found")
    if requirement in ("sms_provider", "email_provider", "voice_provider"):
        kind = requirement.replace("_provider", "")
        present = _provider_configured(kind)
        return present, (f"{kind} provider configured" if present else
                         f"no {kind} provider is configured")
    return None, f"unknown requirement '{requirement}'"


def check_all(requirements) -> Tuple[bool, str]:
    """Check a tool's requirements. Returns ``(ok, explanation)``.

    ``ok`` is False only when something is *definitely* missing.
    """
    for requirement in requirements or []:
        available, detail = probe(requirement)
        if available is False:
            workaround = WORKAROUNDS.get(requirement, "")
            message = detail + (f". {workaround}" if workaround else "")
            return False, message
    return True, "requirements met"


def report() -> Dict[str, dict]:
    """Full capability snapshot — useful in reports and the dashboard."""
    out = {}
    for requirement in ("pyautogui", "display", "screen", "audio_out",
                        "sms_provider", "email_provider", "voice_provider"):
        available, detail = probe(requirement)
        out[requirement] = {"available": available, "detail": detail,
                            "workaround": WORKAROUNDS.get(requirement, "")}
    return out
