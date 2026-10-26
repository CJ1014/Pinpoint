"""Emergency stop — the one control the agent cannot talk its way past.

Engaged state lives in two places at once: an in-process flag (so a running
loop halts immediately) and a file on disk (so it survives a crash or restart,
and so an outside process can trip it). Either one being set means stopped.

Clearing requires ``source="human"``. There is deliberately no agent-facing
tool that clears the stop, and ``clear()`` refuses any other source, so a model
cannot resume itself by claiming authority it doesn't have.
"""

import os
import threading
from datetime import datetime, timezone
from typing import Optional

from pinpoint import jsonstore, paths

_flag = threading.Event()
_lock = threading.Lock()

HUMAN = "human"


class EmergencyStopError(RuntimeError):
    """Raised when an action is attempted while the stop is engaged."""


def _stop_file() -> str:
    return paths.output_dir("EMERGENCY_STOP.json")


def engage(reason: str = "", source: str = "unknown") -> dict:
    """Halt autonomous execution. Anyone may engage — including the agent."""
    record = {
        "engaged": True,
        "reason": reason or "no reason given",
        "source": source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with _lock:
        _flag.set()
        jsonstore.write_json(_stop_file(), record)
    try:
        from pinpoint.security import audit
        audit.record("emergency_stop_engaged", params={"reason": reason, "source": source})
    except Exception:
        pass
    return record


def is_engaged() -> bool:
    """True if execution is halted, in this process or on disk."""
    if _flag.is_set():
        return True
    return bool(jsonstore.read_json(_stop_file(), {}).get("engaged"))


def status() -> dict:
    """Full stop record, or ``{"engaged": False}``."""
    if _flag.is_set():
        disk = jsonstore.read_json(_stop_file(), {})
        return disk if disk.get("engaged") else {"engaged": True, "reason": "in-process"}
    disk = jsonstore.read_json(_stop_file(), {})
    return disk if disk.get("engaged") else {"engaged": False}


def clear(source: str = "", reason: str = "") -> bool:
    """Resume execution. Only a human may clear the stop.

    Returns False for any other source — including the agent asserting that it
    has decided the emergency is over.
    """
    if source != HUMAN:
        try:
            from pinpoint.security import audit
            audit.record(
                "emergency_stop_clear_refused",
                params={"source": source},
                error="only a human may clear the emergency stop",
            )
        except Exception:
            pass
        return False
    with _lock:
        _flag.clear()
        jsonstore.write_json(_stop_file(), {
            "engaged": False,
            "cleared_by": source,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    try:
        from pinpoint.security import audit
        audit.record("emergency_stop_cleared", params={"source": source, "reason": reason})
    except Exception:
        pass
    return True


def check(action: str = "") -> None:
    """Raise :class:`EmergencyStopError` if the stop is engaged."""
    if is_engaged():
        info = status()
        raise EmergencyStopError(
            f"EMERGENCY STOP engaged ({info.get('reason', '')}). "
            f"Refusing to run {action or 'any action'}. A human must clear it."
        )


def reset_for_tests() -> None:
    """Drop in-process state. Test helper only."""
    _flag.clear()
    try:
        os.unlink(_stop_file())
    except OSError:
        pass


def phrase_engages(text: str) -> Optional[str]:
    """Detect an emergency-stop phrase in user input. Returns the matched phrase.

    Deterministic string matching — the halt must not depend on a model
    correctly interpreting the user's panic.
    """
    lowered = (text or "").lower()
    for phrase in ("stop everything", "emergency stop", "halt everything",
                   "abort everything", "shut it down now", "full stop"):
        if phrase in lowered:
            return phrase
    return None
