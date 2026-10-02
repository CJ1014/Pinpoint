"""Independent verification of action effects.

The rule this module exists to enforce: **never infer completion merely from
the absence of an error.** Each verification method goes and checks the world —
the file is stat-ed, the exit code is parsed, the port is connected to, the
provider's confirmation id is looked for. When no independent check is possible
the honest answer is ``None`` (unverified), not ``True``.
"""

import os
import re
import socket
from dataclasses import dataclass
from typing import Any, Dict, Optional

from pinpoint import epistemics
from pinpoint.tools import registry

_EXIT_CODE = re.compile(r"exit code:\s*(-?\d+)", re.IGNORECASE)
_WRITTEN_PATH = re.compile(r"Written\s+\d+\s+chars?\s+to\s+(.+?)\s*$", re.MULTILINE)
_PORT = re.compile(r":(\d{2,5})\b")

# Phrases that mean the tool itself reported a failure. Used only as a
# *negative* signal — their absence never counts as proof of success.
_FAILURE_MARKERS = (
    "traceback (most recent call last)", "permission denied", "no such file",
    "not found", "command not found", "syntax error", "error:", "failed to",
    "unable to", "cannot ", "could not ", "refused", "timed out",
    "blocked by guardrail", "rejected:",
)


@dataclass
class VerificationOutcome:
    """``verified`` is True / False / None — None means genuinely unknown."""
    method: str
    verified: Optional[bool]
    detail: str = ""
    confidence: float = 0.5
    epistemic: str = epistemics.UNKNOWN

    def to_dict(self) -> dict:
        return {"method": self.method, "verified": self.verified,
                "detail": self.detail, "confidence": self.confidence,
                "epistemic": self.epistemic}


def _reports_failure(output: str) -> Optional[str]:
    lowered = (output or "").lower()
    for marker in _FAILURE_MARKERS:
        if marker in lowered:
            return marker
    return None


def _resolve_path(candidate: str) -> str:
    """Resolve a tool-supplied path the way tools.py would."""
    if not candidate:
        return ""
    if os.path.isabs(candidate):
        return candidate
    try:
        import tools
        return tools._safe_path(candidate)
    except Exception:
        return os.path.abspath(candidate)


def _target_path(params: Dict[str, Any], output: str) -> str:
    """Work out which file to actually check, reconciling both signals.

    Taking the tool's word for it is a spoofing hole — a tool reporting
    "Written 5 chars to /etc/hostname" would verify green against a file it
    never touched. Ignoring the tool entirely is wrong too, because a relative
    filename only resolves correctly against the root the tool actually used.

    So: an absolute request is authoritative, and a relative request accepts
    the tool's resolved path only when it names the same file.
    """
    requested = ""
    for key in ("path", "filename", "file", "output_file", "save_path", "target"):
        value = params.get(key)
        if value:
            requested = str(value)
            break

    match = _WRITTEN_PATH.search(output or "")
    reported = match.group(1).strip() if match else ""

    if not requested:
        return reported

    if os.path.isabs(requested):
        return requested

    if reported and os.path.basename(reported) == os.path.basename(requested):
        return reported

    return _resolve_path(requested)


def _verify_file_exists(params, output) -> VerificationOutcome:
    path = _target_path(params, output)
    if not path:
        return VerificationOutcome(registry.V_FILE_EXISTS, None,
                                   "no target path to check", 0.4)
    if os.path.isfile(path):
        size = os.path.getsize(path)
        if size == 0:
            return VerificationOutcome(registry.V_FILE_EXISTS, False,
                                       f"{os.path.basename(path)} exists but is empty",
                                       0.9, epistemics.OBSERVED)
        return VerificationOutcome(registry.V_FILE_EXISTS, True,
                                   f"{os.path.basename(path)} exists ({size} bytes)",
                                   0.95, epistemics.OBSERVED)
    return VerificationOutcome(registry.V_FILE_EXISTS, False,
                               f"{path} does not exist after the write", 0.95,
                               epistemics.OBSERVED)


def _verify_file_absent(params, output) -> VerificationOutcome:
    path = _target_path(params, output)
    if not path:
        return VerificationOutcome(registry.V_FILE_ABSENT, None,
                                   "no target path to check", 0.4)
    if os.path.exists(path):
        return VerificationOutcome(registry.V_FILE_ABSENT, False,
                                   f"{path} still exists after the delete", 0.95,
                                   epistemics.OBSERVED)
    return VerificationOutcome(registry.V_FILE_ABSENT, True,
                               f"{os.path.basename(path)} is gone", 0.95,
                               epistemics.OBSERVED)


def _verify_exit_code(params, output) -> VerificationOutcome:
    match = _EXIT_CODE.search(output or "")
    if match:
        code = int(match.group(1))
        if code == 0:
            return VerificationOutcome(registry.V_EXIT_CODE, True, "exit code 0", 0.95,
                                       epistemics.OBSERVED)
        return VerificationOutcome(registry.V_EXIT_CODE, False, f"exit code {code}", 0.95,
                                   epistemics.OBSERVED)
    marker = _reports_failure(output)
    if marker:
        return VerificationOutcome(registry.V_EXIT_CODE, False,
                                   f"output reports failure ({marker.strip()})", 0.75,
                                   epistemics.INFERRED)
    return VerificationOutcome(registry.V_EXIT_CODE, None,
                               "no exit code in the output — cannot confirm it ran cleanly",
                               0.45, epistemics.UNKNOWN)


def _verify_process_running(params, output) -> VerificationOutcome:
    port = params.get("port")
    if not port:
        match = _PORT.search(output or "")
        port = match.group(1) if match else None
    if port:
        try:
            with socket.create_connection(("127.0.0.1", int(port)), timeout=1.5):
                return VerificationOutcome(registry.V_PROCESS_RUNNING, True,
                                           f"port {port} is accepting connections", 0.95,
                                           epistemics.OBSERVED)
        except OSError:
            return VerificationOutcome(registry.V_PROCESS_RUNNING, False,
                                       f"nothing is listening on port {port}", 0.9,
                                       epistemics.OBSERVED)
    if _reports_failure(output):
        return VerificationOutcome(registry.V_PROCESS_RUNNING, False,
                                   "the launch reported an error", 0.7, epistemics.INFERRED)
    return VerificationOutcome(registry.V_PROCESS_RUNNING, None,
                               "cannot confirm the process is running", 0.4,
                               epistemics.UNKNOWN)


def _verify_provider_confirmation(params, output) -> VerificationOutcome:
    """External sends count only when the provider hands back an identifier."""
    text = output or ""
    match = re.search(r"(confirmation|message[_ ]?sid|call[_ ]?sid|message[_ ]?id|id)"
                      r"\s*[:=]\s*([A-Za-z0-9_\-]{6,})", text, re.IGNORECASE)
    if match:
        return VerificationOutcome(registry.V_PROVIDER_CONFIRMATION, True,
                                   f"provider confirmed ({match.group(2)[:24]})", 0.95,
                                   epistemics.OBSERVED)
    if _reports_failure(text):
        return VerificationOutcome(registry.V_PROVIDER_CONFIRMATION, False,
                                   "the provider reported an error", 0.9,
                                   epistemics.OBSERVED)
    return VerificationOutcome(
        registry.V_PROVIDER_CONFIRMATION, None,
        "no provider confirmation id — do not claim this was delivered", 0.2,
        epistemics.UNKNOWN)


def _verify_state_change(params, output, before=None, after=None) -> VerificationOutcome:
    if before is not None or after is not None:
        changed = before != after
        return VerificationOutcome(registry.V_STATE_CHANGE, changed,
                                   "state changed" if changed else "state is unchanged",
                                   0.9 if changed else 0.85, epistemics.OBSERVED)
    if _reports_failure(output):
        return VerificationOutcome(registry.V_STATE_CHANGE, False,
                                   "the tool reported an error", 0.75, epistemics.INFERRED)
    return VerificationOutcome(registry.V_STATE_CHANGE, None,
                               "no before/after snapshot to compare", 0.5,
                               epistemics.UNKNOWN)


def _verify_content(params, output, expected: str = "") -> VerificationOutcome:
    if expected:
        if expected in (output or ""):
            return VerificationOutcome(registry.V_CONTENT, True,
                                       f"output contains {expected[:40]!r}", 0.9,
                                       epistemics.OBSERVED)
        return VerificationOutcome(registry.V_CONTENT, False,
                                   f"output does not contain {expected[:40]!r}", 0.9,
                                   epistemics.OBSERVED)
    if _reports_failure(output):
        return VerificationOutcome(registry.V_CONTENT, False, "the tool reported an error",
                                   0.8, epistemics.INFERRED)
    if (output or "").strip():
        return VerificationOutcome(registry.V_CONTENT, True, "returned content", 0.7,
                                   epistemics.INFERRED)
    return VerificationOutcome(registry.V_CONTENT, None, "empty output", 0.4,
                               epistemics.UNKNOWN)


def _verify_none(params, output) -> VerificationOutcome:
    """Pure-reasoning tools have no external effect to check."""
    if _reports_failure(output):
        return VerificationOutcome(registry.V_NONE, False, "the tool reported an error",
                                   0.8, epistemics.INFERRED)
    return VerificationOutcome(registry.V_NONE, True, "no external effect to verify",
                               0.85, epistemics.OBSERVED)


def _verify_unverifiable(params, output) -> VerificationOutcome:
    """An unregistered tool. Its effect is unknown, and unknown is not success."""
    if _reports_failure(output):
        return VerificationOutcome(registry.V_UNVERIFIABLE, False,
                                   "the tool reported an error", 0.8,
                                   epistemics.INFERRED)
    return VerificationOutcome(
        registry.V_UNVERIFIABLE, None,
        "this tool is not in the registry, so there is no way to check what it "
        "actually did — do not treat it as done", 0.2, epistemics.UNKNOWN)


def verify(tool: str, params: Dict[str, Any], output: str, *,
           method: str = "", before: Any = None, after: Any = None,
           expected: str = "") -> VerificationOutcome:
    """Independently check whether ``tool`` actually did what it claims."""
    params = params or {}
    method = method or registry.require(tool).verification

    if method == registry.V_FILE_EXISTS:
        return _verify_file_exists(params, output)
    if method == registry.V_FILE_ABSENT:
        return _verify_file_absent(params, output)
    if method == registry.V_EXIT_CODE:
        return _verify_exit_code(params, output)
    if method == registry.V_PROCESS_RUNNING:
        return _verify_process_running(params, output)
    if method == registry.V_PROVIDER_CONFIRMATION:
        return _verify_provider_confirmation(params, output)
    if method == registry.V_STATE_CHANGE:
        return _verify_state_change(params, output, before, after)
    if method == registry.V_CONTENT:
        return _verify_content(params, output, expected)
    if method == registry.V_UNVERIFIABLE:
        return _verify_unverifiable(params, output)
    return _verify_none(params, output)
