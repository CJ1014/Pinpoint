"""Stage 4: long-horizon task memory for PinPoint.

Checkpoints an autonomous build session to disk so a long build can survive
context limits, crashes, or restarts. Checkpoints live in output/sessions/ as
browsable JSON files; an unfinished one can be resumed next session.
"""

import os
import json
import time
import tempfile
from datetime import datetime, timezone

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(ROOT_DIR, "output", "sessions")

# Checkpoints older than this are considered stale and won't auto-resume.
STALE_AFTER_SECONDS = 24 * 60 * 60  # 24h


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _path(task_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{task_id}.json")


def new_task_id(session_num: int) -> str:
    return f"build_session_{session_num}_{int(time.time())}"


def save_checkpoint(task_id: str, step: int, goal: str, completed_steps: list,
                    current_context: str = "", status: str = "in_progress") -> None:
    """Atomically write a checkpoint. Best-effort — never raises."""
    try:
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        # Summarize completed steps into compact bullets to save tokens on resume.
        bullets = []
        for s in completed_steps[-40:]:
            if isinstance(s, dict):
                bullets.append({
                    "step": s.get("step"),
                    "action": str(s.get("action", ""))[:40],
                    "result": str(s.get("result", ""))[:120],
                })
        data = {
            "task_id": task_id,
            "step": step,
            "status": status,
            "goal": goal,
            "completed_steps": bullets,
            "current_context": (current_context or "")[:2000],
            "timestamp": _now(),
            "epoch": int(time.time()),
        }
        fd, tmp = tempfile.mkstemp(dir=SESSIONS_DIR, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, _path(task_id))
    except Exception:
        pass


def load_checkpoint(task_id: str):
    try:
        with open(_path(task_id), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def mark_complete(task_id: str, summary: str = "") -> None:
    cp = load_checkpoint(task_id)
    if not cp:
        return
    cp["status"] = "complete"
    cp["summary"] = (summary or "")[:300]
    cp["timestamp"] = _now()
    try:
        with open(_path(task_id), "w", encoding="utf-8") as f:
            json.dump(cp, f, indent=2)
    except Exception:
        pass


def list_checkpoints() -> list:
    """All checkpoints, newest first — for browsing past builds."""
    out = []
    if not os.path.isdir(SESSIONS_DIR):
        return out
    for fn in os.listdir(SESSIONS_DIR):
        if fn.endswith(".json"):
            cp = load_checkpoint(fn[:-5])
            if cp:
                out.append(cp)
    out.sort(key=lambda c: c.get("epoch", 0), reverse=True)
    return out


def latest_incomplete():
    """Newest non-stale in_progress checkpoint, or None."""
    now = int(time.time())
    for cp in list_checkpoints():
        if cp.get("status") == "in_progress" and (now - cp.get("epoch", 0)) < STALE_AFTER_SECONDS:
            return cp
    return None


def resume_prompt(cp: dict) -> str:
    """Build a compact context note so the model can pick up where it left off."""
    steps = cp.get("completed_steps", [])
    bullets = "\n".join(
        f"  - step {s.get('step')}: {s.get('action')} → {s.get('result')}" for s in steps[-12:]
    ) or "  (no recorded steps)"
    ctx = cp.get("current_context", "").strip()
    ctx_block = f"\nWhere you left off:\n{ctx}\n" if ctx else ""
    return (
        f"[RESUMING A PREVIOUS BUILD]\n"
        f"You were working on: {cp.get('goal', 'unknown')}\n"
        f"You had completed {cp.get('step', 0)} steps:\n{bullets}\n{ctx_block}"
        f"Continue from here. Pick up the thread — don't start over. "
        f"If it's already finished, verify it and call done()."
    )
