"""AGI-layer cross-session checkpoint module for PinPoint.

Handles long-horizon project continuity across separate sessions.  Works
alongside pinpoint_checkpoint.py (which handles per-build session step logs).
Checkpoints are stored as JSON files in output/checkpoints/ and carry enough
context to resume a multi-session project with a single get_resume_context()
call at the start of a new session.

No LLM calls are made here — this module is pure file I/O and JSON.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Output directory for cross-session AGI checkpoints.
CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), "output", "checkpoints")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _timestamp_compact() -> str:
    """Return a compact timestamp safe for use in filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _slugify(text: str) -> str:
    """Convert a project name into a safe filename component (max 40 chars)."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:40] if slug else "project"


def _checkpoint_path(checkpoint_id: str) -> str:
    return os.path.join(CHECKPOINTS_DIR, f"{checkpoint_id}.json")


# ── AGICheckpoint class ───────────────────────────────────────────────────────

class AGICheckpoint:
    """A single cross-session AGI project checkpoint.

    Typical lifecycle::

        cp = AGICheckpoint("my_project", session_num=3)
        cp_id = cp.create(
            root_goal="Build a complete web app",
            completed_tasks=["scaffold", "auth"],
            pending_tasks=["payment integration", "testing"],
            reasoning_summary="Auth works; next is Stripe integration.",
            lessons_learned=["Use PKCE not implicit flow"],
            files_created=["output/app.py", "output/auth.py"],
        )
        # In a later session:
        context_str = get_resume_context(cp_id)
    """

    def __init__(self, project_name: str, session_num: int) -> None:
        self.project_name: str = project_name
        self.session_num: int = int(session_num)
        # Fields populated by create(); empty/zero before that.
        self.id: str = ""
        self.timestamp: str = ""
        self.root_goal: str = ""
        self.completed_tasks: list = []
        self.pending_tasks: list = []
        self.reasoning_summary: str = ""
        self.lessons_learned: list = []
        self.files_created: list = []

    # ── create ────────────────────────────────────────────────────────────────

    def create(
        self,
        root_goal: str,
        completed_tasks: list,
        pending_tasks: list,
        reasoning_summary: str,
        lessons_learned: list,
        files_created: list,
    ) -> str:
        """Persist the checkpoint and return the checkpoint ID.

        The checkpoint ID is the filename stem (no .json suffix).  It has the
        form ``{project_slug}_{timestamp}``, e.g.
        ``my_project_20260704_153012``.

        Memory is updated: ``active_checkpoint_id`` is set and the ID is
        appended to ``checkpoint_chain`` (last 20 kept).
        """
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

        self.timestamp = _now_iso()
        self.root_goal = str(root_goal)
        self.completed_tasks = list(completed_tasks or [])
        self.pending_tasks = list(pending_tasks or [])
        self.reasoning_summary = str(reasoning_summary)
        self.lessons_learned = list(lessons_learned or [])
        self.files_created = list(files_created or [])

        slug = _slugify(self.project_name)
        ts = _timestamp_compact()
        self.id = f"{slug}_{ts}"

        data = self.to_dict()

        # Atomic write: write to a temp file then replace the target.
        fd, tmp_path = tempfile.mkstemp(dir=CHECKPOINTS_DIR, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, _checkpoint_path(self.id))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        # Update persistent memory — best-effort, never blocks the save.
        self._update_memory()

        return self.id

    # ── to_dict ───────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return the checkpoint as a plain, JSON-serialisable dict."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "project_name": self.project_name,
            "session_num": self.session_num,
            "root_goal": self.root_goal,
            "completed_tasks": self.completed_tasks,
            "pending_tasks": self.pending_tasks,
            "reasoning_summary": self.reasoning_summary,
            "lessons_learned": self.lessons_learned,
            "files_created": self.files_created,
        }

    # ── format_resume_context ─────────────────────────────────────────────────

    def format_resume_context(self) -> str:
        """Return a human-readable block for injecting into LLM context.

        The returned string summarises what was accomplished, what remains,
        where reasoning left off, and any lessons learned — giving a resuming
        model enough information to continue without restarting.
        """
        lines: list[str] = []
        sep = "=" * 60
        lines.append(sep)
        lines.append("[AGI CHECKPOINT -- RESUMING PROJECT]")
        lines.append(f"Project     : {self.project_name}")
        lines.append(f"Checkpoint  : {self.id}")
        lines.append(f"Saved       : {self.timestamp}")
        lines.append(f"Session num : {self.session_num}")
        lines.append(sep)

        lines.append(f"\nROOT GOAL:\n  {self.root_goal}")

        if self.completed_tasks:
            lines.append("\nCOMPLETED TASKS:")
            for task in self.completed_tasks:
                lines.append(f"  [x] {task}")
        else:
            lines.append("\nCOMPLETED TASKS:\n  (none recorded)")

        if self.pending_tasks:
            lines.append("\nPENDING TASKS (do these next, in order):")
            for task in self.pending_tasks:
                lines.append(f"  [ ] {task}")
        else:
            lines.append("\nPENDING TASKS:\n  (none -- project may be complete)")

        lines.append(f"\nREASONING / WHERE WE LEFT OFF:\n  {self.reasoning_summary}")

        if self.lessons_learned:
            lines.append("\nLESSONS LEARNED SO FAR:")
            for lesson in self.lessons_learned:
                lines.append(f"  * {lesson}")

        if self.files_created:
            lines.append("\nFILES CREATED SO FAR:")
            for filepath in self.files_created:
                lines.append(f"  {filepath}")

        lines.append("")
        lines.append(sep)
        lines.append(
            "Continue from the first pending task above. "
            "Do NOT restart -- pick up where you left off."
        )
        lines.append(sep)

        return "\n".join(lines)

    # ── _update_memory ────────────────────────────────────────────────────────

    def _update_memory(self) -> None:
        """Write active_checkpoint_id and checkpoint_chain into memory.json."""
        try:
            from tools import _load_memory, _save_memory_file  # lazy import
            mem = _load_memory()
            mem["active_checkpoint_id"] = self.id
            chain: list = mem.get("checkpoint_chain", [])
            chain.append(self.id)
            mem["checkpoint_chain"] = chain[-20:]  # keep last 20
            _save_memory_file(mem)
        except Exception as exc:
            logger.warning("AGICheckpoint._update_memory failed: %s", exc)


# ── Module-level functions ────────────────────────────────────────────────────

def create_checkpoint(
    project_name: str,
    session_num: int,
    root_goal: str,
    completed_tasks: list,
    pending_tasks: list,
    reasoning_summary: str,
    lessons_learned: list = None,
    files_created: list = None,
) -> str:
    """Create, persist, and return the ID of a new AGI checkpoint.

    Also updates ``memory["active_checkpoint_id"]`` and appends the new ID to
    ``memory["checkpoint_chain"]``.

    Parameters
    ----------
    project_name:
        Human-readable project name (e.g. ``"My Web App"``).
    session_num:
        Which session this checkpoint belongs to (integer).
    root_goal:
        The top-level goal the project is working toward.
    completed_tasks:
        Tasks that have been finished.
    pending_tasks:
        Tasks still to do, in priority order.
    reasoning_summary:
        A short paragraph describing where reasoning left off and why.
    lessons_learned:
        Optional list of lessons / gotchas discovered during the project.
    files_created:
        Optional list of file paths produced so far.

    Returns
    -------
    str
        The checkpoint ID (filename without ``.json``).
    """
    cp = AGICheckpoint(project_name, session_num)
    return cp.create(
        root_goal=root_goal,
        completed_tasks=completed_tasks,
        pending_tasks=pending_tasks,
        reasoning_summary=reasoning_summary,
        lessons_learned=lessons_learned or [],
        files_created=files_created or [],
    )


def list_checkpoints(project_name: str = None) -> list:
    """Return a list of checkpoint dicts, sorted newest first.

    Parameters
    ----------
    project_name:
        If given, only checkpoints whose ``project_name`` field matches
        (case-insensitive) are returned.

    Returns
    -------
    list[dict]
        Each element is the dict stored in the corresponding ``.json`` file.
    """
    if not os.path.isdir(CHECKPOINTS_DIR):
        return []

    results: list[dict] = []
    for filename in os.listdir(CHECKPOINTS_DIR):
        if not filename.endswith(".json"):
            continue
        checkpoint_id = filename[:-5]  # strip .json
        try:
            cp_dict = load_checkpoint(checkpoint_id)
        except Exception:
            continue
        if project_name is not None:
            stored = cp_dict.get("project_name", "")
            if stored.lower() != project_name.lower():
                continue
        results.append(cp_dict)

    # ISO-8601 timestamps sort lexicographically — newest first.
    results.sort(key=lambda d: d.get("timestamp", ""), reverse=True)
    return results


def load_checkpoint(checkpoint_id: str) -> dict:
    """Load a specific checkpoint from disk and return its dict.

    Raises
    ------
    FileNotFoundError
        If no checkpoint with that ID exists on disk.
    json.JSONDecodeError
        If the checkpoint file is corrupt.
    """
    path = _checkpoint_path(checkpoint_id)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_id!r}  (expected at {path})"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_resume_context(checkpoint_id: str) -> str:
    """Load a checkpoint and return its formatted resume context string.

    This is the main entry-point for injecting a past checkpoint into a new
    session.  Call it at session start, then include the returned string in
    the LLM's system prompt or first user turn.
    """
    cp_dict = load_checkpoint(checkpoint_id)

    # Reconstruct a lightweight AGICheckpoint so we can reuse
    # format_resume_context() without duplicating the formatting logic.
    cp = AGICheckpoint(
        project_name=cp_dict.get("project_name", "unknown"),
        session_num=cp_dict.get("session_num", 0),
    )
    cp.id = cp_dict.get("id", checkpoint_id)
    cp.timestamp = cp_dict.get("timestamp", "")
    cp.root_goal = cp_dict.get("root_goal", "")
    cp.completed_tasks = cp_dict.get("completed_tasks", [])
    cp.pending_tasks = cp_dict.get("pending_tasks", [])
    cp.reasoning_summary = cp_dict.get("reasoning_summary", "")
    cp.lessons_learned = cp_dict.get("lessons_learned", [])
    cp.files_created = cp_dict.get("files_created", [])

    return cp.format_resume_context()


def format_checkpoint_list() -> str:
    """Return a formatted multi-line string listing all checkpoints.

    Suitable for printing to the console or for embedding in an LLM prompt
    as a summary of all saved project checkpoints.
    """
    all_checkpoints = list_checkpoints()
    if not all_checkpoints:
        return "No AGI checkpoints found."

    lines: list[str] = [
        f"AGI Checkpoints ({len(all_checkpoints)} total):",
        "-" * 52,
    ]

    for cp in all_checkpoints:
        cid = cp.get("id", "?")
        project = cp.get("project_name", "?")
        ts = cp.get("timestamp", "?")
        session = cp.get("session_num", "?")
        root_goal = cp.get("root_goal", "")
        pending = cp.get("pending_tasks", [])
        completed = cp.get("completed_tasks", [])

        lines.append(f"  [{cid}]")
        lines.append(f"    Project  : {project}  (session {session})")
        lines.append(f"    Saved    : {ts}")
        lines.append(f"    Goal     : {root_goal[:78]}")
        lines.append(
            f"    Progress : {len(completed)} task(s) done  |  "
            f"{len(pending)} pending"
        )
        if pending:
            lines.append(f"    Next up  : {pending[0][:72]}")
        lines.append("")

    lines.append("-" * 52)
    lines.append("Use get_resume_context('<id>') to load a checkpoint.")
    return "\n".join(lines)
