"""Long-horizon checkpoints, verified against reality on resume.

A checkpoint records the objective, the plan graph, what was observed, what was
learned, and what to do next. Resuming does *not* trust it: every effect the
checkpoint claims (a file written, a port serving) is re-checked against the
world as it is now, and anything that no longer holds reopens the task that
claimed it.

Three checkpoint formats existed in v2 (checkpoint.py, agi_checkpoint.py,
pinpoint_checkpoint.py). This is the one the v3 loop uses; the older modules
keep working untouched for the code that still calls them.
"""

import os
import socket
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pinpoint import jsonstore, paths

# Effect kinds a checkpoint can re-verify.
FILE_EFFECT = "file"
PORT_EFFECT = "port"


@dataclass
class Discrepancy:
    """Something the checkpoint believed that is no longer true."""
    kind: str
    detail: str
    task_id: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Checkpoint:
    """A resumable snapshot of a long-running objective."""
    goal: str = ""
    objective: Dict[str, Any] = field(default_factory=dict)
    plan: Dict[str, Any] = field(default_factory=dict)
    completed: List[str] = field(default_factory=list)
    active: List[str] = field(default_factory=list)
    pending: List[str] = field(default_factory=list)
    blocked: List[Dict[str, Any]] = field(default_factory=list)
    interrupted: List[Dict[str, Any]] = field(default_factory=list)
    run_status: str = "IN_PROGRESS"    # IN_PROGRESS | INTERRUPTED | BLOCKED
    observations: List[str] = field(default_factory=list)
    lessons: List[str] = field(default_factory=list)
    effects: List[Dict[str, Any]] = field(default_factory=list)
    permissions: Dict[str, Any] = field(default_factory=dict)
    next_action: str = ""
    session: Any = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict) -> "Checkpoint":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def resume_text(self) -> str:
        """What to tell the agent when picking this back up."""
        lines = [f"Resuming: {self.goal}",
                 f"Checkpointed {self.timestamp[:19]} (session {self.session})"]
        if self.completed:
            lines.append("Already done:")
            lines += [f"  ✓ {item}" for item in self.completed[:6]]
        if self.blocked:
            lines.append("Blocked:")
            lines += [f"  ⊘ {b.get('description', '')} — {b.get('reason', '')[:80]}"
                      for b in self.blocked[:4]]
        if self.lessons:
            lines.append("Learned so far:")
            lines += [f"  - {lesson}" for lesson in self.lessons[:4]]
        if self.next_action:
            lines.append(f"Next: {self.next_action}")
        return "\n".join(lines)


def _directory() -> str:
    path = paths.output_dir("v3_checkpoints", "_")
    return os.path.dirname(path)


def _path(checkpoint_id: str) -> str:
    return os.path.join(_directory(), f"{checkpoint_id}.json")


# ── Saving ────────────────────────────────────────────────────────────────────

def effects_from_results(results) -> List[Dict[str, Any]]:
    """Extract re-checkable effects from a run's verified action results."""
    effects: List[Dict[str, Any]] = []
    for result in results or []:
        if getattr(result, "status", "") != "SUCCESS":
            continue
        verification = getattr(result, "verification", {}) or {}
        method = verification.get("method", "")
        params = getattr(result, "params", {}) or {}
        if method == "file_exists":
            target = params.get("path") or params.get("filename")
            if target:
                effects.append({"kind": FILE_EFFECT, "target": str(target),
                                "tool": result.tool})
        elif method == "process_running" and params.get("port"):
            effects.append({"kind": PORT_EFFECT, "target": str(params["port"]),
                            "tool": result.tool})
    return effects


def save(plan, *, session: Any = "", observations: Optional[List[str]] = None,
         lessons: Optional[List[str]] = None, results=None,
         next_action: str = "", checkpoint_id: str = "",
         run_status: str = "") -> Checkpoint:
    """Snapshot a plan mid-flight.

    Passing ``checkpoint_id`` overwrites that checkpoint instead of creating a
    new one, so periodic saves during a long run leave one current snapshot
    rather than a pile of stale ones.
    """
    from pinpoint.agent import planner as P
    from pinpoint.security import permissions

    tasks = plan.tasks
    checkpoint = Checkpoint(
        goal=plan.objective.goal if plan.objective else "",
        objective=plan.objective.to_dict() if plan.objective else {},
        plan=plan.to_dict(),
        completed=[t.description for t in tasks.values() if t.status == P.DONE],
        active=[t.description for t in tasks.values() if t.status == P.ACTIVE],
        pending=[t.description for t in tasks.values() if t.status == P.PENDING],
        blocked=[{"id": t.id, "description": t.description,
                  "reason": t.blocked_reason or "failed"}
                 for t in tasks.values() if t.status in (P.BLOCKED, P.FAILED)],
        observations=list(observations or [])[-30:],
        lessons=list(lessons or [])[-20:],
        effects=effects_from_results(results),
        permissions={"profile": permissions.get_profile(),
                     "grants": list(permissions.list_grants())},
        next_action=next_action or (plan.next_task().description
                                    if plan.next_task() else ""),
        session=session,
    )
    checkpoint.interrupted = [
        {"id": t.id, "description": t.description}
        for t in tasks.values() if t.status == P.INTERRUPTED]
    checkpoint.run_status = run_status or ("INTERRUPTED" if checkpoint.interrupted
                                           else "IN_PROGRESS")
    if checkpoint_id:
        checkpoint.id = checkpoint_id
    jsonstore.write_json(_path(checkpoint.id), checkpoint.to_dict())
    return checkpoint


# ── Loading ───────────────────────────────────────────────────────────────────

def load(checkpoint_id: str) -> Optional[Checkpoint]:
    data = jsonstore.read_json(_path(checkpoint_id), {})
    return Checkpoint.from_dict(data) if data else None


def list_all() -> List[Checkpoint]:
    directory = _directory()
    if not os.path.isdir(directory):
        return []
    out = []
    for name in os.listdir(directory):
        if name.endswith(".json"):
            data = jsonstore.read_json(os.path.join(directory, name), {})
            if data:
                out.append(Checkpoint.from_dict(data))
    return sorted(out, key=lambda c: c.timestamp, reverse=True)


def latest(goal: str = "") -> Optional[Checkpoint]:
    for checkpoint in list_all():
        if not goal or checkpoint.goal == goal:
            return checkpoint
    return None


# ── Reality verification ──────────────────────────────────────────────────────

def _effect_holds(effect: Dict[str, Any]) -> bool:
    kind = effect.get("kind")
    target = effect.get("target", "")
    if kind == FILE_EFFECT:
        return bool(target) and os.path.isfile(target) and os.path.getsize(target) > 0
    if kind == PORT_EFFECT:
        try:
            with socket.create_connection(("127.0.0.1", int(target)), timeout=1.5):
                return True
        except (OSError, ValueError):
            return False
    return True


def verify_against_reality(checkpoint: Checkpoint) -> List[Discrepancy]:
    """Re-check what the checkpoint claims. The world moves while we're away."""
    discrepancies: List[Discrepancy] = []
    for effect in checkpoint.effects:
        if not _effect_holds(effect):
            kind = effect.get("kind")
            target = effect.get("target", "")
            detail = (f"{target} no longer exists" if kind == FILE_EFFECT
                      else f"nothing is listening on port {target}")
            discrepancies.append(Discrepancy(kind=kind, detail=detail))
    return discrepancies


def resume(checkpoint: Checkpoint):
    """Rebuild the plan, then reopen anything reality no longer supports.

    Returns ``(plan, discrepancies)``.
    """
    from pinpoint.agent import planner as P

    plan = P.Plan.from_dict(checkpoint.plan)
    # Work that was halted mid-flight goes back into the ready pool. It was
    # never finished, so resuming has to actually redo it.
    plan.resume_interrupted()
    discrepancies = verify_against_reality(checkpoint)

    if discrepancies:
        # A vanished effect invalidates the task that produced it. Rather than
        # guess which one, reopen the completed tasks whose tool touched it.
        stale_targets = {d.detail.split()[0] for d in discrepancies}
        for task in plan.tasks.values():
            if task.status != P.DONE:
                continue
            if any(str(target) in (task.result_summary or "") or
                   str(target) in task.description for target in stale_targets):
                task.status = P.PENDING
                task.finished_at = ""
                task.notes.append("reopened on resume: its effect no longer holds")

        # Nothing matched by description — reopen the last completed task, which
        # is the most likely producer, and say so rather than assuming it stands.
        if all(t.status != P.PENDING or not t.notes or
               "reopened on resume" not in (t.notes[-1] if t.notes else "")
               for t in plan.tasks.values()):
            completed = [plan.tasks[i] for i in plan.order
                         if plan.tasks[i].status == P.DONE]
            if completed:
                last = completed[-1]
                last.status = P.PENDING
                last.finished_at = ""
                last.notes.append("reopened on resume: a recorded effect is gone")

    return plan, discrepancies


def delete(checkpoint_id: str) -> bool:
    try:
        os.unlink(_path(checkpoint_id))
        return True
    except OSError:
        return False
