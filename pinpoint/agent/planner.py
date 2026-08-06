"""Dynamic execution planner.

This is the v2 goal tree grown up. A plan is a dependency graph of tasks, each
with its own attempt budget and a list of alternative strategies. When a task
fails the plan does not stop — it retries, then switches strategy, and only
when both are exhausted does it mark the task failed and propagate a *blocked*
state to whatever depended on it.

New work discovered mid-flight (a missing dependency, an unexpected second
bug) gets inserted into the graph rather than appended to a static list.
"""

import itertools
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from pinpoint.agent import intent as I

PENDING = "pending"
ACTIVE = "active"
DONE = "done"
FAILED = "failed"
BLOCKED = "blocked"
SKIPPED = "skipped"

OPEN_STATUSES = (PENDING, ACTIVE)
_PRIORITY_VALUE = {I.PRIORITY_LOW: 0, I.PRIORITY_NORMAL: 1,
                   I.PRIORITY_HIGH: 2, I.PRIORITY_URGENT: 3}

_ids = itertools.count(1)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Task:
    """One unit of work in the plan."""
    id: str
    description: str
    kind: str = ""
    status: str = PENDING
    depends_on: List[str] = field(default_factory=list)
    priority: int = 1
    attempts: int = 0
    max_attempts: int = 2
    strategies: List[str] = field(default_factory=list)
    strategy_index: int = 0
    success_criteria: List[str] = field(default_factory=list)
    tool_hint: str = ""
    notes: List[str] = field(default_factory=list)
    blocked_reason: str = ""
    result_summary: str = ""
    deadline: str = ""
    parent: str = ""
    created_at: str = field(default_factory=_now)
    finished_at: str = ""

    @property
    def strategy(self) -> str:
        if self.strategies and self.strategy_index < len(self.strategies):
            return self.strategies[self.strategy_index]
        return ""

    @property
    def has_alternative(self) -> bool:
        return self.strategy_index + 1 < len(self.strategies)

    @property
    def open(self) -> bool:
        return self.status in OPEN_STATUSES

    def instruction(self) -> str:
        """What to actually do right now, including the current strategy."""
        if self.strategy:
            return f"{self.description} — approach: {self.strategy}"
        return self.description

    def to_dict(self) -> dict:
        return {
            "id": self.id, "description": self.description, "kind": self.kind,
            "status": self.status, "depends_on": self.depends_on,
            "priority": self.priority, "attempts": self.attempts,
            "max_attempts": self.max_attempts, "strategies": self.strategies,
            "strategy_index": self.strategy_index,
            "success_criteria": self.success_criteria, "tool_hint": self.tool_hint,
            "notes": self.notes, "blocked_reason": self.blocked_reason,
            "result_summary": self.result_summary, "deadline": self.deadline,
            "parent": self.parent, "created_at": self.created_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        return cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})


@dataclass
class ReplanOutcome:
    """What the planner decided to do about a failure."""
    action: str          # "retry" | "new_strategy" | "blocked" | "escalate"
    task_id: str = ""
    detail: str = ""
    blocked_tasks: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"action": self.action, "task_id": self.task_id,
                "detail": self.detail, "blocked_tasks": self.blocked_tasks}


class Plan:
    """A dependency graph of tasks serving one objective."""

    def __init__(self, objective: Optional[I.Objective] = None,
                 max_actions: int = 200):
        self.objective = objective
        self.tasks: Dict[str, Task] = {}
        self.order: List[str] = []
        self.max_actions = max_actions
        self.actions_used = 0
        self.created_at = _now()
        self.replans: List[dict] = []

    # ── construction ─────────────────────────────────────────────────────────

    def add(self, description: str, *, depends_on: Optional[List[str]] = None,
            task_id: str = "", priority: Optional[int] = None,
            max_attempts: int = 2, strategies: Optional[List[str]] = None,
            success_criteria: Optional[List[str]] = None, tool_hint: str = "",
            kind: str = "", parent: str = "") -> Task:
        task_id = task_id or f"t{next(_ids)}"
        if priority is None:
            priority = _PRIORITY_VALUE.get(
                self.objective.priority if self.objective else I.PRIORITY_NORMAL, 1)
        task = Task(
            id=task_id, description=description, kind=kind,
            depends_on=list(depends_on or []), priority=priority,
            max_attempts=max_attempts, strategies=list(strategies or []),
            success_criteria=list(success_criteria or []), tool_hint=tool_hint,
            parent=parent,
        )
        self.tasks[task_id] = task
        self.order.append(task_id)
        return task

    def insert_after(self, task_id: str, descriptions: List[str],
                     **kwargs) -> List[Task]:
        """Add newly discovered work that must happen after ``task_id``.

        Anything that already depended on ``task_id`` is rewired to depend on
        the last inserted task, so discovered prerequisites really do gate the
        work that needed them.
        """
        anchor = self.tasks.get(task_id)
        if anchor is None:
            return []
        dependents = [t for t in self.tasks.values() if task_id in t.depends_on]

        created: List[Task] = []
        previous = task_id
        for description in descriptions:
            task = self.add(description, depends_on=[previous], parent=task_id,
                            **kwargs)
            created.append(task)
            previous = task.id

        if created:
            # Place the new tasks immediately after the anchor in display order.
            for task in created:
                self.order.remove(task.id)
            anchor_position = self.order.index(task_id) + 1
            self.order[anchor_position:anchor_position] = [t.id for t in created]

            for dependent in dependents:
                dependent.depends_on = [
                    created[-1].id if d == task_id else d
                    for d in dependent.depends_on
                ]
        return created

    # ── scheduling ───────────────────────────────────────────────────────────

    def _deps_satisfied(self, task: Task) -> bool:
        return all(self.tasks.get(dep, Task(id=dep, description="")).status
                   in (DONE, SKIPPED) for dep in task.depends_on)

    def _dep_problem(self, task: Task) -> str:
        for dep in task.depends_on:
            other = self.tasks.get(dep)
            if other is None:
                return f"depends on unknown task {dep}"
            if other.status in (FAILED, BLOCKED):
                return f"depends on {dep} which is {other.status}"
        return ""

    def ready(self) -> List[Task]:
        """Tasks that can start right now, highest priority first."""
        candidates = [self.tasks[i] for i in self.order
                      if self.tasks[i].status == PENDING
                      and self._deps_satisfied(self.tasks[i])]
        return sorted(candidates,
                      key=lambda t: (-t.priority, self.order.index(t.id)))

    def next_task(self) -> Optional[Task]:
        active = [self.tasks[i] for i in self.order if self.tasks[i].status == ACTIVE]
        if active:
            return active[0]
        ready = self.ready()
        return ready[0] if ready else None

    def start(self, task_id: str) -> Optional[Task]:
        task = self.tasks.get(task_id)
        if task is None:
            return None
        task.status = ACTIVE
        task.attempts += 1
        self.actions_used += 1
        return task

    # ── state transitions ────────────────────────────────────────────────────

    def complete(self, task_id: str, summary: str = "") -> Optional[Task]:
        task = self.tasks.get(task_id)
        if task is None:
            return None
        task.status = DONE
        task.result_summary = summary
        task.finished_at = _now()
        self._unblock_dependents(task_id)
        return task

    def skip(self, task_id: str, reason: str = "") -> Optional[Task]:
        task = self.tasks.get(task_id)
        if task is None:
            return None
        task.status = SKIPPED
        task.notes.append(f"skipped: {reason}")
        task.finished_at = _now()
        self._unblock_dependents(task_id)
        return task

    def block(self, task_id: str, reason: str) -> List[str]:
        """Mark a task blocked and propagate to everything downstream."""
        task = self.tasks.get(task_id)
        if task is None:
            return []
        task.status = BLOCKED
        task.blocked_reason = reason
        return self._propagate_block(task_id, f"depends on blocked task {task_id}")

    def _propagate_block(self, task_id: str, reason: str) -> List[str]:
        blocked: List[str] = []
        frontier = [task_id]
        while frontier:
            current = frontier.pop()
            for task in self.tasks.values():
                if current in task.depends_on and task.status in OPEN_STATUSES:
                    task.status = BLOCKED
                    task.blocked_reason = reason
                    blocked.append(task.id)
                    frontier.append(task.id)
        return blocked

    def _unblock_dependents(self, task_id: str) -> None:
        """A task finishing may clear a block it was causing."""
        for task in self.tasks.values():
            if task.status == BLOCKED and task_id in task.depends_on:
                if not self._dep_problem(task):
                    task.status = PENDING
                    task.blocked_reason = ""

    # ── failure handling and replanning ──────────────────────────────────────

    def fail(self, task_id: str, reason: str) -> ReplanOutcome:
        """Record a failed attempt and decide what happens next.

        Retry within budget, then switch to the next strategy, then block —
        and blocking a task blocks whatever was waiting on it.
        """
        task = self.tasks.get(task_id)
        if task is None:
            return ReplanOutcome("escalate", task_id, "unknown task")

        task.notes.append(f"attempt {task.attempts} failed: {reason[:200]}")

        if task.attempts < task.max_attempts:
            task.status = PENDING
            outcome = ReplanOutcome(
                "retry", task_id,
                f"attempt {task.attempts}/{task.max_attempts} failed — retrying")
        elif task.has_alternative:
            task.strategy_index += 1
            task.attempts = 0
            task.status = PENDING
            outcome = ReplanOutcome(
                "new_strategy", task_id,
                f"switching to: {task.strategy}")
        else:
            task.status = FAILED
            task.finished_at = _now()
            blocked = self._propagate_block(
                task_id, f"depends on failed task {task_id}: {reason[:120]}")
            outcome = ReplanOutcome("blocked", task_id,
                                    f"exhausted every approach: {reason[:200]}",
                                    blocked)

        self.replans.append({"timestamp": _now(), **outcome.to_dict()})
        return outcome

    def revive(self, task_id: str, strategy_index: Optional[int] = None) -> Optional[Task]:
        """Reopen a task for another attempt, optionally on a different strategy.

        Used by the recovery engine, which owns the decision about *whether* to
        try again; the plan only owns what that does to the graph.
        """
        task = self.tasks.get(task_id)
        if task is None:
            return None
        if strategy_index is not None and 0 <= strategy_index < len(task.strategies):
            task.strategy_index = strategy_index
        task.attempts = 0
        task.status = PENDING
        task.finished_at = ""
        task.blocked_reason = ""
        self._unblock_after_revival(task_id)
        return task

    def add_strategy(self, task_id: str, strategy: str) -> bool:
        """Give a task another way to succeed — used by the recovery engine."""
        task = self.tasks.get(task_id)
        if task is None:
            return False
        task.strategies.append(strategy)
        if task.status == FAILED:
            task.status = PENDING
            task.strategy_index = len(task.strategies) - 1
            task.attempts = 0
            task.finished_at = ""
            self._unblock_after_revival(task_id)
        return True

    def _unblock_after_revival(self, task_id: str) -> None:
        for task in self.tasks.values():
            if task.status == BLOCKED and task_id in task.depends_on:
                task.status = PENDING
                task.blocked_reason = ""

    # ── status ───────────────────────────────────────────────────────────────

    def progress(self) -> dict:
        counts = {status: 0 for status in
                  (PENDING, ACTIVE, DONE, FAILED, BLOCKED, SKIPPED)}
        for task in self.tasks.values():
            counts[task.status] = counts.get(task.status, 0) + 1
        total = len(self.tasks)
        finished = counts[DONE] + counts[SKIPPED]
        return {
            "total": total,
            "done": counts[DONE],
            "skipped": counts[SKIPPED],
            "failed": counts[FAILED],
            "blocked": counts[BLOCKED],
            "active": counts[ACTIVE],
            "pending": counts[PENDING],
            "percent": round(100.0 * finished / total, 1) if total else 0.0,
            "actions_used": self.actions_used,
            "actions_remaining": max(0, self.max_actions - self.actions_used),
        }

    def is_complete(self) -> bool:
        return bool(self.tasks) and not any(t.open for t in self.tasks.values())

    def succeeded(self) -> bool:
        """Complete with nothing failed or blocked."""
        return self.is_complete() and not any(
            t.status in (FAILED, BLOCKED) for t in self.tasks.values())

    def is_stuck(self) -> bool:
        """Nothing can proceed, but work remains."""
        if self.is_complete():
            return False
        return not self.ready() and not any(
            t.status == ACTIVE for t in self.tasks.values())

    def budget_exhausted(self) -> bool:
        return self.actions_used >= self.max_actions

    def blockers(self) -> List[dict]:
        return [{"task": t.id, "description": t.description,
                 "status": t.status,
                 "reason": t.blocked_reason or (t.notes[-1] if t.notes else "")}
                for t in self.tasks.values() if t.status in (FAILED, BLOCKED)]

    def render(self) -> str:
        glyphs = {DONE: "✓", FAILED: "✗", BLOCKED: "⊘", ACTIVE: "▶",
                  SKIPPED: "–", PENDING: "·"}
        lines = []
        if self.objective:
            lines.append(f"GOAL: {self.objective.goal}")
        progress = self.progress()
        lines.append(f"Progress: {progress['done']}/{progress['total']} "
                     f"({progress['percent']}%)")
        for task_id in self.order:
            task = self.tasks[task_id]
            marker = glyphs.get(task.status, "?")
            suffix = ""
            if task.status == BLOCKED and task.blocked_reason:
                suffix = f"  ← {task.blocked_reason[:70]}"
            elif task.strategy_index and task.strategy:
                suffix = f"  (strategy {task.strategy_index + 1}: {task.strategy[:50]})"
            deps = f" [after {', '.join(task.depends_on)}]" if task.depends_on else ""
            lines.append(f"  {marker} {task.id}: {task.description}{deps}{suffix}")
        return "\n".join(lines)

    # ── persistence ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "objective": self.objective.to_dict() if self.objective else None,
            "tasks": [self.tasks[i].to_dict() for i in self.order],
            "max_actions": self.max_actions,
            "actions_used": self.actions_used,
            "created_at": self.created_at,
            "replans": self.replans[-40:],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Plan":
        objective_data = data.get("objective")
        objective = None
        if objective_data:
            objective = I.Objective(**{
                k: v for k, v in objective_data.items()
                if k in I.Objective.__dataclass_fields__})
        plan = cls(objective, max_actions=data.get("max_actions", 200))
        plan.actions_used = data.get("actions_used", 0)
        plan.created_at = data.get("created_at", _now())
        plan.replans = data.get("replans", [])
        for task_data in data.get("tasks", []):
            task = Task.from_dict(task_data)
            plan.tasks[task.id] = task
            plan.order.append(task.id)
        return plan


# ── Plan templates ────────────────────────────────────────────────────────────

_TEMPLATES: Dict[str, List[dict]] = {
    I.FIX: [
        {"d": "Inspect the project and locate the failing component",
         "hint": "read_file/list_files/run_shell"},
        {"d": "Reproduce the failure and capture the actual error output",
         "hint": "run_shell/run_python", "deps": [0]},
        {"d": "Identify the root cause from that output, not from a guess",
         "deps": [1]},
        {"d": "Develop and apply a fix", "deps": [2], "attempts": 2,
         "strategies": ["fix the most likely root cause directly",
                        "try the next most likely cause",
                        "work around the failure and document the limitation"]},
        {"d": "Run the tests", "hint": "run_tests", "deps": [3]},
        {"d": "Verify the original failure no longer occurs", "deps": [4]},
        {"d": "Report what changed and why", "deps": [5]},
    ],
    I.BUILD: [
        {"d": "Pin down what the finished thing has to do"},
        {"d": "Choose an approach and sketch the design", "deps": [0]},
        {"d": "Implement it", "deps": [1], "attempts": 2,
         "strategies": ["implement the straightforward version",
                        "simplify the approach and implement that"]},
        {"d": "Test it", "hint": "run_tests/run_python", "deps": [2]},
        {"d": "Verify it does what was asked", "deps": [3]},
        {"d": "Report what was built", "deps": [4]},
    ],
    I.INVESTIGATE: [
        {"d": "Gather evidence from the actual system",
         "hint": "read_file/run_shell"},
        {"d": "Form hypotheses that fit that evidence", "deps": [0]},
        {"d": "Test each hypothesis against reality", "deps": [1], "attempts": 2,
         "strategies": ["test the leading hypothesis",
                        "test the remaining hypotheses"]},
        {"d": "State the conclusion and how confident it is", "deps": [2]},
        {"d": "Report the findings", "deps": [3]},
    ],
    I.RESEARCH: [
        {"d": "Search for relevant sources", "hint": "search_web"},
        {"d": "Read the most promising sources", "hint": "fetch_url/deep_research",
         "deps": [0]},
        {"d": "Synthesise the findings, keeping sources attached", "deps": [1]},
        {"d": "Report the answer with its sources", "deps": [2]},
    ],
    I.MONITOR: [
        {"d": "Identify exactly what to watch and what 'healthy' means"},
        {"d": "Record the current baseline state", "deps": [0]},
        {"d": "Start the watcher", "deps": [1]},
        {"d": "Decide which responses are safe to take automatically", "deps": [2]},
        {"d": "Confirm the monitor is active and report", "deps": [3]},
    ],
    I.COMMUNICATE: [
        {"d": "Resolve the recipient unambiguously — ask if more than one matches",
         "hint": "resolve_contact"},
        {"d": "Confirm exactly what the message should say", "deps": [0]},
        {"d": "Check an authorised provider is configured", "deps": [1]},
        {"d": "Send it", "hint": "send_message", "deps": [2]},
        {"d": "Confirm the provider accepted it before claiming it was sent",
         "deps": [3]},
        {"d": "Report the outcome", "deps": [4]},
    ],
    I.SCHEDULE: [
        {"d": "Work out the exact time being asked for"},
        {"d": "Create the scheduled task", "deps": [0]},
        {"d": "Confirm it back to the user", "deps": [1]},
    ],
    I.MAINTAIN: [
        {"d": "Inspect the current state"},
        {"d": "Apply the change", "deps": [0], "attempts": 2,
         "strategies": ["apply the change directly",
                        "apply it incrementally, verifying each step"]},
        {"d": "Run the tests", "hint": "run_tests", "deps": [1]},
        {"d": "Verify nothing else broke", "deps": [2]},
        {"d": "Report what changed", "deps": [3]},
    ],
    I.UNKNOWN: [
        {"d": "Understand what is actually being asked"},
        {"d": "Work out how to do it", "deps": [0]},
        {"d": "Do it", "deps": [1], "attempts": 2,
         "strategies": ["the direct approach", "an alternative approach"]},
        {"d": "Verify the result", "deps": [2]},
        {"d": "Report what happened", "deps": [3]},
    ],
}


def plan_for(objective: I.Objective, max_actions: int = 200) -> Plan:
    """Build the initial plan for an objective."""
    plan = Plan(objective, max_actions=max_actions)
    template = _TEMPLATES.get(objective.kind, _TEMPLATES[I.UNKNOWN])

    created: List[Task] = []
    for step in template:
        deps = [created[i].id for i in step.get("deps", []) if i < len(created)]
        task = plan.add(
            step["d"],
            depends_on=deps,
            max_attempts=step.get("attempts", 2),
            strategies=step.get("strategies"),
            tool_hint=step.get("hint", ""),
            kind=objective.kind,
        )
        created.append(task)

    # Unresolved facts become explicit first-class work, so they get answered
    # rather than assumed.
    if objective.unknowns and created:
        clarify = plan.add(
            "Resolve what the request leaves unspecified: "
            + "; ".join(objective.unknowns[:3]),
            task_id="t_clarify", priority=_PRIORITY_VALUE[I.PRIORITY_URGENT],
            kind=objective.kind,
        )
        plan.order.remove(clarify.id)
        plan.order.insert(0, clarify.id)
        created[0].depends_on.append(clarify.id)

    if created and objective.success_conditions:
        created[-1].success_criteria = list(objective.success_conditions)

    return plan
