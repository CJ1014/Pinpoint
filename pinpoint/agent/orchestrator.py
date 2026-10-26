"""The execution loop.

    UNDERSTAND → PLAN → CHECK CAPABILITY → EXECUTE → OBSERVE → VERIFY
              → LEARN → REPLAN → CONTINUE → COMPLETE / BLOCKED

Control flow lives here and is deterministic. Choosing *which tool* advances a
task is the model's job, supplied as ``step_fn`` — which is also what makes the
loop testable: a stub step function exercises every path without an LLM.

One user request is not one model turn. A single run can span hundreds of
actions, retries, replans and escalations.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from pinpoint.action import executor as EX, result as R
from pinpoint.agent import intent as I, persona, planner as P, recovery as RC
from pinpoint.memory import checkpoint as CP, failures as F, store as MS
from pinpoint.security import emergency_stop

COMPLETED = "completed"
BLOCKED = "blocked"
NEEDS_INPUT = "needs_input"
STOPPED = "stopped"
UNAVAILABLE = "unavailable"
UNVERIFIED_COMPLETION = "unverified_completion"

# Objective kinds where finishing means something changed in the world. For
# these, a plan that ran to the end without a single confirmed effect has not
# been completed — it has been narrated.
_EVIDENCE_REQUIRED = (I.FIX, I.BUILD, I.MAINTAIN, I.COMMUNICATE,
                      I.INVESTIGATE, I.RESEARCH, I.MONITOR, I.SCHEDULE)

# What step_fn returns: a list of (tool, params) to run for this task.
Step = Tuple[str, Dict[str, Any]]


@dataclass
class RunReport:
    """Everything about one run, for the user and for the record."""
    status: str
    objective: Optional[I.Objective] = None
    plan: Optional[P.Plan] = None
    results: List[R.ActionResult] = field(default_factory=list)
    question: str = ""
    blockers: List[dict] = field(default_factory=list)
    iterations: int = 0
    lessons: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    checkpoint_id: str = ""
    discrepancies: List[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == COMPLETED

    def message(self) -> str:
        """What to actually say to the user."""
        return persona.report(self)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "objective": self.objective.to_dict() if self.objective else None,
            "progress": self.plan.progress() if self.plan else {},
            "question": self.question, "blockers": self.blockers,
            "iterations": self.iterations, "lessons": self.lessons,
            "checkpoint_id": self.checkpoint_id,
            "actions": [r.to_dict() for r in self.results],
        }


class Orchestrator:
    """Drives an objective from words to a verified outcome, or to an honest stop."""

    def __init__(self, executor: EX.Executor,
                 step_fn: Callable[[P.Task, str], List[Step]],
                 *, memory: Optional[MS.MemoryStore] = None,
                 failure_memory: Optional[F.FailureMemory] = None,
                 recovery_engine: Optional[RC.RecoveryEngine] = None,
                 session: Any = "", max_iterations: int = 60,
                 checkpoint_every: int = 4,
                 on_event: Optional[Callable[[str, dict], None]] = None):
        self.executor = executor
        self.step_fn = step_fn
        self.session = session
        self.max_iterations = max_iterations
        self.checkpoint_every = checkpoint_every
        self.on_event = on_event
        self.checkpoint_id = ""
        self.plan: Optional[P.Plan] = None
        self._memory = memory
        self._failures = failure_memory
        self.recovery = recovery_engine or RC.RecoveryEngine(memory=failure_memory)

        self.results: List[R.ActionResult] = []
        self.observations: List[str] = []
        self.lessons: List[str] = []

    # ── accessors that respect a redirected PINPOINT_HOME ────────────────────

    @property
    def memory(self) -> MS.MemoryStore:
        return self._memory if self._memory is not None else MS.store()

    @property
    def failures(self) -> F.FailureMemory:
        return self._failures if self._failures is not None else F.memory()

    def _emit(self, kind: str, data: dict) -> None:
        if self.on_event is not None:
            try:
                self.on_event(kind, data)
            except Exception:
                pass

    # ── UNDERSTAND ───────────────────────────────────────────────────────────

    def understand(self, request: Any) -> I.Objective:
        return request if isinstance(request, I.Objective) else I.parse(str(request))

    @staticmethod
    def _capability_refusal(objective: I.Objective) -> str:
        """Ask the v2 capability map whether this is even possible here."""
        try:
            import json

            from capability_map import check as capability_check
            verdict = json.loads(capability_check(objective.goal))
            if not verdict.get("can_do", True):
                return persona.unavailable(
                    objective.goal.lower().rstrip("."),
                    verdict.get("workaround", "") or verdict.get("reason", ""))
        except Exception:
            pass
        return ""

    # ── PLAN ─────────────────────────────────────────────────────────────────

    def build_plan(self, objective: I.Objective) -> P.Plan:
        plan = P.plan_for(objective, max_actions=self.max_iterations * 4)
        for advice in self.failures.advice_for(objective.goal):
            self.lessons.append(advice)
        return plan

    def _context(self, task: P.Task, plan: P.Plan) -> str:
        """What the step function needs to choose an action."""
        parts = [plan.render()]
        if task.success_criteria:
            parts.append("Success criteria: " + "; ".join(task.success_criteria))
        memory_block = self.memory.context_block(task.description)
        if memory_block:
            parts.append(memory_block)
        procedures = self.memory.procedures_for(
            plan.objective.goal if plan.objective else task.description)
        if procedures:
            parts.append("Approaches that have worked before:\n"
                         + "\n".join(f"  - {p.content[:160]}" for p in procedures))
        if self.lessons:
            parts.append("Worth keeping in mind:\n"
                         + "\n".join(f"  - {lesson}" for lesson in self.lessons[:3]))
        if self.observations:
            parts.append("Observed so far:\n"
                         + "\n".join(f"  - {o}" for o in self.observations[-5:]))
        return "\n\n".join(parts)

    # ── EXECUTE / OBSERVE / VERIFY ───────────────────────────────────────────

    def _run_task(self, task: P.Task, plan: P.Plan) -> Tuple[bool, Optional[R.ActionResult]]:
        """Run one task's actions. Returns ``(succeeded, first_failure)``."""
        try:
            steps = self.step_fn(task, self._context(task, plan)) or []
        except Exception as exc:
            failure = R.ActionResult(tool="plan_step", status=R.FAILED,
                                     error=f"could not decide what to do: {exc}")
            self.results.append(failure)
            return False, failure

        if not steps:
            # Nothing to do is a legitimate outcome — reasoning-only steps hit it.
            return True, None

        for tool, params in steps:
            result = self.executor.execute(
                tool, params,
                goal=plan.objective.goal if plan.objective else "",
                attempt=task.attempts)
            self.results.append(result)
            self.observations.append(result.as_observation())
            self._emit("action", {"task": task.id, "tool": tool,
                                  "status": result.status})

            self.memory.remember(
                result.as_observation(), tier=MS.WORKING, category="observation",
                relevance=0.6 if result.ok else 0.75,
                confidence=result.confidence, source=f"tool:{tool}",
                session=str(self.session))

            if not result.ok:
                return False, result

        return True, None

    # ── LEARN / REPLAN ───────────────────────────────────────────────────────

    def _handle_failure(self, plan: P.Plan, task: P.Task,
                        failure: R.ActionResult) -> Optional[str]:
        """Recover, replan, or escalate. Returns an escalation question if stuck."""
        goal = plan.objective.goal if plan.objective else ""
        decision = self.recovery.recover(plan, task, failure, goal=goal)
        self._emit("recovery", {"task": task.id, "action": decision.action,
                                "detail": decision.detail})

        if decision.diagnosis and decision.diagnosis.lessons:
            for lesson in decision.diagnosis.lessons:
                if lesson not in self.lessons:
                    self.lessons.append(lesson)

        self.recovery.apply(plan, task, decision)

        if decision.action in (RC.ESCALATE, RC.ABANDON):
            return decision.question or decision.detail
        return None

    def _record_success(self, plan: P.Plan, task: P.Task) -> None:
        plan.complete(task.id, summary=self.observations[-1] if self.observations else "")
        # A recovery that preceded this success earned its lesson.
        for decision in reversed(self.recovery.decisions):
            if decision.failure_id:
                self.recovery.confirm_worked(decision.failure_id,
                                             f"completed after: {decision.detail}")
                break

    # ── the loop ─────────────────────────────────────────────────────────────

    def run(self, request: Any, plan: Optional[P.Plan] = None) -> RunReport:
        objective = self.understand(request)

        question = I.needs_clarification(objective)
        if question:
            return RunReport(NEEDS_INPUT, objective=objective, question=question)

        refusal = self._capability_refusal(objective)
        if refusal:
            return RunReport(UNAVAILABLE, objective=objective, question=refusal)

        plan = plan or self.build_plan(objective)
        self.plan = plan
        plan.resume_interrupted()
        self._emit("plan", {"goal": objective.goal, "tasks": len(plan.tasks)})

        iterations = 0
        escalation = ""
        completed_since_checkpoint = 0

        while iterations < self.max_iterations:
            if emergency_stop.is_engaged():
                return self._halt(objective, plan, iterations)

            if plan.budget_exhausted():
                escalation = ("I've used the action budget for this job without "
                              "finishing it.")
                break

            task = plan.next_task()
            if task is None:
                break

            iterations += 1
            plan.start(task.id)
            self._emit("task", {"id": task.id, "description": task.description,
                                "attempt": task.attempts})

            succeeded, failure = self._run_task(task, plan)

            # The stop can be engaged while a task is mid-flight. It outranks
            # whatever the task was doing, including how that task failed.
            if emergency_stop.is_engaged():
                return self._halt(objective, plan, iterations)

            if succeeded:
                self._record_success(plan, task)
                completed_since_checkpoint += 1
                # Checkpoint as we go. Saving only on exit means a crash loses
                # everything the run had achieved.
                if completed_since_checkpoint >= self.checkpoint_every:
                    completed_since_checkpoint = 0
                    self._checkpoint(plan)
                continue

            escalation = self._handle_failure(plan, task, failure) or ""
            if escalation:
                break

        if plan.is_complete() and plan.succeeded() and not escalation:
            # Every step is marked done — but "done" has to mean something
            # happened. A plan that ran to the end on reasoning alone has no
            # evidence behind it, and saying "Done" would be the exact failure
            # this whole architecture exists to prevent.
            if objective.kind in _EVIDENCE_REQUIRED and not self.evidence():
                return self._finish(
                    UNVERIFIED_COMPLETION, objective, plan, iterations,
                    question="I worked through the plan but nothing I did produced "
                             "a checkable result, so I can't claim it's done. "
                             "Tell me what the finished thing should look like and "
                             "I'll go and make it exist.")
            status = COMPLETED
            self._learn_procedure(plan)
        elif escalation:
            status = BLOCKED
        elif plan.is_complete():
            status = BLOCKED
        else:
            status = BLOCKED
            escalation = escalation or ("I stopped before finishing — "
                                        "I'd rather check in than keep guessing.")

        return self._finish(status, objective, plan, iterations, question=escalation)

    def _halt(self, objective: I.Objective, plan: P.Plan,
              iterations: int) -> RunReport:
        """Stop cleanly: mark in-flight work interrupted, then checkpoint it."""
        interrupted = plan.interrupt("emergency stop engaged")
        self._emit("interrupted", {"tasks": interrupted})
        return self._finish(STOPPED, objective, plan, iterations,
                            question="Stopped on your say-so.")

    # ── evidence and checkpointing ───────────────────────────────────────────

    def evidence(self) -> List[R.ActionResult]:
        """Confirmed actions that actually changed or observed something.

        Reasoning tools are excluded on purpose: thinking about a task is not
        evidence that the task happened.
        """
        from pinpoint.tools import registry

        out = []
        for result in self.results:
            if result.status != R.SUCCESS:
                continue
            spec = registry.get(result.tool)
            method = spec.verification if spec else registry.V_UNVERIFIABLE
            if method not in (registry.V_NONE, registry.V_UNVERIFIABLE):
                out.append(result)
        return out

    def _checkpoint(self, plan: P.Plan, run_status: str = "") -> str:
        """Save (or overwrite) this run's checkpoint. Never raises."""
        try:
            checkpoint = CP.save(plan, session=self.session,
                                 observations=self.observations,
                                 lessons=self.lessons, results=self.results,
                                 checkpoint_id=self.checkpoint_id,
                                 run_status=run_status)
            self.checkpoint_id = checkpoint.id
            return checkpoint.id
        except Exception:
            return self.checkpoint_id

    def _learn_procedure(self, plan: P.Plan) -> None:
        """A completed plan is a workflow worth keeping."""
        if not plan.objective:
            return
        steps = [plan.tasks[i].description for i in plan.order
                 if plan.tasks[i].status == P.DONE]
        if len(steps) >= 2:
            self.memory.record_procedure(
                f"how to {plan.objective.goal[:60]}", steps,
                goal_kind=plan.objective.kind, outcome="worked")

    def _finish(self, status: str, objective: I.Objective, plan: P.Plan,
                iterations: int, question: str = "") -> RunReport:
        checkpoint_id = self.checkpoint_id
        if status != COMPLETED:
            # Unfinished work gets a checkpoint so it can be picked back up.
            run_status = {STOPPED: "INTERRUPTED", BLOCKED: "BLOCKED",
                          UNVERIFIED_COMPLETION: "UNVERIFIED"}.get(status, "IN_PROGRESS")
            checkpoint_id = self._checkpoint(plan, run_status=run_status)

        self.memory.consolidate(
            session=str(self.session),
            summary=f"{objective.goal} — {status}" if objective else status)

        report = RunReport(
            status=status, objective=objective, plan=plan, results=list(self.results),
            question=question, blockers=plan.blockers() if plan else [],
            iterations=iterations, lessons=list(self.lessons),
            observations=list(self.observations), checkpoint_id=checkpoint_id,
        )
        self._emit("finished", {"status": status, "iterations": iterations})
        return report


# ── Resuming ──────────────────────────────────────────────────────────────────

def resume(orchestrator: Orchestrator,
           checkpoint: Optional[CP.Checkpoint] = None) -> Optional[RunReport]:
    """Pick a checkpointed objective back up, re-verifying it against reality first."""
    checkpoint = checkpoint or CP.latest()
    if checkpoint is None:
        return None

    plan, discrepancies = CP.resume(checkpoint)
    if plan.objective is None:
        return None

    for discrepancy in discrepancies:
        orchestrator.observations.append(
            f"On resume, something the checkpoint assumed no longer holds: "
            f"{discrepancy.detail}")

    report = orchestrator.run(plan.objective, plan=plan)
    report.discrepancies = [d.to_dict() for d in discrepancies]
    return report
