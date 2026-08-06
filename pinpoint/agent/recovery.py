"""Autonomous recovery.

When an action fails, this decides what happens next: retry, fix the
prerequisite that was actually missing, switch strategy, or stop and ask the
human. It is bounded on purpose — attempts per task, recoveries per session,
and a repeat-failure detector — because an agent that never gives up is an
agent that grinds forever on something impossible.

Diagnosis is deterministic: the error text is classified into a root-cause
category, and each category carries its own ordered strategies. Past failures
inform the choice but never override what is being observed right now.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from pinpoint.action import result as R
from pinpoint.agent import planner as P
from pinpoint.memory import failures as F

# Recovery decisions.
RETRY = "retry"
PREREQUISITE = "prerequisite"     # insert work that must happen first
NEW_STRATEGY = "new_strategy"
ESCALATE = "escalate"             # ask the human
ABANDON = "abandon"               # stop pursuing this branch


@dataclass
class Diagnosis:
    """What went wrong, in terms the planner can act on."""
    category: str
    root_cause: str
    strategies: List[str] = field(default_factory=list)
    preventability: str = "low"
    prerequisite: str = ""
    escalate: bool = False
    prior_failures: int = 0
    lessons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"category": self.category, "root_cause": self.root_cause,
                "strategies": self.strategies, "preventability": self.preventability,
                "prerequisite": self.prerequisite, "escalate": self.escalate,
                "prior_failures": self.prior_failures, "lessons": self.lessons}

    def render(self) -> str:
        lines = [f"Root cause: {self.root_cause} ({self.category})"]
        if self.prior_failures:
            lines.append(f"Seen {self.prior_failures} time(s) before")
        if self.strategies:
            lines.append("Options: " + "; ".join(self.strategies[:3]))
        for lesson in self.lessons[:2]:
            lines.append(f"Lesson: {lesson}")
        return "\n".join(lines)


@dataclass
class RecoveryDecision:
    """The action the recovery engine wants the orchestrator to take."""
    action: str
    detail: str = ""
    diagnosis: Optional[Diagnosis] = None
    insert_tasks: List[str] = field(default_factory=list)
    strategy: str = ""
    question: str = ""
    failure_id: str = ""

    def to_dict(self) -> dict:
        return {"action": self.action, "detail": self.detail,
                "insert_tasks": self.insert_tasks, "strategy": self.strategy,
                "question": self.question, "failure_id": self.failure_id,
                "diagnosis": self.diagnosis.to_dict() if self.diagnosis else None}


class RecoveryEngine:
    """Bounded failure recovery with escalation."""

    def __init__(self, memory: Optional[F.FailureMemory] = None,
                 max_recoveries: int = 12, max_repeats: int = 2):
        self._memory = memory
        self.max_recoveries = max_recoveries
        self.max_repeats = max_repeats
        self.recoveries_used = 0
        self.decisions: List[RecoveryDecision] = []
        self._signatures: List[str] = []

    @property
    def memory(self) -> F.FailureMemory:
        return self._memory if self._memory is not None else F.memory()

    # ── diagnosis ────────────────────────────────────────────────────────────

    def diagnose(self, result: R.ActionResult, goal: str = "") -> Diagnosis:
        """Classify a failed action into an actionable root cause."""
        evidence = " ".join(filter(None, [
            result.error,
            result.output[:1500],
            (result.verification or {}).get("detail", ""),
        ]))

        if result.status == R.BLOCKED:
            category = F.BLOCKED_BY_POLICY
        elif result.status == R.DENIED:
            category = F.BLOCKED_BY_POLICY
        elif result.status == R.UNAVAILABLE:
            category = F.CAPABILITY_MISSING
        elif result.status == R.UNVERIFIED:
            category = F.UNVERIFIED_EFFECT
        else:
            category = F.classify(evidence)

        profile = F.profile_for(category)
        prior = self.memory.similar(evidence)
        return Diagnosis(
            category=category,
            root_cause=profile.root_cause,
            strategies=list(profile.strategies),
            preventability=profile.preventability,
            prerequisite=profile.prerequisite,
            escalate=profile.escalate,
            prior_failures=len(prior),
            lessons=[f.lesson for f in prior[:2]],
        )

    # ── decision ─────────────────────────────────────────────────────────────

    def _signature(self, task_id: str, diagnosis: Diagnosis) -> str:
        return f"{task_id}:{diagnosis.category}"

    def recover(self, plan: P.Plan, task: P.Task,
                result: R.ActionResult, goal: str = "") -> RecoveryDecision:
        """Decide how to respond to a failed action, and record the postmortem."""
        diagnosis = self.diagnose(result, goal)
        failure = self.memory.record(
            error=result.error or result.output[:300],
            goal=goal, task=task.description, tool=result.tool,
            action=result.action_id, observation=result.summary(),
            category=diagnosis.category,
        )

        signature = self._signature(task.id, diagnosis)
        repeats = self._signatures.count(signature)
        self._signatures.append(signature)
        self.recoveries_used += 1

        decision = self._choose(plan, task, result, diagnosis, repeats)
        decision.diagnosis = diagnosis
        decision.failure_id = failure.id
        self.decisions.append(decision)
        return decision

    def _choose(self, plan: P.Plan, task: P.Task, result: R.ActionResult,
                diagnosis: Diagnosis, repeats: int) -> RecoveryDecision:
        # Budget first — an exhausted budget stops everything else.
        if self.recoveries_used > self.max_recoveries:
            return RecoveryDecision(
                ESCALATE,
                f"recovery budget spent ({self.max_recoveries} attempts)",
                question=f"I've hit my recovery limit on '{task.description}'. "
                         f"The last problem was: {diagnosis.root_cause}. "
                         f"How would you like me to proceed?")

        # A human-shaped problem stays a human-shaped problem however often it recurs.
        if diagnosis.escalate:
            return RecoveryDecision(
                ESCALATE, f"{diagnosis.category} needs authority PinPoint does not have",
                question=self._question_for(task, diagnosis, result))

        # The same category failing repeatedly means the approach is wrong,
        # not that the attempt was unlucky.
        if repeats >= self.max_repeats:
            if task.has_alternative:
                return RecoveryDecision(
                    NEW_STRATEGY,
                    f"{diagnosis.category} recurred {repeats + 1}x — changing approach",
                    strategy=task.strategies[task.strategy_index + 1])
            return RecoveryDecision(
                ESCALATE,
                f"{diagnosis.category} recurred {repeats + 1}x with no approaches left",
                question=self._question_for(task, diagnosis, result))

        # A missing prerequisite is real work, not a retry.
        if diagnosis.prerequisite and repeats == 0:
            detail = self._prerequisite_detail(diagnosis, result)
            return RecoveryDecision(
                PREREQUISITE, f"{diagnosis.root_cause} — handling it first",
                insert_tasks=[detail])

        if task.attempts < task.max_attempts:
            return RecoveryDecision(
                RETRY, f"retrying after {diagnosis.category} "
                       f"(attempt {task.attempts + 1}/{task.max_attempts})")

        if task.has_alternative:
            return RecoveryDecision(
                NEW_STRATEGY, "attempts exhausted — switching approach",
                strategy=task.strategies[task.strategy_index + 1])

        if diagnosis.strategies:
            return RecoveryDecision(
                NEW_STRATEGY, "no approaches left — adding one from the diagnosis",
                strategy=diagnosis.strategies[0])

        return RecoveryDecision(
            ESCALATE, "nothing left to try",
            question=self._question_for(task, diagnosis, result))

    @staticmethod
    def _prerequisite_detail(diagnosis: Diagnosis, result: R.ActionResult) -> str:
        """Turn a category's generic prerequisite into a concrete task."""
        if diagnosis.category == F.MISSING_DEPENDENCY:
            import re
            blob = f"{result.error} {result.output}"
            match = re.search(r"no module named ['\"]?([\w.\-]+)", blob, re.I)
            if match:
                return f"install the missing Python package '{match.group(1)}'"
            match = re.search(r"([\w.\-]+): command not found", blob, re.I)
            if match:
                return f"install or locate the missing command '{match.group(1)}'"
        if diagnosis.category == F.FILE_NOT_FOUND:
            import re
            match = re.search(r"no such file[^:]*:\s*['\"]?([^\s'\"]+)",
                              f"{result.error} {result.output}", re.I)
            if match:
                return f"find or create the missing path '{match.group(1)}'"
        return diagnosis.prerequisite

    @staticmethod
    def _question_for(task: P.Task, diagnosis: Diagnosis,
                      result: R.ActionResult) -> str:
        """The escalation message — what is stuck, why, and what is needed."""
        detail = result.error or (result.verification or {}).get("detail", "")
        return (f"I'm stuck on: {task.description}.\n"
                f"What's happening: {diagnosis.root_cause}"
                + (f" ({detail[:160]})" if detail else "") + ".\n"
                f"I can't get past this on my own — it needs a decision or access "
                f"I don't have.")

    # ── applying a decision ──────────────────────────────────────────────────

    def apply(self, plan: P.Plan, task: P.Task,
              decision: RecoveryDecision) -> P.ReplanOutcome:
        """Fold a recovery decision back into the plan."""
        if decision.action == PREREQUISITE and decision.insert_tasks:
            created = plan.insert_after(task.id, decision.insert_tasks)
            # The failed task must run again once the prerequisite is met.
            task.status = P.PENDING
            task.attempts = max(0, task.attempts - 1)
            if created:
                task.depends_on = list(dict.fromkeys(task.depends_on + [created[-1].id]))
                # insert_after rewired dependents; undo that for the task itself.
                created[0].depends_on = [d for d in created[0].depends_on
                                         if d != task.id] or []
                plan.order.remove(task.id)
                plan.order.append(task.id)
            return P.ReplanOutcome(PREREQUISITE, task.id, decision.detail)

        if decision.action == NEW_STRATEGY and decision.strategy:
            if decision.strategy not in task.strategies:
                task.strategies.append(decision.strategy)
            task.notes.append(f"switching approach: {decision.strategy[:120]}")
            plan.revive(task.id, task.strategies.index(decision.strategy))
            return P.ReplanOutcome(NEW_STRATEGY, task.id, decision.strategy)

        if decision.action == RETRY:
            task.status = P.PENDING
            task.notes.append(f"retrying: {decision.detail[:120]}")
            return P.ReplanOutcome(RETRY, task.id, decision.detail)

        if decision.action in (ESCALATE, ABANDON):
            plan.block(task.id, decision.detail or "escalated to the human")
            return P.ReplanOutcome(decision.action, task.id, decision.detail)

        return P.ReplanOutcome("noop", task.id, decision.detail)

    def confirm_worked(self, failure_id: str, recovery: str) -> None:
        """Tell failure memory the recovery succeeded, so the lesson gains weight."""
        self.memory.mark_resolved(failure_id, recovery)

    def stats(self) -> dict:
        by_action = {}
        for decision in self.decisions:
            by_action[decision.action] = by_action.get(decision.action, 0) + 1
        return {
            "recoveries": self.recoveries_used,
            "remaining": max(0, self.max_recoveries - self.recoveries_used),
            "by_action": by_action,
            "escalations": by_action.get(ESCALATE, 0),
        }
