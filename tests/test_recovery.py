"""Stage 6: failure memory and bounded recovery."""

import pytest

from pinpoint.action import result as R
from pinpoint.agent import intent as I, planner as P, recovery as RC
from pinpoint.memory import failures as F


@pytest.fixture
def mem():
    return F.FailureMemory()


@pytest.fixture
def engine(mem):
    return RC.RecoveryEngine(memory=mem)


def failed(tool="run_python", error="", output="", status=R.FAILED):
    return R.ActionResult(tool=tool, status=status, error=error, output=output)


# ── Classification ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("ModuleNotFoundError: No module named 'flask'", F.MISSING_DEPENDENCY),
    ("bash: gradle: command not found", F.MISSING_DEPENDENCY),
    ("OSError: [Errno 98] Address already in use", F.PORT_IN_USE),
    ("PermissionError: [Errno 13] Permission denied", F.PERMISSION),
    ("FileNotFoundError: No such file or directory: 'a.txt'", F.FILE_NOT_FOUND),
    ("  File \"x.py\", line 3\n    SyntaxError: invalid syntax", F.SYNTAX),
    ("TypeError: unsupported operand type(s)", F.TYPE_ERROR),
    ("AssertionError: expected 3 got 4", F.ASSERTION),
    ("requests.exceptions.ConnectionError: Connection refused", F.NETWORK),
    ("HTTP 401 Unauthorized", F.AUTH),
    ("Command timed out after 300 seconds", F.TIMEOUT),
    ("OSError: [Errno 28] No space left on device", F.DISK),
    ("something entirely novel happened", F.UNKNOWN),
])
def test_error_classification(text, expected):
    assert F.classify(text) == expected


def test_every_category_has_a_profile():
    for category in (F.MISSING_DEPENDENCY, F.PERMISSION, F.NETWORK, F.UNKNOWN):
        profile = F.profile_for(category)
        assert profile.strategies and profile.lesson


# ── Failure memory ────────────────────────────────────────────────────────────

def test_record_captures_a_postmortem(mem):
    failure = mem.record(error="No module named 'flask'", goal="build the app")
    assert failure.category == F.MISSING_DEPENDENCY
    assert failure.root_cause and failure.lesson


def test_failures_persist(mem):
    mem.record(error="boom", goal="g")
    assert len(F.FailureMemory().all()) == 1


def test_similar_finds_the_same_category(mem):
    mem.record(error="No module named 'flask'", goal="a")
    matches = mem.similar("ModuleNotFoundError: No module named 'django'")
    assert len(matches) == 1


def test_similar_ignores_unrelated_categories(mem):
    mem.record(error="Permission denied", goal="a")
    assert mem.similar("No module named 'x'") == []


def test_resolution_raises_confidence(mem):
    failure = mem.record(error="No module named 'flask'", goal="a")
    before = failure.confidence
    updated = mem.mark_resolved(failure.id, "installed flask")
    assert updated.confidence > before and updated.resolved


def test_disputing_lowers_confidence(mem):
    failure = mem.record(error="No module named 'flask'", goal="a")
    updated = mem.dispute(failure.id, "the real cause was a bad path")
    assert updated.confidence < failure.confidence and updated.disputed == 1


def test_lessons_are_not_treated_as_facts(mem):
    failure = mem.record(error="No module named 'flask'", goal="a")
    advice = mem.advice_for("build a flask app")
    assert advice and "check it still applies" in advice[0]


def test_repeatedly_disputed_lessons_stop_being_offered(mem):
    failure = mem.record(error="No module named 'flask'", goal="build a flask app")
    for _ in range(3):
        mem.dispute(failure.id)
    assert mem.advice_for("build a flask app") == []


def test_advice_prefers_lessons_from_related_goals(mem):
    mem.record(error="Permission denied", goal="deploy the website")
    mem.record(error="No module named 'flask'", goal="build the flask server")
    advice = mem.advice_for("build the flask server again")
    assert "dependencies" in advice[0]


def test_credible_lessons_deduplicate(mem):
    for _ in range(3):
        mem.record(error="No module named 'x'", goal="a")
    assert len(mem.credible_lessons()) == 1


def test_stats_summarise_categories(mem):
    mem.record(error="No module named 'x'", goal="a")
    mem.record(error="No module named 'y'", goal="b")
    mem.record(error="Permission denied", goal="c")
    stats = mem.stats()
    assert stats["total"] == 3 and stats["most_common"] == F.MISSING_DEPENDENCY


# ── Diagnosis ─────────────────────────────────────────────────────────────────

def test_diagnosis_identifies_missing_dependency(engine):
    diagnosis = engine.diagnose(failed(error="ModuleNotFoundError: No module named 'flask'"))
    assert diagnosis.category == F.MISSING_DEPENDENCY and diagnosis.prerequisite


def test_blocked_action_diagnoses_as_policy(engine):
    diagnosis = engine.diagnose(R.ActionResult(tool="run_shell", status=R.BLOCKED,
                                               error="blocked by guardrail"))
    assert diagnosis.category == F.BLOCKED_BY_POLICY and diagnosis.escalate


def test_unavailable_capability_diagnoses_as_capability(engine):
    diagnosis = engine.diagnose(R.ActionResult(tool="type_text", status=R.UNAVAILABLE,
                                               error="pyautogui is not installed"))
    assert diagnosis.category == F.CAPABILITY_MISSING


def test_unverified_result_is_its_own_category(engine):
    diagnosis = engine.diagnose(R.ActionResult(tool="send_message", status=R.UNVERIFIED))
    assert diagnosis.category == F.UNVERIFIED_EFFECT


def test_diagnosis_counts_prior_occurrences(engine, mem):
    mem.record(error="No module named 'flask'", goal="a")
    diagnosis = engine.diagnose(failed(error="No module named 'django'"))
    assert diagnosis.prior_failures == 1


# ── Recovery decisions ────────────────────────────────────────────────────────

def build_plan():
    plan = P.plan_for(I.parse("fix my build"))
    task = plan.ready()[0]
    plan.start(task.id)
    return plan, task


def test_missing_dependency_inserts_an_install_step(engine):
    plan, task = build_plan()
    decision = engine.recover(plan, task,
                              failed(error="ModuleNotFoundError: No module named 'flask'"))
    assert decision.action == RC.PREREQUISITE
    assert "flask" in decision.insert_tasks[0]


def test_prerequisite_is_added_to_the_plan(engine):
    plan, task = build_plan()
    decision = engine.recover(plan, task, failed(error="No module named 'flask'"))
    engine.apply(plan, task, decision)
    assert any("flask" in t.description for t in plan.tasks.values())


def test_failed_task_reruns_after_the_prerequisite(engine):
    plan, task = build_plan()
    decision = engine.recover(plan, task, failed(error="No module named 'flask'"))
    engine.apply(plan, task, decision)
    prerequisite = next(t for t in plan.tasks.values() if "flask" in t.description)
    assert prerequisite.id in task.depends_on and task.status == P.PENDING


def test_missing_command_is_named_in_the_prerequisite(engine):
    plan, task = build_plan()
    decision = engine.recover(plan, task, failed(error="gradle: command not found"))
    assert "gradle" in decision.insert_tasks[0]


def test_permission_failure_escalates(engine):
    plan, task = build_plan()
    decision = engine.recover(plan, task, failed(error="Permission denied"))
    assert decision.action == RC.ESCALATE and decision.question


def test_escalation_explains_what_is_stuck(engine):
    plan, task = build_plan()
    decision = engine.recover(plan, task, failed(error="Permission denied"))
    assert task.description[:20] in decision.question


def test_policy_block_escalates_rather_than_retrying(engine):
    plan, task = build_plan()
    decision = engine.recover(plan, task,
                              R.ActionResult(tool="run_shell", status=R.BLOCKED,
                                             error="blocked by guardrail"))
    assert decision.action == RC.ESCALATE


def test_generic_failure_retries_within_budget(engine):
    plan, task = build_plan()
    decision = engine.recover(plan, task, failed(error="something odd happened"))
    assert decision.action == RC.RETRY


def test_repeated_same_failure_switches_strategy(engine):
    plan = P.plan_for(I.parse("fix my build"))
    task = next(t for t in plan.tasks.values() if len(t.strategies) > 1)
    for _ in range(3):
        plan.start(task.id)
        decision = engine.recover(plan, task, failed(error="SyntaxError: invalid syntax"))
        engine.apply(plan, task, decision)
    assert decision.action == RC.NEW_STRATEGY


def test_recovery_does_not_loop_forever(engine):
    plan = P.Plan()
    task = plan.add("impossible", max_attempts=1)
    actions = []
    for _ in range(20):
        plan.start(task.id)
        decision = engine.recover(plan, task, failed(error="mysterious failure"))
        engine.apply(plan, task, decision)
        actions.append(decision.action)
        if decision.action == RC.ESCALATE:
            break
    assert RC.ESCALATE in actions


def test_recovery_budget_is_enforced():
    engine = RC.RecoveryEngine(max_recoveries=2)
    plan = P.Plan()
    task = plan.add("x", max_attempts=99)
    last = None
    for _ in range(4):
        plan.start(task.id)
        last = engine.recover(plan, task, failed(error="odd"))
    assert last.action == RC.ESCALATE and "budget" in last.detail


def test_escalation_blocks_the_task(engine):
    plan, task = build_plan()
    decision = engine.recover(plan, task, failed(error="Permission denied"))
    engine.apply(plan, task, decision)
    assert plan.tasks[task.id].status == P.BLOCKED


def test_strategy_switch_is_applied_to_the_plan(engine):
    plan = P.Plan()
    task = plan.add("do it", max_attempts=1, strategies=["A", "B"])
    plan.start(task.id)
    decision = engine.recover(plan, task, failed(error="odd"))
    engine.apply(plan, task, decision)
    assert task.strategy == "B" and task.status == P.PENDING


def test_successful_recovery_strengthens_the_lesson(engine, mem):
    plan, task = build_plan()
    decision = engine.recover(plan, task, failed(error="No module named 'flask'"))
    engine.confirm_worked(decision.failure_id, "installed flask and retried")
    stored = next(f for f in mem.all() if f.id == decision.failure_id)
    assert stored.resolved and stored.confidence > 0.65


def test_every_recovery_records_a_postmortem(engine, mem):
    plan, task = build_plan()
    engine.recover(plan, task, failed(error="No module named 'flask'"))
    assert len(mem.all()) == 1


def test_recovery_stats(engine):
    plan, task = build_plan()
    engine.recover(plan, task, failed(error="Permission denied"))
    stats = engine.stats()
    assert stats["recoveries"] == 1 and stats["escalations"] == 1
