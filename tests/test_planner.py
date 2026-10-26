"""Stage 5b: dynamic planning and replanning.

A failing step must not end the run. The planner retries, then switches
strategy, then blocks — and blocking something must block whatever was waiting
on it rather than letting the agent march on and report success.
"""

from pinpoint.agent import intent as I, planner as P


def simple_plan():
    plan = P.Plan()
    a = plan.add("first")
    b = plan.add("second", depends_on=[a.id])
    c = plan.add("third", depends_on=[b.id])
    return plan, a, b, c


# ── Templates ─────────────────────────────────────────────────────────────────

def test_fix_plan_has_reproduce_and_verify_steps():
    plan = P.plan_for(I.parse("my project won't build"))
    text = plan.render().lower()
    assert "reproduce" in text and "verify" in text


def test_communicate_plan_resolves_recipient_before_sending():
    plan = P.plan_for(I.parse("Text John that I'm running late"))
    ids = plan.order
    resolve = next(t for t in plan.tasks.values() if "Resolve the recipient" in t.description)
    send = next(t for t in plan.tasks.values() if t.description == "Send it")
    assert ids.index(resolve.id) < ids.index(send.id)


def test_communicate_plan_requires_provider_confirmation_step():
    plan = P.plan_for(I.parse("Text John hello"))
    assert any("provider accepted" in t.description for t in plan.tasks.values())


def test_unknowns_become_an_explicit_first_task():
    plan = P.plan_for(I.parse("Text John that I'm late"))
    assert plan.order[0] == "t_clarify"


def test_clarify_task_gates_the_rest():
    plan = P.plan_for(I.parse("Text John that I'm late"))
    ready = plan.ready()
    assert ready[0].id == "t_clarify"


def test_monitor_plan_defines_healthy_first():
    plan = P.plan_for(I.parse("Monitor my server and restart it if it crashes"))
    assert "healthy" in plan.render()


def test_final_task_carries_success_criteria():
    plan = P.plan_for(I.parse("build me a game"))
    last = plan.tasks[plan.order[-1]]
    assert last.success_criteria


def test_fix_plan_has_alternative_strategies():
    plan = P.plan_for(I.parse("fix the build"))
    assert any(len(t.strategies) > 1 for t in plan.tasks.values())


# ── Dependency scheduling ─────────────────────────────────────────────────────

def test_only_dependency_free_tasks_are_ready():
    plan, a, b, c = simple_plan()
    assert [t.id for t in plan.ready()] == [a.id]


def test_completing_a_task_unlocks_the_next():
    plan, a, b, c = simple_plan()
    plan.complete(a.id)
    assert [t.id for t in plan.ready()] == [b.id]


def test_higher_priority_runs_first():
    plan = P.Plan()
    plan.add("low", priority=0)
    urgent = plan.add("urgent", priority=3)
    assert plan.ready()[0].id == urgent.id


def test_active_task_is_returned_before_new_ones():
    plan, a, b, c = simple_plan()
    plan.start(a.id)
    assert plan.next_task().id == a.id


def test_starting_a_task_counts_against_the_action_budget():
    plan, a, _, _ = simple_plan()
    plan.start(a.id)
    assert plan.progress()["actions_used"] == 1


# ── Failure, retry, strategy switching ────────────────────────────────────────

def test_first_failure_retries():
    plan = P.Plan()
    task = plan.add("flaky", max_attempts=2)
    plan.start(task.id)
    outcome = plan.fail(task.id, "boom")
    assert outcome.action == "retry" and plan.tasks[task.id].status == P.PENDING


def test_exhausting_attempts_switches_strategy():
    plan = P.Plan()
    task = plan.add("hard", max_attempts=1, strategies=["approach A", "approach B"])
    plan.start(task.id)
    outcome = plan.fail(task.id, "A didn't work")
    assert outcome.action == "new_strategy"
    assert plan.tasks[task.id].strategy == "approach B"


def test_strategy_switch_resets_the_attempt_counter():
    plan = P.Plan()
    task = plan.add("hard", max_attempts=1, strategies=["A", "B"])
    plan.start(task.id)
    plan.fail(task.id, "nope")
    assert plan.tasks[task.id].attempts == 0


def test_exhausting_every_strategy_fails_the_task():
    plan = P.Plan()
    task = plan.add("doomed", max_attempts=1, strategies=["A"])
    plan.start(task.id)
    outcome = plan.fail(task.id, "still broken")
    assert outcome.action == "blocked" and plan.tasks[task.id].status == P.FAILED


def test_failure_blocks_dependents():
    plan, a, b, c = simple_plan()
    a.max_attempts = 1
    plan.start(a.id)
    outcome = plan.fail(a.id, "cannot proceed")
    assert set(outcome.blocked_tasks) == {b.id, c.id}
    assert plan.tasks[c.id].status == P.BLOCKED


def test_blocked_dependents_record_why():
    plan, a, b, _ = simple_plan()
    a.max_attempts = 1
    plan.start(a.id)
    plan.fail(a.id, "missing dependency")
    assert a.id in plan.tasks[b.id].blocked_reason


def test_a_new_strategy_revives_a_failed_task_and_its_dependents():
    plan, a, b, _ = simple_plan()
    a.max_attempts = 1
    plan.start(a.id)
    plan.fail(a.id, "dead end")
    plan.add_strategy(a.id, "try it from the other direction")
    assert plan.tasks[a.id].status == P.PENDING
    assert plan.tasks[b.id].status == P.PENDING


def test_instruction_includes_the_current_strategy():
    plan = P.Plan()
    task = plan.add("fix it", strategies=["patch the caller"])
    assert "patch the caller" in task.instruction()


def test_replans_are_recorded():
    plan = P.Plan()
    task = plan.add("x", max_attempts=1)
    plan.start(task.id)
    plan.fail(task.id, "boom")
    assert plan.replans and plan.replans[-1]["action"] == "blocked"


# ── Discovered work ───────────────────────────────────────────────────────────

def test_discovered_work_is_inserted_into_the_graph():
    plan, a, b, _ = simple_plan()
    created = plan.insert_after(a.id, ["install the missing dependency"])
    assert created and created[0].depends_on == [a.id]


def test_discovered_work_gates_what_followed():
    plan, a, b, _ = simple_plan()
    created = plan.insert_after(a.id, ["install the missing dependency"])
    assert plan.tasks[b.id].depends_on == [created[0].id]


def test_discovered_work_runs_before_the_original_next_step():
    plan, a, b, _ = simple_plan()
    created = plan.insert_after(a.id, ["install the missing dependency"])
    plan.complete(a.id)
    assert [t.id for t in plan.ready()] == [created[0].id]


def test_insert_after_unknown_task_is_a_noop():
    plan, _, _, _ = simple_plan()
    assert plan.insert_after("nope", ["x"]) == []


# ── Plan state ────────────────────────────────────────────────────────────────

def test_progress_percentages():
    plan, a, b, c = simple_plan()
    plan.complete(a.id)
    assert plan.progress()["percent"] == round(100 / 3, 1)


def test_plan_completes_when_everything_finishes():
    plan, a, b, c = simple_plan()
    for task in (a, b, c):
        plan.complete(task.id)
    assert plan.is_complete() and plan.succeeded()


def test_plan_with_a_failure_is_complete_but_not_successful():
    plan, a, b, c = simple_plan()
    a.max_attempts = 1
    plan.start(a.id)
    plan.fail(a.id, "no")
    assert plan.is_complete() and not plan.succeeded()


def test_stuck_plan_is_detected():
    plan, a, b, c = simple_plan()
    a.max_attempts = 1
    plan.start(a.id)
    plan.fail(a.id, "no")
    assert plan.is_complete()          # everything terminal
    assert plan.blockers()


def test_blockers_explain_themselves():
    plan, a, _, _ = simple_plan()
    a.max_attempts = 1
    plan.start(a.id)
    plan.fail(a.id, "needs sudo")
    assert "needs sudo" in plan.blockers()[0]["reason"]


def test_budget_exhaustion_is_detected():
    plan = P.Plan(max_actions=1)
    task = plan.add("x")
    plan.start(task.id)
    assert plan.budget_exhausted()


def test_skipping_a_task_unlocks_dependents():
    plan, a, b, _ = simple_plan()
    plan.skip(a.id, "not needed")
    assert [t.id for t in plan.ready()] == [b.id]


# ── Persistence ───────────────────────────────────────────────────────────────

def test_plan_round_trips_through_json():
    plan = P.plan_for(I.parse("fix my build"))
    first = plan.order[0]
    plan.start(first)
    plan.complete(first, "done")
    restored = P.Plan.from_dict(plan.to_dict())
    assert restored.order == plan.order
    assert restored.tasks[first].status == P.DONE
    assert restored.objective.kind == plan.objective.kind


def test_restored_plan_resumes_at_the_right_task():
    plan = P.plan_for(I.parse("build me a game"))
    first = plan.order[0]
    plan.complete(first)
    restored = P.Plan.from_dict(plan.to_dict())
    assert restored.next_task().id == plan.next_task().id


def test_render_marks_status():
    plan, a, _, _ = simple_plan()
    plan.complete(a.id)
    assert "✓" in plan.render()
