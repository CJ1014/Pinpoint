"""Stage 10a: the execution loop end to end.

These are the acceptance scenarios: a multi-step task that succeeds, a tool
failure that recovers, a wrong plan that replans, an external action that needs
approval, an ambiguous recipient, a restart mid-task, a permission denial, and
an emergency stop during execution.
"""

import pytest

from pinpoint.action import executor as EX, result as R
from pinpoint.agent import intent as I, orchestrator as O, planner as P
from pinpoint.memory import checkpoint as CP, failures as F, store as MS
from pinpoint.security import approval, emergency_stop, permissions


@pytest.fixture(autouse=True)
def clean():
    emergency_stop.reset_for_tests()
    approval.reset_for_tests()
    permissions.set_profile(permissions.AUTONOMOUS, source=permissions.HUMAN)
    yield
    emergency_stop.reset_for_tests()
    approval.reset_for_tests()


class Tools:
    """A scriptable fake toolset. Records every call it receives."""

    def __init__(self, tmp_path, failures=None):
        self.tmp = tmp_path
        self.calls = []
        self.failures = failures or {}     # tool -> list of outputs (strings)

    def __call__(self, tool, params):
        self.calls.append((tool, params))
        queued = self.failures.get(tool)
        if queued:
            return queued.pop(0)
        if tool == "write_file":
            target = self.tmp / params["filename"]
            target.write_text(params.get("content", "x"))
            return f"Written {len(params.get('content', 'x'))} chars to {target}"
        if tool in ("run_shell", "run_python", "run_tests"):
            return "stdout:\nall good\nexit code: 0"
        return "ok"


def build(tmp_path, step_fn, tools=None, **kwargs):
    tools = tools or Tools(tmp_path)
    executor = EX.Executor(dispatch=tools, session="test")
    return O.Orchestrator(executor, step_fn, session="test", **kwargs), tools


def one_write(task, context):
    """A step function that writes a file for every task."""
    return [("write_file", {"filename": f"{task.id}.txt", "content": "done"})]


def nothing(task, context):
    return []


# ── Scenario 1: a multi-step task that succeeds ───────────────────────────────

def test_multi_step_task_completes(tmp_path):
    orchestrator, tools = build(tmp_path, one_write)
    report = orchestrator.run("build me a small html game")
    assert report.status == O.COMPLETED and report.ok


def test_completion_runs_every_planned_step(tmp_path):
    orchestrator, tools = build(tmp_path, one_write)
    report = orchestrator.run("build me a small html game")
    assert len(tools.calls) == len(report.plan.tasks)


def test_completed_run_reports_naturally(tmp_path):
    orchestrator, _ = build(tmp_path, one_write)
    message = orchestrator.run("build me a small html game").message()
    assert "[TOOL CALL]" not in message and message.startswith("Done")


def test_reasoning_only_steps_complete_without_actions(tmp_path):
    orchestrator, tools = build(tmp_path, nothing)
    assert orchestrator.run("build me a game").status == O.COMPLETED
    assert tools.calls == []


def test_successful_run_learns_a_procedure(tmp_path):
    orchestrator, _ = build(tmp_path, one_write)
    orchestrator.run("build me a small html game")
    assert orchestrator.memory.procedures_for("build me a small html game")


# ── Scenario 2: a tool failure followed by recovery ───────────────────────────

def test_transient_failure_is_retried_and_succeeds(tmp_path):
    tools = Tools(tmp_path, failures={"run_shell": ["Error: something odd"]})

    def step(task, context):
        return [("run_shell", {"command": "make"})]

    orchestrator, _ = build(tmp_path, step, tools)
    report = orchestrator.run("fix my build")
    assert report.status == O.COMPLETED


def test_missing_dependency_inserts_a_real_install_step(tmp_path):
    tools = Tools(tmp_path, failures={
        "run_shell": ["ModuleNotFoundError: No module named 'flask'"]})

    def step(task, context):
        return [("run_shell", {"command": "python app.py"})]

    orchestrator, _ = build(tmp_path, step, tools)
    report = orchestrator.run("fix my build")
    assert any("flask" in t.description for t in report.plan.tasks.values())


def test_failures_are_recorded_as_postmortems(tmp_path):
    tools = Tools(tmp_path, failures={"run_shell": ["Error: boom"]})

    def step(task, context):
        return [("run_shell", {"command": "make"})]

    orchestrator, _ = build(tmp_path, step, tools)
    orchestrator.run("fix my build")
    assert orchestrator.failures.all()


def test_permanent_failure_escalates_with_a_question(tmp_path):
    def step(task, context):
        return [("run_shell", {"command": "make"})]

    tools = Tools(tmp_path, failures={"run_shell": ["Permission denied"] * 20})
    orchestrator, _ = build(tmp_path, step, tools)
    report = orchestrator.run("fix my build")
    assert report.status == O.BLOCKED and report.question


def test_escalation_message_says_what_is_needed(tmp_path):
    def step(task, context):
        return [("run_shell", {"command": "make"})]

    tools = Tools(tmp_path, failures={"run_shell": ["Permission denied"] * 20})
    orchestrator, _ = build(tmp_path, step, tools)
    message = orchestrator.run("fix my build").message()
    assert "couldn't finish" in message.lower()


def test_run_does_not_loop_forever_on_a_hopeless_task(tmp_path):
    def step(task, context):
        return [("run_shell", {"command": "make"})]

    tools = Tools(tmp_path, failures={"run_shell": ["Error: nope"] * 500})
    orchestrator, _ = build(tmp_path, step, tools, max_iterations=40)
    report = orchestrator.run("fix my build")
    assert report.iterations < 40 and report.status == O.BLOCKED


# ── Scenario 3: replanning ────────────────────────────────────────────────────

def test_repeated_failure_switches_strategy(tmp_path):
    tools = Tools(tmp_path, failures={"run_shell": ["SyntaxError: invalid syntax"] * 6})

    def step(task, context):
        return [("run_shell", {"command": "build"})]

    orchestrator, _ = build(tmp_path, step, tools)
    report = orchestrator.run("fix my build")
    from pinpoint.agent import recovery as RC
    assert any(d.action == RC.NEW_STRATEGY for d in orchestrator.recovery.decisions)
    assert any("switching approach" in note
               for task in report.plan.tasks.values() for note in task.notes)


def test_retry_precedes_a_strategy_switch(tmp_path):
    tools = Tools(tmp_path, failures={"run_shell": ["SyntaxError: invalid syntax"] * 6})

    def step(task, context):
        return [("run_shell", {"command": "build"})]

    orchestrator, _ = build(tmp_path, step, tools)
    orchestrator.run("fix my build")
    from pinpoint.agent import recovery as RC
    actions = [d.action for d in orchestrator.recovery.decisions]
    assert actions.index(RC.RETRY) < actions.index(RC.NEW_STRATEGY)


def test_step_function_can_read_the_plan(tmp_path):
    seen = {}

    def step(task, context):
        seen["context"] = context
        return []

    orchestrator, _ = build(tmp_path, step)
    orchestrator.run("build me a game")
    assert "GOAL:" in seen["context"]


def test_context_carries_prior_observations(tmp_path):
    contexts = []

    def step(task, context):
        contexts.append(context)
        return [("write_file", {"filename": f"{task.id}.txt", "content": "x"})]

    orchestrator, _ = build(tmp_path, step)
    orchestrator.run("build me a game")
    assert "Observed so far" in contexts[-1]


def test_step_function_exception_is_a_failure_not_a_crash(tmp_path):
    def step(task, context):
        raise RuntimeError("model returned nonsense")

    orchestrator, _ = build(tmp_path, step)
    report = orchestrator.run("build me a game")
    assert report.status == O.BLOCKED


# ── Scenario 4: approval for an external action ───────────────────────────────

def test_external_action_requires_approval_even_when_autonomous(tmp_path):
    contacts_asked = []

    def step(task, context):
        return [("send_message", {"to": "+15550001111", "body": "running late"})]

    approval.set_hook(lambda req: contacts_asked.append(req) or False)
    orchestrator, tools = build(tmp_path, step)
    report = orchestrator.run("text +15550001111 that I'm running late")
    assert contacts_asked and report.status == O.BLOCKED


def test_denied_action_is_not_executed(tmp_path):
    def step(task, context):
        return [("send_message", {"to": "+15550001111", "body": "hi"})]

    approval.set_hook(lambda req: False)
    orchestrator, tools = build(tmp_path, step)
    orchestrator.run("text +15550001111 that I'm late")
    assert ("send_message", {"to": "+15550001111", "body": "hi"}) not in tools.calls


def test_send_without_a_provider_is_unavailable_not_attempted(tmp_path, monkeypatch):
    """No provider configured means the tool is never called and never claimed."""
    monkeypatch.delenv("PINPOINT_SMS_PROVIDER", raising=False)

    def step(task, context):
        return [("send_message", {"to": "+15550001111", "body": "hi"})]

    approval.set_hook(lambda req: True)
    orchestrator, tools = build(tmp_path, step)
    report = orchestrator.run("text +15550001111 that I'm late")
    sends = [r for r in report.results if r.tool == "send_message"]
    assert sends and sends[0].status == R.UNAVAILABLE
    assert tools.calls == []


@pytest.fixture
def sms_configured():
    """Install a configured provider so the capability preflight passes."""
    from pinpoint.communication import manager as CM
    from pinpoint.communication.providers import base as CB

    class Configured(CB.Provider):
        name = "test-sms"
        kind = CB.SMS

        def available(self):
            return True

    CM.set_provider(CB.SMS, Configured())
    yield
    CM.reset_providers()


def test_approved_send_without_confirmation_is_unverified(tmp_path, sms_configured):
    def step(task, context):
        return [("send_message", {"to": "+15550001111", "body": "hi"})]

    approval.set_hook(lambda req: True)
    tools = Tools(tmp_path, failures={"send_message": ["Message sent successfully!"] * 9})
    orchestrator, _ = build(tmp_path, step, tools)
    report = orchestrator.run("text +15550001111 that I'm late")
    sends = [r for r in report.results if r.tool == "send_message"]
    assert sends and all(r.status == R.UNVERIFIED for r in sends)


def test_unconfirmed_send_is_never_reported_as_done(tmp_path, sms_configured):
    def step(task, context):
        return [("send_message", {"to": "+15550001111", "body": "hi"})]

    approval.set_hook(lambda req: True)
    tools = Tools(tmp_path, failures={"send_message": ["Message sent successfully!"] * 9})
    orchestrator, _ = build(tmp_path, step, tools)
    report = orchestrator.run("text +15550001111 that I'm late")
    assert report.status != O.COMPLETED


def test_confirmed_send_completes(tmp_path, sms_configured):
    def step(task, context):
        return [("send_message", {"to": "+15550001111", "body": "hi"})]

    approval.set_hook(lambda req: True)
    tools = Tools(tmp_path, failures={
        "send_message": ["sms accepted by test-sms. confirmation_id: SM7788990011"] * 9})
    orchestrator, _ = build(tmp_path, step, tools)
    report = orchestrator.run("text +15550001111 that I'm late")
    sends = [r for r in report.results if r.tool == "send_message"]
    assert sends and sends[0].status == R.SUCCESS


# ── Scenario 5: an ambiguous recipient ────────────────────────────────────────

def test_ambiguous_recipient_stops_and_asks(tmp_path):
    orchestrator, tools = build(tmp_path, one_write)
    report = orchestrator.run("Text John that I'm running late")
    assert report.status == O.NEEDS_INPUT and "John" in report.question
    assert tools.calls == []


def test_needs_input_renders_as_a_question(tmp_path):
    orchestrator, _ = build(tmp_path, one_write)
    assert orchestrator.run("Text John that I'm late").message().endswith("?")


# ── Scenario 6: permission denial and emergency stop ──────────────────────────

def test_red_action_is_blocked_and_recovered_from(tmp_path):
    def step(task, context):
        return [("run_shell", {"command": "rm -rf /"})]

    orchestrator, tools = build(tmp_path, step)
    report = orchestrator.run("clean up the project")
    assert tools.calls == [] and report.status == O.BLOCKED


def test_emergency_stop_halts_mid_run(tmp_path):
    def step(task, context):
        emergency_stop.engage("CJ said stop everything", source="human")
        return [("write_file", {"filename": f"{task.id}.txt", "content": "x"})]

    orchestrator, _ = build(tmp_path, step)
    report = orchestrator.run("build me a game")
    assert report.status == O.STOPPED


def test_stopped_run_saves_a_checkpoint(tmp_path):
    def step(task, context):
        emergency_stop.engage("stop", source="human")
        return []

    orchestrator, _ = build(tmp_path, step)
    assert orchestrator.run("build me a game").checkpoint_id


def test_stopped_run_says_so_plainly(tmp_path):
    def step(task, context):
        emergency_stop.engage("stop", source="human")
        return []

    orchestrator, _ = build(tmp_path, step)
    assert "Stopped" in orchestrator.run("build me a game").message()


# ── Scenario 7: restart mid-task ──────────────────────────────────────────────

def test_unfinished_run_leaves_a_resumable_checkpoint(tmp_path):
    def step(task, context):
        return [("run_shell", {"command": "make"})]

    tools = Tools(tmp_path, failures={"run_shell": ["Permission denied"] * 20})
    orchestrator, _ = build(tmp_path, step, tools)
    report = orchestrator.run("fix my build")
    assert CP.load(report.checkpoint_id) is not None


def test_resume_picks_up_where_it_left_off(tmp_path):
    def failing(task, context):
        return [("run_shell", {"command": "make"})]

    tools = Tools(tmp_path, failures={"run_shell": ["Permission denied"] * 20})
    first, _ = build(tmp_path, failing, tools)
    first.run("fix my build")

    second, _ = build(tmp_path, nothing)
    resumed = O.resume(second)
    assert resumed is not None and resumed.plan is not None


def test_resume_reverifies_recorded_effects(tmp_path):
    """A file the checkpoint says exists, but doesn't any more, reopens its task."""
    def write_then_fail(task, context):
        if task.id == "t_probe":
            return []
        return [("write_file", {"filename": "artifact.txt", "content": "hello"}),
                ("run_shell", {"command": "boom"})]

    tools = Tools(tmp_path, failures={"run_shell": ["Permission denied"] * 20})
    orchestrator, _ = build(tmp_path, write_then_fail, tools)
    report = orchestrator.run("build me a game")

    checkpoint = CP.load(report.checkpoint_id)
    assert checkpoint.effects, "the checkpoint should record the file it wrote"

    (tmp_path / "artifact.txt").unlink()
    discrepancies = CP.verify_against_reality(checkpoint)
    assert discrepancies and "artifact.txt" in discrepancies[0].detail


def test_resume_reopens_a_task_whose_effect_vanished(tmp_path):
    plan = P.plan_for(I.parse("build me a game"))
    for task_id in plan.order:
        plan.complete(task_id, summary="wrote artifact.txt")
    checkpoint = CP.save(plan, session="1")
    checkpoint.effects = [{"kind": CP.FILE_EFFECT,
                           "target": str(tmp_path / "gone.txt"), "tool": "write_file"}]
    restored, discrepancies = CP.resume(checkpoint)
    assert discrepancies
    assert any(t.status == P.PENDING for t in restored.tasks.values())


def test_checkpoint_resume_text_is_readable(tmp_path):
    plan = P.plan_for(I.parse("fix my build"))
    plan.complete(plan.order[0], "inspected the project")
    checkpoint = CP.save(plan, session="3", lessons=["check deps first"])
    text = checkpoint.resume_text()
    assert "Resuming" in text and "check deps first" in text


# ── Reporting ─────────────────────────────────────────────────────────────────

def test_report_serializes(tmp_path):
    orchestrator, _ = build(tmp_path, one_write)
    data = orchestrator.run("build me a game").to_dict()
    assert data["status"] == O.COMPLETED and data["actions"]


def test_events_are_emitted_for_observers(tmp_path):
    seen = []
    orchestrator, _ = build(tmp_path, one_write, on_event=lambda kind, data: seen.append(kind))
    orchestrator.run("build me a game")
    assert "plan" in seen and "task" in seen and "finished" in seen


def test_observer_errors_do_not_break_the_run(tmp_path):
    def boom(kind, data):
        raise RuntimeError("observer exploded")

    orchestrator, _ = build(tmp_path, one_write, on_event=boom)
    assert orchestrator.run("build me a game").status == O.COMPLETED


def test_memory_is_consolidated_at_the_end(tmp_path):
    orchestrator, _ = build(tmp_path, one_write)
    orchestrator.run("build me a game")
    assert orchestrator.memory.all(MS.WORKING) == []
    assert orchestrator.memory.all(MS.EPISODIC)
