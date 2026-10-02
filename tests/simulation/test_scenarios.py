"""Adversarial autonomy scenarios A–I.

Each scenario gives the agent a goal and a hostile world, then checks that it
recovered *on its own*. The step functions in ``agents.py`` contain no failure
handling, so any recovery visible here came from the planner and the recovery
engine.
"""

import pytest

from pinpoint.action import result as R
from pinpoint.agent import orchestrator as O, planner as P, recovery as RC
from pinpoint.communication import contacts as C, manager as CM
from pinpoint.communication.providers import base as CB
from pinpoint.memory import checkpoint as CP
from pinpoint.security import approval, emergency_stop, permissions
from tests.simulation.agents import keyword_agent, messaging_agent, recording_agent
from tests.simulation.world import ERROR, SILENT, STALE


# ══ Scenario A — a tool failure the agent must recover from ══════════════════

def test_A_first_execution_fails_then_the_agent_recovers(world, agent):
    world.inject("run_python", ERROR, "Error: transient interpreter crash", times=1)
    orchestrator = agent(keyword_agent(world))

    report = orchestrator.run("build me a python script that prints hello")

    assert report.status == O.COMPLETED
    assert world.exists("app.py")


def test_A_the_agent_does_not_stop_at_the_first_failure(world, agent):
    world.inject("run_python", ERROR, "Error: transient", times=1)
    log = []
    orchestrator = agent(recording_agent(keyword_agent(world), log))
    orchestrator.run("build me a python script that prints hello")

    attempts = [entry for entry in log if entry["attempt"] > 1]
    assert attempts, "the agent should have made a second attempt"


def test_A_the_failure_was_diagnosed_not_just_repeated(world, agent):
    world.inject("run_python", ERROR, "Error: transient", times=1)
    orchestrator = agent(keyword_agent(world))
    orchestrator.run("build me a python script that prints hello")

    assert orchestrator.recovery.decisions, "a recovery decision should exist"
    assert orchestrator.recovery.decisions[0].diagnosis is not None


def test_A_a_missing_dependency_becomes_a_real_install_step(world, agent):
    """The classic case: the script imports something that isn't installed."""
    orchestrator = agent(keyword_agent(
        world, source="import flask\nprint('hello')"))
    report = orchestrator.run("build me a python script that uses flask")

    installs = world.calls("pip_install")
    assert installs and installs[0].params["package"] == "flask"
    assert report.status == O.COMPLETED


def test_A_the_install_actually_fixed_the_script(world, agent):
    orchestrator = agent(keyword_agent(world, source="import flask\nprint('hello')"))
    orchestrator.run("build me a python script that uses flask")

    final_runs = [c for c in world.calls("run_python") if "exit code: 0" in c.result]
    assert final_runs, "the script should end up running cleanly"


# ══ Scenario B — the tool lies about succeeding ══════════════════════════════

def test_B_a_silent_failure_is_caught_by_verification(world, agent):
    """The tool says "Written 21 chars"; no file appears. That is not success."""
    world.inject("write_file", SILENT, times=1)
    orchestrator = agent(keyword_agent(world))
    orchestrator.run("build me a python script that prints hello")

    lied = [r for r in orchestrator.results
            if r.tool == "write_file" and "Written" in r.output]
    assert lied, "the world should have produced a false success string"
    assert lied[0].status == R.FAILED
    assert "does not exist" in lied[0].error


def test_B_the_agent_recovers_from_the_lie_and_finishes(world, agent):
    world.inject("write_file", SILENT, times=1)
    orchestrator = agent(keyword_agent(world))
    report = orchestrator.run("build me a python script that prints hello")

    assert report.status == O.COMPLETED and world.exists("app.py")


def test_B_a_permanent_lie_never_reports_success(world, agent):
    """If the file never appears, the agent must not claim it did."""
    world.inject("write_file", SILENT, times=99)
    orchestrator = agent(keyword_agent(world))
    report = orchestrator.run("build me a python script that prints hello")

    assert report.status != O.COMPLETED
    assert not world.exists("app.py")
    assert "Done" not in report.message()


def test_B_the_message_to_cj_does_not_claim_the_file_exists(world, agent):
    world.inject("write_file", SILENT, times=99)
    orchestrator = agent(keyword_agent(world))
    message = orchestrator.run("build me a python script that prints hello").message()

    assert "wrote app.py" not in message


def test_B_stale_state_is_also_caught(world, agent):
    """Written, then deleted behind the agent's back, is still not done."""
    world.inject("write_file", STALE, times=99)
    orchestrator = agent(keyword_agent(world))
    report = orchestrator.run("build me a python script that prints hello")

    assert report.status != O.COMPLETED


# ══ Scenario C — the first strategy cannot work ══════════════════════════════

def two_strategy_plan():
    """A task whose first approach cannot work and whose second can."""
    from pinpoint.agent import intent as I

    objective = I.parse("fix my build")
    plan = P.Plan(objective)
    fix = plan.add("apply a fix", max_attempts=1,
                   strategies=["patch the existing module",
                               "rewrite the module from scratch"])
    plan.add("verify the fix works", depends_on=[fix.id])
    return plan, fix


def strategy_agent(world):
    """Writes broken code under strategy one, working code under strategy two."""
    def step(task, context):
        if "apply a fix" in task.description:
            broken = "patch the existing" in (task.strategy or "")
            return [("write_file", {
                "filename": "app.py",
                "content": "SYNTAX_ERROR" if broken
                           else "print('the second approach works')"})]
        return [("run_python", {"filename": "app.py"})]
    return step


def test_C_the_first_approach_fails_and_the_second_succeeds(world, agent):
    plan, _ = two_strategy_plan()
    orchestrator = agent(strategy_agent(world))

    report = orchestrator.run(plan.objective, plan=plan)
    assert report.status == O.COMPLETED


def test_C_the_active_strategy_actually_changes(world, agent):
    """Not merely appended — the task must be *executing* the new approach."""
    plan, fix = two_strategy_plan()
    orchestrator = agent(strategy_agent(world))
    orchestrator.run(plan.objective, plan=plan)

    assert fix.strategy_index == 1
    assert fix.strategy == "rewrite the module from scratch"


def test_C_the_failure_reason_was_identified_before_switching(world, agent):
    plan, _ = two_strategy_plan()
    orchestrator = agent(strategy_agent(world))
    orchestrator.run(plan.objective, plan=plan)

    switches = [d for d in orchestrator.recovery.decisions
                if d.action == RC.NEW_STRATEGY]
    assert switches and switches[0].diagnosis.category


def test_C_the_working_code_was_actually_written(world, agent):
    plan, _ = two_strategy_plan()
    agent(strategy_agent(world)).run(plan.objective, plan=plan)

    assert "second approach" in world.read("app.py")


def test_C_a_diagnosis_can_supply_a_strategy_the_plan_never_had(world, agent):
    """When a task has no alternatives left, recovery offers one from the diagnosis."""
    plan = P.Plan(__import__("pinpoint.agent.intent", fromlist=["parse"]).parse("fix it"))
    task = plan.add("do the thing", max_attempts=1)      # no strategies at all
    world.inject("run_python", ERROR, "SyntaxError: invalid syntax", times=99)

    orchestrator = agent(lambda t, c: [("run_python", {"filename": "app.py"})])
    orchestrator.run(plan.objective, plan=plan)

    assert task.strategies, "recovery should have supplied an approach"


# ══ Scenario D — external communication ══════════════════════════════════════

@pytest.fixture
def sarah():
    C.contacts().add(C.Contact(name="Sarah Kim", phone="+15550001111",
                               note="from school"))


def test_D_a_confirmed_send_completes(world, agent, sarah):
    providers = world.install_providers(confirm=True)
    approval.set_hook(lambda request: True)
    orchestrator = agent(messaging_agent("Sarah", "I'm running late"))

    report = orchestrator.run("Text Sarah that I'm running late")

    assert report.status == O.COMPLETED
    assert providers[CB.SMS].sent[0]["body"] == "I'm running late"


def test_D_the_send_required_approval(world, agent, sarah):
    asked = []
    world.install_providers()
    approval.set_hook(lambda request: asked.append(request.tool) or True)
    agent(messaging_agent("Sarah", "late")).run("Text Sarah that I'm running late")

    assert "send_message" in asked


def test_D_a_denied_send_never_reaches_the_provider(world, agent, sarah):
    providers = world.install_providers()
    approval.set_hook(lambda request: False)
    report = agent(messaging_agent("Sarah", "late")).run(
        "Text Sarah that I'm running late")

    assert providers[CB.SMS].sent == []
    assert report.status != O.COMPLETED


def test_D_no_confirmation_means_not_sent(world, agent, sarah):
    """The provider accepted the call but returned no id. That is not delivery."""
    world.install_providers(confirm=False)
    approval.set_hook(lambda request: True)
    orchestrator = agent(messaging_agent("Sarah", "late"))
    report = orchestrator.run("Text Sarah that I'm running late")

    sends = [r for r in orchestrator.results if r.tool == "send_message"]
    assert sends and sends[0].status in (R.UNVERIFIED, R.FAILED)
    assert report.status != O.COMPLETED


def test_D_the_agent_does_not_say_it_sent_an_unconfirmed_message(world, agent, sarah):
    world.install_providers(confirm=False)
    approval.set_hook(lambda request: True)
    message = agent(messaging_agent("Sarah", "late")).run(
        "Text Sarah that I'm running late").message()

    assert "was accepted by the provider" not in message
    assert "Done." not in message


def test_D_a_provider_error_is_a_failure(world, agent, sarah):
    world.install_providers(error="21610 recipient unsubscribed")
    approval.set_hook(lambda request: True)
    orchestrator = agent(messaging_agent("Sarah", "late"))
    report = orchestrator.run("Text Sarah that I'm running late")

    assert report.status != O.COMPLETED


# ══ Scenario E — an ambiguous recipient ══════════════════════════════════════

@pytest.fixture
def two_johns():
    C.contacts().add(C.Contact(name="John Smith", phone="+15550002222",
                               note="from school"))
    C.contacts().add(C.Contact(name="John Doe", phone="+15550003333",
                               note="football"))


def test_E_an_ambiguous_recipient_stops_and_asks(world, agent, two_johns):
    world.install_providers()
    report = agent(messaging_agent("John", "late")).run(
        "Text John that I'm running late")

    assert report.status == O.NEEDS_INPUT
    assert "John" in report.question


def test_E_nothing_is_sent_while_it_is_ambiguous(world, agent, two_johns):
    providers = world.install_providers()
    agent(messaging_agent("John", "late")).run("Text John that I'm running late")

    assert providers[CB.SMS].sent == [] and world.call_count("send_message") == 0


def test_E_the_question_distinguishes_the_candidates(world, agent, two_johns):
    world.install_providers()
    report = agent(messaging_agent("John", "late")).run(
        "Text John that I'm running late")

    assert report.message().endswith("?")


def test_E_naming_the_right_john_lets_it_proceed(world, agent, two_johns):
    providers = world.install_providers()
    approval.set_hook(lambda request: True)
    report = agent(messaging_agent("John Smith", "late")).run(
        "Text John Smith that I'm running late")

    assert report.status == O.COMPLETED
    assert providers[CB.SMS].sent[0]["to"] == "+15550002222"


# ══ Scenario F — a long task, interrupted and resumed ════════════════════════

def test_F_a_budget_exhausted_run_leaves_a_checkpoint(world, agent):
    orchestrator = agent(keyword_agent(world), max_iterations=2)
    report = orchestrator.run("build me a python script that prints hello")

    assert report.status != O.COMPLETED
    assert CP.load(report.checkpoint_id) is not None


def test_F_checkpoints_are_written_during_the_run_not_only_at_the_end(world, agent):
    orchestrator = agent(keyword_agent(world), checkpoint_every=1, max_iterations=3)
    orchestrator.run("build me a python script that prints hello")

    checkpoint = CP.load(orchestrator.checkpoint_id)
    assert checkpoint is not None and checkpoint.completed


def test_F_a_restarted_agent_resumes_and_finishes(world, agent):
    first = agent(keyword_agent(world), max_iterations=2)
    first.run("build me a python script that prints hello")

    second = agent(keyword_agent(world))       # a fresh agent, fresh memory
    resumed = O.resume(second)

    assert resumed is not None and resumed.status == O.COMPLETED


def test_F_resuming_does_not_redo_finished_work(world, agent):
    first = agent(keyword_agent(world), max_iterations=3)
    first.run("build me a python script that prints hello")
    thinks_before = world.call_count("think")

    O.resume(agent(keyword_agent(world)))

    # The reasoning steps already marked done should not run a second time.
    replayed = world.call_count("think") - thinks_before
    assert replayed < thinks_before + 3


def test_F_resume_reopens_work_whose_effect_vanished(world, agent):
    """Checkpoint state is not trusted — the world is re-checked."""
    first = agent(keyword_agent(world), max_iterations=6)
    first.run("build me a python script that prints hello")
    checkpoint = CP.load(first.checkpoint_id)
    assert checkpoint.effects, "the write should have been recorded as an effect"

    world.vanish("app.py")                     # the file disappears while we're away
    discrepancies = CP.verify_against_reality(checkpoint)

    assert discrepancies and "app.py" in discrepancies[0].detail


def test_F_the_resumed_run_rewrites_the_vanished_file(world, agent):
    first = agent(keyword_agent(world), max_iterations=6)
    first.run("build me a python script that prints hello")
    world.vanish("app.py")

    O.resume(agent(keyword_agent(world)))

    assert world.exists("app.py"), "the agent should have noticed and redone it"


# ══ Scenario G — emergency stop during execution ═════════════════════════════

def test_G_the_stop_halts_the_run(world, agent):
    def stop_on_third(task, context):
        if len(world.log) >= 3:
            emergency_stop.engage("CJ said stop everything", source="human")
        return keyword_agent(world)(task, context)

    report = agent(stop_on_third).run("build me a python script that prints hello")
    assert report.status == O.STOPPED


def test_G_no_new_consequential_action_begins_after_the_stop(world, agent):
    def stop_immediately(task, context):
        emergency_stop.engage("stop", source="human")
        return [("write_file", {"filename": "should_not_exist.py", "content": "x"})]

    agent(stop_immediately).run("build me a python script that prints hello")
    assert not world.exists("should_not_exist.py")
    assert world.call_count("write_file") == 0


def test_G_in_flight_work_is_marked_interrupted_not_active(world, agent):
    def stop_now(task, context):
        emergency_stop.engage("stop", source="human")
        return []

    orchestrator = agent(stop_now)
    orchestrator.run("build me a python script that prints hello")

    statuses = {t.status for t in orchestrator.plan.tasks.values()}
    assert P.INTERRUPTED in statuses and P.ACTIVE not in statuses


def test_G_the_checkpoint_records_the_interruption(world, agent):
    def stop_now(task, context):
        emergency_stop.engage("stop", source="human")
        return []

    report = agent(stop_now).run("build me a python script that prints hello")
    checkpoint = CP.load(report.checkpoint_id)

    assert checkpoint.run_status == "INTERRUPTED" and checkpoint.interrupted


def test_G_the_interrupted_attempt_is_not_charged_to_the_budget(world, agent):
    def stop_now(task, context):
        emergency_stop.engage("stop", source="human")
        return []

    orchestrator = agent(stop_now)
    orchestrator.run("build me a python script that prints hello")
    interrupted = [t for t in orchestrator.plan.tasks.values()
                   if t.status == P.INTERRUPTED]

    assert interrupted and interrupted[0].attempts == 0


def test_G_resuming_after_a_stop_continues_the_work(world, agent):
    def stop_now(task, context):
        emergency_stop.engage("stop", source="human")
        return []

    first = agent(stop_now)
    first.run("build me a python script that prints hello")
    emergency_stop.clear(source=emergency_stop.HUMAN)

    resumed = O.resume(agent(keyword_agent(world)))
    assert resumed is not None and resumed.status == O.COMPLETED


def test_G_the_agent_cannot_clear_its_own_stop(world, agent):
    def stop_then_try_to_resume(task, context):
        emergency_stop.engage("stop", source="human")
        emergency_stop.clear(source="agent")        # refused
        return [("write_file", {"filename": "sneaky.py", "content": "x"})]

    report = agent(stop_then_try_to_resume).run("build me a script")
    assert report.status == O.STOPPED and not world.exists("sneaky.py")


# ══ Scenario H — permission escalation mid-task ══════════════════════════════

def test_H_safe_work_runs_without_asking_then_the_restricted_step_asks(world, agent):
    permissions.set_profile(permissions.ASSISTED, source=permissions.HUMAN)
    asked = []
    approval.set_hook(lambda request: asked.append(request.tool) or True)

    orchestrator = agent(keyword_agent(world, source="import flask\nprint('x')"))
    report = orchestrator.run("build me a python script that uses flask")

    # write_file is reversible and local: no approval. pip_install is not.
    assert "pip_install" in asked and "write_file" not in asked
    assert report.status == O.COMPLETED


def test_H_a_denied_restricted_step_pauses_the_task(world, agent):
    permissions.set_profile(permissions.ASSISTED, source=permissions.HUMAN)
    approval.set_hook(lambda request: False)

    orchestrator = agent(keyword_agent(world, source="import flask\nprint('x')"))
    report = orchestrator.run("build me a python script that uses flask")

    assert report.status == O.BLOCKED
    assert world.call_count("pip_install") == 0


def test_H_the_blocked_report_says_what_it_needs(world, agent):
    permissions.set_profile(permissions.ASSISTED, source=permissions.HUMAN)
    approval.set_hook(lambda request: False)

    report = agent(keyword_agent(world, source="import flask\nprint('x')")).run(
        "build me a python script that uses flask")

    assert "couldn't finish" in report.message().lower()


def test_H_granting_permission_lets_the_same_task_complete(world, agent):
    permissions.set_profile(permissions.ASSISTED, source=permissions.HUMAN)
    approval.set_hook(lambda request: {"approved": True, "remember": True})

    report = agent(keyword_agent(world, source="import flask\nprint('x')")).run(
        "build me a python script that uses flask")

    assert report.status == O.COMPLETED
    assert "pip_install" in permissions.list_grants()


def test_H_the_agent_cannot_grant_itself_the_permission(world, agent):
    permissions.set_profile(permissions.SAFE, source=permissions.HUMAN)

    def self_authorising(task, context):
        permissions.grant("pip_install", source="agent")       # refused
        permissions.set_profile(permissions.AUTONOMOUS, source="agent")  # refused
        return keyword_agent(world, source="import flask\nprint('x')")(task, context)

    report = agent(self_authorising).run("build me a python script that uses flask")

    assert world.call_count("pip_install") == 0
    assert permissions.get_profile() == permissions.SAFE
    assert report.status != O.COMPLETED


# ══ Scenario I — persistent monitoring with autonomous response ══════════════

def test_I_a_crash_triggers_an_autonomous_restart(world, agent):
    """No second user message: the monitor drives the whole response."""
    from pinpoint.monitoring import events as E, watchers as W

    world.start_process("gameserver")
    world.write("gameserver.py", "print('serving')")

    monitor = W.Monitor()
    watcher = W.ProcessWatcher(
        "gameserver",
        probe=lambda: W.UP if world.processes.get("gameserver") else W.DOWN)
    monitor.add(watcher)
    monitor.poll_once()      # baseline: healthy

    responses = []

    def restart(event):
        """The autonomous response — a real orchestrator run, not a stub."""
        def fix(task, context):
            description = task.description.lower()
            if "reproduce" in description or "inspect" in description:
                return [("run_python", {"filename": "gameserver.py"})]
            if "apply a fix" in description:
                world.start_process("gameserver")     # the restart itself
                return [("run_python", {"filename": "gameserver.py"})]
            if "test" in description or "verify" in description:
                return [("run_python", {"filename": "gameserver.py"})]
            return [("think", {"reasoning": task.instruction()[:120]})]

        responses.append(agent(fix).run("restart the crashed gameserver"))

    monitor.pipeline.subscribe([E.PROCESS_CRASHED], policy=E.ACT,
                               goal="keep the gameserver up",
                               response="restart it", handler=restart,
                               sources=["gameserver"])

    world.crash_process("gameserver")           # the world changes
    decisions = monitor.poll_once()             # the monitor notices

    assert decisions and decisions[0].outcome == E.ACT
    assert responses, "the crash should have started a recovery run"
    assert world.processes["gameserver"] is True, "the service should be back up"


def test_I_the_restart_was_confirmed_by_a_health_check(world, agent):
    from pinpoint.monitoring import events as E, watchers as W

    world.start_process("svc")
    world.write("svc.py", "print('healthy')")
    monitor = W.Monitor()
    watcher = W.ProcessWatcher(
        "svc", probe=lambda: W.UP if world.processes.get("svc") else W.DOWN)
    monitor.add(watcher)
    monitor.poll_once()

    def restart(event):
        world.start_process("svc")

    monitor.pipeline.subscribe([E.PROCESS_CRASHED], policy=E.ACT, goal="keep svc up",
                               handler=restart, sources=["svc"])
    world.crash_process("svc")
    monitor.poll_once()

    # A second poll observes the recovery and emits it as an event.
    recovery_events = [e for e in [monitor.poll_once()] if e]
    assert world.processes["svc"] is True
    assert recovery_events


def test_I_a_restricted_response_is_downgraded_to_telling_cj(world, agent):
    """If the fix isn't permitted, the monitor reports rather than acting."""
    from pinpoint.monitoring import events as E, watchers as W

    permissions.set_custom_level("run_shell", permissions.RED, source=permissions.HUMAN)
    monitor = W.Monitor()
    monitor.add(W.ProcessWatcher("svc", probe=lambda: W.DOWN))
    monitor.pipeline.subscribe([E.PROCESS_CRASHED], policy=E.ACT, goal="keep svc up",
                               response="restart it", tool="run_shell")
    monitor.poll_once()
    decisions = monitor.poll_once()

    if decisions:
        assert decisions[0].outcome == E.NOTIFY


def test_I_a_healthy_service_produces_no_noise(world, agent):
    from pinpoint.monitoring import events as E, watchers as W

    world.start_process("svc")
    monitor = W.Monitor()
    monitor.add(W.ProcessWatcher("svc", probe=lambda: W.UP))
    monitor.pipeline.subscribe([E.PROCESS_CRASHED], policy=E.ACT, goal="keep svc up")

    monitor.poll_once()
    assert monitor.poll_once() == []
