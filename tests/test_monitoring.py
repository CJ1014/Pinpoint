"""Stage 9: event pipeline, watchers, scheduler."""

from datetime import datetime, timedelta, timezone

import pytest

from pinpoint.monitoring import events as E, scheduler as SCH, watchers as W
from pinpoint.security import permissions


def crash(source="minecraft"):
    return E.Event(type=E.PROCESS_CRASHED, source=source,
                   detail="process gone", severity=E.CRITICAL)


@pytest.fixture
def pipeline():
    return E.EventPipeline(dedupe_seconds=30.0)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def test_unsubscribed_events_are_ignored(pipeline):
    assert pipeline.decide(crash()).outcome == E.IGNORE


def test_subscribed_event_notifies(pipeline):
    pipeline.subscribe([E.PROCESS_CRASHED], response="tell CJ")
    assert pipeline.decide(crash()).outcome == E.NOTIFY


def test_act_policy_acts(pipeline):
    pipeline.subscribe([E.PROCESS_CRASHED], policy=E.ACT, goal="keep the server up",
                       response="restart it", tool="run_shell")
    permissions.set_profile(permissions.AUTONOMOUS, source=permissions.HUMAN)
    assert pipeline.decide(crash()).outcome == E.ACT


def test_duplicate_events_are_suppressed(pipeline):
    pipeline.subscribe([E.PROCESS_CRASHED], response="tell CJ")
    first = pipeline.decide(crash())
    second = pipeline.decide(crash())
    assert first.outcome == E.NOTIFY and second.outcome == E.IGNORE


def test_different_sources_are_not_duplicates(pipeline):
    pipeline.subscribe([E.PROCESS_CRASHED], response="tell CJ")
    pipeline.decide(crash("minecraft"))
    assert pipeline.decide(crash("webserver")).outcome == E.NOTIFY


def test_severity_floor_filters_noise():
    pipeline = E.EventPipeline(min_severity=E.WARNING)
    pipeline.subscribe([E.FILE_CHANGED], response="look")
    quiet = E.Event(type=E.FILE_CHANGED, source="a.txt", severity=E.INFO)
    assert pipeline.decide(quiet).outcome == E.IGNORE


def test_subscription_severity_floor(pipeline):
    pipeline.subscribe([E.FILE_CHANGED], response="look", min_severity=E.CRITICAL)
    warning = E.Event(type=E.FILE_CHANGED, source="a.txt", severity=E.WARNING)
    assert pipeline.decide(warning).outcome == E.IGNORE


def test_source_filter(pipeline):
    pipeline.subscribe([E.PROCESS_CRASHED], sources=["minecraft"], response="restart")
    assert pipeline.decide(crash("something_else")).outcome == E.IGNORE


def test_blocked_response_downgrades_to_notify(pipeline):
    pipeline.subscribe([E.PROCESS_CRASHED], policy=E.ACT, goal="keep it up",
                       response="wipe the disk", tool="run_shell")
    permissions.set_custom_level("run_shell", permissions.RED, source=permissions.HUMAN)
    decision = pipeline.decide(crash())
    assert decision.outcome == E.NOTIFY and "blocked by policy" in decision.reason


def test_act_flags_when_approval_would_be_needed(pipeline):
    permissions.set_profile(permissions.SAFE, source=permissions.HUMAN)
    pipeline.subscribe([E.PROCESS_CRASHED], policy=E.ACT, goal="keep it up",
                       response="restart", tool="run_shell")
    assert pipeline.decide(crash()).requires_approval is True


def test_goal_linked_subscription_wins(pipeline):
    pipeline.subscribe([E.PROCESS_CRASHED], response="just a note")
    pipeline.subscribe([E.PROCESS_CRASHED], goal="keep the server up",
                       response="restart it")
    assert pipeline.decide(crash()).subscription.goal == "keep the server up"


def test_handler_runs_on_act(pipeline):
    seen = []
    permissions.set_profile(permissions.AUTONOMOUS, source=permissions.HUMAN)
    pipeline.subscribe([E.PROCESS_CRASHED], policy=E.ACT, goal="g",
                       handler=seen.append)
    pipeline.handle(crash())
    assert len(seen) == 1


def test_handler_does_not_run_on_notify(pipeline):
    seen = []
    pipeline.subscribe([E.PROCESS_CRASHED], policy=E.NOTIFY, handler=seen.append)
    pipeline.handle(crash())
    assert seen == []


def test_handler_failure_is_contained(pipeline):
    def boom(event):
        raise RuntimeError("handler exploded")

    permissions.set_profile(permissions.AUTONOMOUS, source=permissions.HUMAN)
    pipeline.subscribe([E.PROCESS_CRASHED], policy=E.ACT, goal="g", handler=boom)
    assert "handler failed" in pipeline.handle(crash()).reason


def test_unsubscribe(pipeline):
    subscription = pipeline.subscribe([E.PROCESS_CRASHED], response="x")
    assert pipeline.unsubscribe(subscription.id)
    assert pipeline.decide(crash()).outcome == E.IGNORE


def test_pipeline_makes_no_model_calls(pipeline, monkeypatch):
    # Any attempt to reach an LLM would have to import openai; assert it isn't used.
    import sys
    before = sys.modules.get("openai")
    pipeline.subscribe([E.PROCESS_CRASHED], response="x")
    pipeline.decide(crash())
    assert sys.modules.get("openai") is before


# ── Watchers ──────────────────────────────────────────────────────────────────

def test_port_watcher_emits_on_going_down():
    state = {"value": W.UP}
    watcher = W.PortWatcher("localhost", 25565, probe=lambda: state["value"])
    watcher.poll()
    state["value"] = W.DOWN
    events = watcher.poll()
    assert events and events[0].type == E.PORT_DOWN


def test_watcher_does_not_repeat_while_state_holds():
    watcher = W.PortWatcher("localhost", 25565, probe=lambda: W.DOWN)
    watcher.poll()
    assert watcher.poll() == []


def test_port_watcher_emits_recovery():
    state = {"value": W.DOWN}
    watcher = W.PortWatcher("localhost", 25565, probe=lambda: state["value"])
    watcher.poll()
    state["value"] = W.UP
    assert watcher.poll()[0].type == E.PORT_UP


def test_process_crash_is_critical():
    state = {"value": W.UP}
    watcher = W.ProcessWatcher("minecraft", probe=lambda: state["value"])
    watcher.poll()
    state["value"] = W.DOWN
    assert watcher.poll()[0].severity == E.CRITICAL


def test_file_watcher_first_poll_is_a_baseline(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("one")
    watcher = W.FileWatcher(str(target))
    assert watcher.poll() == []


def test_file_watcher_notices_a_change(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("one")
    watcher = W.FileWatcher(str(target))
    watcher.poll()
    target.write_text("two different content")
    assert watcher.poll()[0].type == E.FILE_CHANGED


def test_command_watcher_reports_failure_then_recovery():
    state = {"value": W.UP}
    watcher = W.CommandWatcher("pytest", probe=lambda: state["value"])
    watcher.poll()
    state["value"] = W.DOWN
    assert watcher.poll()[0].type == E.COMMAND_FAILED
    state["value"] = W.UP
    assert watcher.poll()[0].type == E.COMMAND_RECOVERED


def test_probe_exception_becomes_a_down_event():
    def broken():
        raise OSError("network gone")

    watcher = W.HttpWatcher("https://example.com", probe=broken)
    events = watcher.poll()
    assert events and events[0].type == E.HTTP_DOWN


def test_inactive_watcher_does_not_poll():
    watcher = W.PortWatcher("localhost", 1, probe=lambda: W.DOWN)
    watcher.active = False
    assert watcher.poll() == [] and watcher.polls == 0


# ── Monitor ───────────────────────────────────────────────────────────────────

def test_monitor_routes_watcher_events_through_the_pipeline():
    monitor = W.Monitor()
    monitor.pipeline.subscribe([E.PORT_DOWN], response="restart the server")
    state = {"value": W.UP}
    monitor.add(W.PortWatcher("localhost", 25565, probe=lambda: state["value"]))
    monitor.poll_once()
    state["value"] = W.DOWN
    decisions = monitor.poll_once()
    assert decisions and decisions[0].outcome == E.NOTIFY


def test_monitor_start_and_stop():
    monitor = W.Monitor()
    assert monitor.start(interval=0.05) is True
    assert monitor.running is True
    monitor.stop()
    assert monitor.running is False


def test_monitor_status_lists_watchers():
    monitor = W.Monitor()
    monitor.add(W.PortWatcher("localhost", 25565, probe=lambda: W.UP))
    assert "port:localhost:25565" in " ".join(monitor.status()["watchers"])


# ── Scheduler ─────────────────────────────────────────────────────────────────

NOW = datetime(2026, 8, 5, 14, 0, tzinfo=timezone.utc)


def test_parse_tomorrow_at_nine():
    due, recurrence, _ = SCH.parse_when("remind me tomorrow at 9 am", NOW)
    assert due.day == 6 and due.hour == 9 and recurrence == SCH.ONCE


def test_parse_in_ten_minutes():
    due, _, _ = SCH.parse_when("check back in 10 minutes", NOW)
    assert due == NOW + timedelta(minutes=10)


def test_parse_pm_converts_to_24h():
    due, _, _ = SCH.parse_when("at 5 pm", NOW)
    assert due.hour == 17


def test_time_already_passed_rolls_to_tomorrow():
    due, _, _ = SCH.parse_when("at 9 am", NOW)
    assert due.day == 6


def test_parse_every_day():
    due, recurrence, interval = SCH.parse_when("every day at 8am", NOW)
    assert recurrence == SCH.DAILY and interval == 86400 and due.hour == 8


def test_parse_every_n_minutes():
    _, recurrence, interval = SCH.parse_when("every 15 minutes", NOW)
    assert recurrence == SCH.INTERVAL and interval == 900


def test_parse_iso_timestamp():
    due, _, _ = SCH.parse_when("at 2026-09-01T10:30", NOW)
    assert due.month == 9 and due.hour == 10


def test_unparseable_time_returns_nothing():
    assert SCH.parse_when("sometime when you feel like it", NOW) is None


def test_scheduling_an_unreadable_time_fails_rather_than_guessing():
    assert SCH.Scheduler().schedule("do a thing", "whenever", now=NOW) is None


def test_scheduled_task_persists():
    SCH.Scheduler().schedule("call mum", "tomorrow at 9 am", now=NOW)
    assert SCH.Scheduler().upcoming()


def test_due_task_fires():
    scheduler = SCH.Scheduler()
    scheduler.schedule("stand-up", "in 1 minute", now=NOW)
    assert scheduler.fire_due(NOW + timedelta(minutes=2))[0].type == E.SCHEDULED


def test_task_not_yet_due_does_not_fire():
    scheduler = SCH.Scheduler()
    scheduler.schedule("stand-up", "in 10 minutes", now=NOW)
    assert scheduler.fire_due(NOW) == []


def test_one_shot_task_deactivates_after_firing():
    scheduler = SCH.Scheduler()
    task = scheduler.schedule("stand-up", "in 1 minute", now=NOW)
    scheduler.fire_due(NOW + timedelta(minutes=2))
    assert scheduler.fire_due(NOW + timedelta(minutes=3)) == []
    assert task.active is False


def test_recurring_task_rolls_forward():
    scheduler = SCH.Scheduler()
    task = scheduler.schedule("daily report", "every day at 8am", now=NOW)
    first_due = task.due_at()
    scheduler.fire_due(first_due + timedelta(minutes=1))
    assert task.due_at() == first_due + timedelta(days=1) and task.active


def test_recurring_task_skips_missed_windows():
    scheduler = SCH.Scheduler()
    task = scheduler.schedule("hourly ping", "every 1 hour", now=NOW)
    scheduler.fire_due(NOW + timedelta(hours=5))
    assert task.due_at() > NOW + timedelta(hours=5)


def test_cancel_stops_a_task():
    scheduler = SCH.Scheduler()
    task = scheduler.schedule("stand-up", "in 1 minute", now=NOW)
    scheduler.cancel(task.id)
    assert scheduler.fire_due(NOW + timedelta(minutes=5)) == []


def test_describe_lists_upcoming():
    scheduler = SCH.Scheduler()
    scheduler.schedule("call mum", "tomorrow at 9 am", now=NOW)
    assert "call mum" in scheduler.describe()


def test_empty_schedule_describes_itself():
    assert "Nothing scheduled" in SCH.Scheduler().describe()
