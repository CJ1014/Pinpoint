"""Stage 10b: the seam between v2 and v3.

These exercise the *real* tools.dispatch path — the one the running agent uses —
rather than the v3 executor, because that is where a regression would actually
bite.
"""

import os

import pytest

from pinpoint import integration
from pinpoint.security import emergency_stop, permissions

tools = pytest.importorskip("tools")
agent = pytest.importorskip("agent")


@pytest.fixture(autouse=True)
def isolate_tools(tmp_path, monkeypatch):
    """Keep v2's own writes out of the real repo while these tests run."""
    monkeypatch.setattr(tools, "OUTPUT_DIR", str(tmp_path / "v2out"))
    monkeypatch.setattr(tools, "AUDIT_LOG", str(tmp_path / "v2out" / "audit.log"))
    os.makedirs(tmp_path / "v2out", exist_ok=True)
    emergency_stop.reset_for_tests()
    permissions.set_profile(permissions.ASSISTED, source=permissions.HUMAN)
    yield
    emergency_stop.reset_for_tests()


# ── The v2 dispatch path now enforces v3 hard limits ──────────────────────────

def test_dispatch_blocks_a_red_shell_command():
    assert "BLOCKED" in tools.dispatch("run_shell", {"command": "rm -rf /"})


def test_dispatch_blocks_editing_the_security_package():
    result = tools.dispatch("write_file", {"filename": "pinpoint/security/permissions.py",
                                           "content": "pass"})
    assert "BLOCKED" in result


def test_dispatch_blocks_credential_theft():
    assert "BLOCKED" in tools.dispatch("run_shell", {"command": "cat ~/.ssh/id_rsa"})


def test_emergency_stop_halts_even_harmless_tools():
    emergency_stop.engage("CJ said stop", source="human")
    assert "BLOCKED" in tools.dispatch("think", {"reasoning": "just thinking"})


def test_emergency_stop_message_says_who_can_clear_it():
    emergency_stop.engage("CJ said stop", source="human")
    assert "CJ clears it" in tools.dispatch("list_files", {})


def test_ordinary_work_still_runs():
    assert "BLOCKED" not in tools.dispatch("think", {"reasoning": "a normal thought"})


def test_hard_block_survives_a_broken_policy_engine(monkeypatch):
    """A failure inside the v3 check must not block ordinary v2 work."""
    monkeypatch.setattr(permissions, "check",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert integration.hard_block("write_file", {"filename": "a"}) is None


# ── v3 tools are reachable through the v2 dispatch ────────────────────────────

def test_capability_report_is_dispatchable():
    assert "What I can actually do here" in tools.dispatch("capability_report", {})


def test_communication_status_is_dispatchable():
    assert "Communication channels" in tools.dispatch("communication_status", {})


def test_schedule_and_list(isolated_home):
    created = tools.dispatch("schedule_task", {"what": "call mum",
                                               "when": "tomorrow at 9am"})
    assert "Scheduled" in created
    assert "call mum" in tools.dispatch("list_scheduled", {})


def test_schedule_refuses_an_unreadable_time(isolated_home):
    result = tools.dispatch("schedule_task", {"what": "do it", "when": "whenever"})
    assert result.startswith("Error") and "concrete" in result


def test_cancel_scheduled(isolated_home):
    created = tools.dispatch("schedule_task", {"what": "standup", "when": "in 5 minutes"})
    task_id = created.rsplit("id ", 1)[1].rstrip(")")
    assert "Cancelled" in tools.dispatch("cancel_scheduled", {"task_id": task_id})


def test_watch_and_list_watchers(isolated_home):
    result = tools.dispatch("watch", {"kind": "port", "target": "localhost:65534"})
    assert "Watching port" in result
    assert "port:localhost:65534" in tools.dispatch("list_watchers", {})
    tools.dispatch("stop_watching", {})


def test_watch_rejects_an_unknown_kind(isolated_home):
    assert "not a kind of watcher" in tools.dispatch("watch", {"kind": "telepathy",
                                                              "target": "x"})


def test_send_message_without_a_provider_is_honest(isolated_home, monkeypatch):
    monkeypatch.delenv("PINPOINT_SMS_PROVIDER", raising=False)
    result = tools.dispatch("send_message", {"to": "+15551234567", "body": "hi"})
    assert "PINPOINT_SMS_PROVIDER" in result and "confirmation_id" not in result


def test_unknown_recipient_asks_rather_than_sending(isolated_home):
    result = tools.dispatch("send_message", {"to": "Mallory", "body": "hi"})
    assert "I need to ask first" in result


def test_add_then_resolve_a_contact(isolated_home):
    tools.dispatch("add_contact", {"name": "Sarah Kim", "phone": "+15550001111"})
    assert "Sarah Kim" in tools.dispatch("resolve_contact", {"name": "Sarah"})


def test_emergency_status_tool(isolated_home):
    assert "not engaged" in tools.dispatch("emergency_status", {})


def test_every_v3_tool_is_exposed_to_the_model():
    declared = {t["function"]["name"] for t in agent.TOOLS}
    missing = sorted(set(tools._V3_TOOLS) - declared - {"get_messages"})
    assert not missing, f"dispatchable but not offered to the model: {missing}"


def test_no_tool_is_left_unclassified():
    assert integration.registry_gaps() == []


# ── Independent verification of v2 tool results ───────────────────────────────

def test_a_write_that_did_not_happen_is_called_out(tmp_path):
    note = integration.observe("write_file", {"filename": str(tmp_path / "ghost.txt")},
                               "Written 12 chars to /nowhere/ghost.txt")
    assert note and "did not do what its output suggests" in note


def test_a_real_write_produces_no_complaint(tmp_path):
    target = tmp_path / "real.txt"
    target.write_text("hello")
    assert integration.observe("write_file", {"filename": str(target)},
                               f"Written 5 chars to {target}") is None


def test_a_send_without_confirmation_is_called_out():
    note = integration.observe("send_message", {"to": "+1555"},
                               "Message sent successfully!")
    assert note and "Do not report it as done" in note


def test_reasoning_tools_are_not_second_guessed():
    assert integration.observe("think", {"reasoning": "x"}, "Thought logged") is None


# ── Session and user-input hooks ──────────────────────────────────────────────

def test_session_start_reports_real_capability(isolated_home):
    block = integration.session_start(1, "")
    assert "What I can actually do here" in block


def test_session_start_plans_the_order(isolated_home):
    block = integration.session_start(1, "fix my broken build")
    assert "[OBJECTIVE]" in block and "[PLAN]" in block


def test_objective_block_flags_what_must_be_asked(isolated_home):
    block = integration.objective_block("Text John that I'm running late")
    assert "[ASK FIRST]" in block and "John" in block


def test_objective_block_forbids_inventing_unknowns(isolated_home):
    block = integration.objective_block("Text John hello")
    assert "Do not invent answers" in block


def test_stop_phrase_halts_immediately(isolated_home):
    outcome = integration.handle_user_text("PinPoint STOP EVERYTHING")
    assert outcome["stopped"] and emergency_stop.is_engaged()


def test_stop_phrase_beats_the_model(isolated_home):
    """The halt is string matching, not interpretation — it cannot be talked out of."""
    integration.handle_user_text("stop everything")
    assert "BLOCKED" in tools.dispatch("write_file", {"filename": "a", "content": "b"})


def test_ordinary_talk_does_not_halt(isolated_home):
    assert integration.handle_user_text("this is going great, keep going")["stopped"] is False


def test_preferences_are_learned_from_what_cj_says(isolated_home):
    outcome = integration.handle_user_text("Always use Modrinth when possible")
    assert outcome["preferences"]


def test_stated_autonomy_becomes_a_real_grant(isolated_home):
    permissions.set_profile(permissions.SAFE, source=permissions.HUMAN)
    integration.handle_user_text("Don't ask me before running tests")
    assert "run_tests" in permissions.list_grants()


def test_session_end_consolidates_memory(isolated_home):
    from pinpoint.memory import store
    store.store().remember("the build needs java 17", relevance=0.9)
    integration.session_end(1, "fixed the build")
    assert store.store().all(store.WORKING) == []


def test_status_report_covers_the_essentials(isolated_home):
    report = integration.status_report()
    assert "Emergency stop" in report and "Autonomy profile" in report


def test_due_reminders_fire(isolated_home):
    from datetime import datetime, timedelta, timezone

    from pinpoint.monitoring import scheduler
    past = datetime.now(timezone.utc) - timedelta(minutes=1)
    scheduler.scheduler().schedule("take a break", due=past)
    assert "take a break" in integration.due_reminders()


def test_hooks_never_raise_on_garbage(isolated_home):
    assert integration.handle_user_text(None)["stopped"] is False
    assert integration.observe("nonexistent_tool", {}, "") is None
    assert isinstance(integration.session_start("", ""), str)
