"""Stage 4: action results, independent verification, executor pipeline.

The theme: a tool that ran without complaining is not the same as a tool that
worked, and the difference has to survive all the way to the caller.
"""

import os

import pytest

from pinpoint import epistemics
from pinpoint.action import capabilities, executor as EX, result as R, verify as V
from pinpoint.security import approval, audit, emergency_stop, permissions
from pinpoint.tools import registry


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    emergency_stop.reset_for_tests()
    approval.reset_for_tests()
    permissions.set_profile(permissions.AUTONOMOUS, source=permissions.HUMAN)
    yield
    emergency_stop.reset_for_tests()
    approval.reset_for_tests()


def make(dispatch):
    return EX.Executor(dispatch=dispatch, session=1, goal="test goal")


# ── ActionResult semantics ────────────────────────────────────────────────────

def test_success_is_ok():
    assert R.ActionResult(tool="t", status=R.SUCCESS).ok is True


def test_unverified_is_not_ok():
    assert R.ActionResult(tool="t", status=R.UNVERIFIED).ok is False


def test_blocked_is_not_retryable():
    assert R.ActionResult(tool="t", status=R.BLOCKED).retryable is False


def test_failed_is_retryable():
    assert R.ActionResult(tool="t", status=R.FAILED).retryable is True


def test_unverified_observation_forbids_claiming_done():
    res = R.ActionResult(tool="send_message", status=R.UNVERIFIED,
                         verification={"detail": "no confirmation id"})
    assert "Do not report this as done" in res.as_observation()


def test_action_ids_are_unique():
    assert R.ActionResult(tool="a").action_id != R.ActionResult(tool="a").action_id


def test_result_serializes():
    d = R.ActionResult(tool="t", output="x").to_dict()
    assert d["tool"] == "t" and "timestamp" in d and "action_id" in d


# ── Verification ──────────────────────────────────────────────────────────────

def test_file_exists_verified_when_written(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("hello")
    out = V.verify("write_file", {"path": str(path)}, f"Written 5 chars to {path}")
    assert out.verified is True and out.epistemic == epistemics.OBSERVED


def test_file_exists_fails_when_missing(tmp_path):
    out = V.verify("write_file", {"path": str(tmp_path / "ghost.txt")},
                   "Written 5 chars to /nope/ghost.txt")
    assert out.verified is False


def test_empty_file_is_a_failure(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("")
    out = V.verify("write_file", {"path": str(path)}, "Written 0 chars")
    assert out.verified is False and "empty" in out.detail


def test_cheerful_output_does_not_beat_a_missing_file(tmp_path):
    out = V.verify("write_file", {"path": str(tmp_path / "nope.txt")},
                   "Successfully wrote the file! Everything worked great.")
    assert out.verified is False


def test_delete_verified_only_when_gone(tmp_path):
    path = tmp_path / "d.txt"
    path.write_text("x")
    assert V.verify("delete_file", {"path": str(path)}, "Deleted").verified is False
    path.unlink()
    assert V.verify("delete_file", {"path": str(path)}, "Deleted").verified is True


def test_exit_code_zero_verifies():
    assert V.verify("run_shell", {"command": "ls"}, "stdout:\nx\nexit code: 0").verified is True


def test_nonzero_exit_code_fails():
    out = V.verify("run_shell", {"command": "false"}, "exit code: 1")
    assert out.verified is False and "exit code 1" in out.detail


def test_missing_exit_code_is_unverified_not_success():
    out = V.verify("run_shell", {"command": "ls"}, "some output with no code")
    assert out.verified is None


def test_provider_confirmation_required_for_sends():
    out = V.verify("send_message", {"to": "a"}, "Message sent successfully!")
    assert out.verified is None
    assert "do not claim" in out.detail.lower()


def test_provider_confirmation_accepts_an_id():
    out = V.verify("send_message", {"to": "a"}, "queued. message_sid: SM123456789")
    assert out.verified is True


def test_process_running_fails_on_dead_port():
    out = V.verify("start_server", {"port": 65533}, "server started on :65533")
    assert out.verified is False


def test_state_change_compares_snapshots():
    assert V.verify("save_memory", {}, "", before={"a": 1}, after={"a": 2}).verified is True
    assert V.verify("save_memory", {}, "", before={"a": 1}, after={"a": 1}).verified is False


def test_reasoning_tools_verify_trivially():
    assert V.verify("think", {"reasoning": "x"}, "thought recorded").verified is True


def test_reasoning_tool_error_is_detected():
    assert V.verify("think", {}, "Error: traceback (most recent call last)").verified is False


# ── Executor: policy enforcement ──────────────────────────────────────────────

def test_executor_runs_and_verifies(tmp_path):
    target = tmp_path / "out.txt"

    def dispatch(tool, params):
        target.write_text(params["content"])
        return f"Written {len(params['content'])} chars to {target}"

    res = make(dispatch).execute("write_file", {"filename": str(target), "content": "hi"})
    assert res.status == R.SUCCESS and res.ok


def test_executor_marks_unverified_when_effect_absent(tmp_path):
    def dispatch(tool, params):
        return "Written 2 chars to /tmp/definitely-not-here-9182/x.txt"

    res = make(dispatch).execute("write_file", {"filename": "x.txt", "content": "hi"})
    assert res.status == R.FAILED


def test_executor_blocks_red_actions():
    called = []
    res = make(lambda t, p: called.append(t) or "ok").execute(
        "run_shell", {"command": "rm -rf /"})
    assert res.status == R.BLOCKED and called == []


def test_executor_honours_emergency_stop():
    called = []
    emergency_stop.engage("halt", source="human")
    res = make(lambda t, p: called.append(t) or "ok").execute("write_file",
                                                              {"filename": "a", "content": "b"})
    assert res.status == R.BLOCKED and called == []


def test_executor_denies_without_approval():
    permissions.set_profile(permissions.SAFE, source=permissions.HUMAN)
    called = []
    res = make(lambda t, p: called.append(t) or "ok").execute("delete_file", {"path": "x"})
    assert res.status == R.DENIED and called == []


def test_executor_proceeds_after_approval(tmp_path):
    permissions.set_profile(permissions.SAFE, source=permissions.HUMAN)
    approval.set_hook(lambda req: True)
    path = tmp_path / "gone.txt"
    res = make(lambda t, p: "Deleted").execute("delete_file", {"path": str(path)})
    assert res.status == R.SUCCESS and res.approval == "approved"


def test_executor_reports_missing_capability(monkeypatch):
    monkeypatch.setattr(capabilities, "probe",
                        lambda req: (False, "pyautogui is not installed"))
    called = []
    res = make(lambda t, p: called.append(t) or "typed").execute("type_text", {"text": "hi"})
    assert res.status == R.UNAVAILABLE and called == []
    assert "pyautogui" in res.error


def test_missing_capability_suggests_a_workaround(monkeypatch):
    monkeypatch.setattr(capabilities, "probe", lambda req: (False, "no graphical display"))
    res = make(lambda t, p: "ok").execute("run_gui", {"filename": "a.py"})
    assert "headless" in res.error


# ── Executor: failures ────────────────────────────────────────────────────────

def test_tool_exception_becomes_failed_result():
    def boom(tool, params):
        raise RuntimeError("disk on fire")

    res = make(boom).execute("write_file", {"filename": "a", "content": "b"})
    assert res.status == R.FAILED and "disk on fire" in res.error


def test_tool_timeout_is_unverified_not_failed():
    """Python can't kill the worker, so the effect may still land. Unknown ≠ failed."""
    import time

    def slow(tool, params):
        time.sleep(3)
        return "done"

    res = make(slow).execute("write_file", {"filename": "a", "content": "b"}, timeout=0.1)
    assert res.status == R.UNVERIFIED and "timed out" in res.error
    assert "may still be running" in res.error


def test_timeout_that_still_landed_is_a_success(tmp_path):
    """If the effect is confirmed despite the timeout, say so."""
    import threading
    import time

    target = tmp_path / "slow.txt"

    def slow(tool, params):
        time.sleep(0.05)
        target.write_text("landed")
        return f"Written 6 chars to {target}"

    res = make(slow).execute("write_file", {"filename": str(target), "content": "x"},
                             timeout=0.3)
    assert res.status == R.SUCCESS


def test_timeout_warns_against_a_blind_retry():
    import time

    res = make(lambda t, p: time.sleep(2)).execute(
        "run_shell", {"command": "sleep 10"}, timeout=0.1)
    assert "Do not retry blindly" in res.error


def test_failure_never_reports_success():
    res = make(lambda t, p: "Traceback (most recent call last): boom").execute(
        "run_python", {"filename": "a.py"})
    assert res.ok is False


# ── Executor: bookkeeping ─────────────────────────────────────────────────────

def test_actions_are_audited(tmp_path):
    make(lambda t, p: "Written 1 chars to " + str(tmp_path / "z")).execute(
        "write_file", {"filename": "z", "content": "x"})
    assert any(e["tool"] == "write_file" for e in audit.query(action="action"))


def test_audit_captures_the_block_reason():
    make(lambda t, p: "ok").execute("run_shell", {"command": "rm -rf /"})
    entry = [e for e in audit.query(action="action") if e["tool"] == "run_shell"][-1]
    assert entry["error"]


def test_side_effects_are_recorded():
    res = make(lambda t, p: "x").execute("delete_file", {"path": "a"})
    assert "irreversible" in res.side_effects


def test_executor_stats_separate_verified_from_unverified(tmp_path):
    ex = make(lambda t, p: "no confirmation here")
    ex.execute("send_message", {"to": "a", "body": "b"})
    permissions.grant("send_message", source=permissions.HUMAN)
    ex.execute("send_message", {"to": "a", "body": "b"})
    stats = ex.stats()
    assert stats["verified_successes"] == 0


def test_on_result_callback_fires(tmp_path):
    seen = []
    ex = EX.Executor(dispatch=lambda t, p: "ok", on_result=seen.append)
    ex.execute("think", {"reasoning": "x"})
    assert len(seen) == 1


def test_callback_errors_do_not_break_execution():
    def bad(res):
        raise ValueError("bad callback")

    ex = EX.Executor(dispatch=lambda t, p: "ok", on_result=bad)
    assert ex.execute("think", {"reasoning": "x"}).status == R.SUCCESS


def test_last_returns_most_recent_matching_action():
    ex = make(lambda t, p: "ok")
    ex.execute("think", {"reasoning": "a"})
    ex.execute("brainstorm", {"topic": "b"})
    assert ex.last("think").params["reasoning"] == "a"
    assert ex.last().tool == "brainstorm"


# ── Capability reporting ──────────────────────────────────────────────────────

def test_capability_report_covers_key_requirements():
    report = capabilities.report()
    assert "display" in report and "sms_provider" in report


def test_unknown_capability_does_not_block():
    ok, _ = capabilities.check_all(["vision_model"])
    assert ok is True
