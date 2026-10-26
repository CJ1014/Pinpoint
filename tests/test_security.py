"""Stage 2: permissions, approval, audit, emergency stop.

The load-bearing assertions here are the ones about what the agent *cannot* do:
escalate its own profile, grant itself permission, clear the emergency stop, or
edit the security package.
"""

import pytest

from pinpoint.security import approval, audit, emergency_stop, permissions


@pytest.fixture(autouse=True)
def clean_security():
    emergency_stop.reset_for_tests()
    approval.reset_for_tests()
    yield
    emergency_stop.reset_for_tests()
    approval.reset_for_tests()


# ── Classification ────────────────────────────────────────────────────────────

def test_read_only_tools_are_green():
    assert permissions.check("read_file", {"filename": "a.txt"}).level == permissions.GREEN


def test_green_actions_never_need_approval():
    d = permissions.check("list_files", {})
    assert d.allowed and not d.requires_approval


def test_shell_is_yellow_not_green():
    assert permissions.check("run_shell", {"command": "ls"}).level == permissions.YELLOW


def test_unknown_tool_defaults_to_yellow():
    assert permissions.check("some_new_tool", {}).level == permissions.YELLOW


# ── RED: hard blocks ──────────────────────────────────────────────────────────

def test_editing_security_package_is_red():
    d = permissions.check("write_file", {"filename": "pinpoint/security/permissions.py",
                                         "content": "x"})
    assert d.level == permissions.RED and d.blocked


def test_deleting_emergency_stop_is_red():
    d = permissions.check("delete_file", {"path": "output/EMERGENCY_STOP.json"})
    assert d.blocked


def test_recursive_root_delete_is_red():
    assert permissions.check("run_shell", {"command": "rm -rf /"}).blocked


def test_fork_bomb_is_red():
    assert permissions.check("run_shell", {"command": ":(){ :|:& };:"}).blocked


def test_reading_ssh_key_is_red():
    assert permissions.check("run_shell", {"command": "cat ~/.ssh/id_rsa"}).blocked


def test_disabling_approval_policy_is_red():
    assert permissions.check("run_shell",
                             {"command": "export PINPOINT_APPROVAL=allow"}).blocked


def test_auth_bypass_intent_is_red_in_any_tool():
    assert permissions.check("write_file",
                             {"filename": "x.py",
                              "content": "code to bypass authentication on the portal"}).blocked


def test_red_is_not_unlocked_by_a_grant():
    permissions.grant("run_shell", source=permissions.HUMAN, scope="always")
    assert permissions.check("run_shell", {"command": "rm -rf /"}).blocked


def test_red_is_not_unlocked_by_autonomous_profile():
    permissions.set_profile(permissions.AUTONOMOUS, source=permissions.HUMAN)
    assert permissions.check("run_shell", {"command": "mkfs.ext4 /dev/sda1"}).blocked


def test_ordinary_shell_is_not_red():
    assert not permissions.check("run_shell", {"command": "rm -rf ./build"}).blocked


# ── Profiles ──────────────────────────────────────────────────────────────────

def test_safe_profile_asks_about_writes():
    permissions.set_profile(permissions.SAFE, source=permissions.HUMAN)
    assert permissions.check("write_file", {"filename": "a", "content": "b"}).requires_approval


def test_assisted_allows_reversible_local_writes():
    permissions.set_profile(permissions.ASSISTED, source=permissions.HUMAN)
    d = permissions.check("write_file", {"filename": "a", "content": "b"})
    assert d.allowed and not d.requires_approval


def test_assisted_still_asks_before_irreversible_delete():
    permissions.set_profile(permissions.ASSISTED, source=permissions.HUMAN)
    assert permissions.check("delete_file", {"path": "notes.txt"}).requires_approval


def test_autonomous_allows_irreversible_local_work():
    permissions.set_profile(permissions.AUTONOMOUS, source=permissions.HUMAN)
    assert permissions.check("delete_file", {"path": "notes.txt"}).allowed


def test_autonomous_still_asks_before_external_messages():
    permissions.set_profile(permissions.AUTONOMOUS, source=permissions.HUMAN)
    assert permissions.check("send_message", {"to": "John", "body": "hi"}).requires_approval


def test_autonomous_still_asks_before_spending_money():
    permissions.set_profile(permissions.AUTONOMOUS, source=permissions.HUMAN)
    assert permissions.check("make_call", {"to": "+1555"}).requires_approval


def test_custom_profile_level_override():
    permissions.set_profile(permissions.CUSTOM, source=permissions.HUMAN)
    permissions.set_custom_level("run_tests", permissions.YELLOW, source=permissions.HUMAN)
    assert permissions.check("run_tests", {}).requires_approval


# ── The agent cannot escalate itself ──────────────────────────────────────────

def test_agent_cannot_change_profile():
    permissions.set_profile(permissions.SAFE, source=permissions.HUMAN)
    assert permissions.set_profile(permissions.AUTONOMOUS, source="agent") is False
    assert permissions.get_profile() == permissions.SAFE


def test_agent_cannot_grant_itself_permission():
    permissions.set_profile(permissions.SAFE, source=permissions.HUMAN)
    assert permissions.grant("run_shell", source="agent") is False
    assert permissions.check("run_shell", {"command": "ls"}).requires_approval


def test_agent_cannot_set_custom_levels():
    assert permissions.set_custom_level("run_shell", permissions.GREEN, source="model") is False


def test_human_grant_suppresses_repeat_asks():
    permissions.set_profile(permissions.SAFE, source=permissions.HUMAN)
    assert permissions.check("run_tests", {}).level == permissions.GREEN
    permissions.grant("write_file", source=permissions.HUMAN)
    d = permissions.check("write_file", {"filename": "a", "content": "b"})
    assert d.allowed and not d.requires_approval


def test_session_grants_cleared_but_always_grants_persist():
    permissions.grant("write_file", source=permissions.HUMAN, scope="session")
    permissions.grant("run_python", source=permissions.HUMAN, scope="always")
    permissions.clear_session_grants()
    grants = permissions.list_grants()
    assert "write_file" not in grants and "run_python" in grants


def test_revoke_removes_grant():
    permissions.grant("write_file", source=permissions.HUMAN)
    assert permissions.revoke("write_file") is True
    assert "write_file" not in permissions.list_grants()


# ── Approval workflow ─────────────────────────────────────────────────────────

def test_no_human_means_denied():
    permissions.set_profile(permissions.SAFE, source=permissions.HUMAN)
    assert approval.request("delete_file", {"path": "x"}).approved is False


def test_hook_approval_is_honored():
    approval.set_hook(lambda req: True)
    assert approval.request("delete_file", {"path": "x"}).approved is True


def test_hook_denial_is_honored():
    approval.set_hook(lambda req: False)
    assert approval.request("delete_file", {"path": "x"}).approved is False


def test_hook_exception_denies():
    def boom(req):
        raise RuntimeError("ui crashed")
    approval.set_hook(boom)
    assert approval.request("delete_file", {"path": "x"}).approved is False


def test_hook_timeout_denies():
    import time
    approval.set_hook(lambda req: time.sleep(5) or True)
    outcome = approval.request("delete_file", {"path": "x"}, timeout=0.1)
    assert outcome.approved is False and outcome.source == "timeout"


def test_remember_creates_standing_grant():
    permissions.set_profile(permissions.SAFE, source=permissions.HUMAN)
    approval.set_hook(lambda req: {"approved": True, "remember": True})
    approval.request("write_file", {"filename": "a", "content": "b"})
    assert "write_file" in permissions.list_grants()
    assert not permissions.check("write_file", {"filename": "a",
                                                "content": "b"}).requires_approval


def test_legacy_hook_signature_still_works():
    seen = {}

    def legacy(desc, risk, preview, two_step):
        seen["desc"] = desc
        return True

    approval.set_legacy_hook(legacy)
    assert approval.request("delete_file", {"path": "x"}).approved is True
    assert "delete_file" in seen["desc"]


def test_request_preview_is_redacted():
    req = approval.build_request("run_shell", {"command": "curl -H 'Bearer sk-abcdefghijklmnop123'"})
    assert "sk-abcdefghijklmnop123" not in req.preview


def test_request_renders_human_readable_block():
    text = approval.build_request("delete_file", {"path": "notes.txt"}).render()
    assert "APPROVAL" in text and "delete_file" in text and "[Approve]" in text


def test_approval_stats_track_decisions():
    approval.set_hook(lambda req: True)
    approval.request("delete_file", {"path": "a"})
    approval.set_hook(lambda req: False)
    approval.request("delete_file", {"path": "b"})
    s = approval.stats()
    assert s["approved"] == 1 and s["denied"] == 1


# ── Emergency stop ────────────────────────────────────────────────────────────

def test_stop_starts_disengaged():
    assert emergency_stop.is_engaged() is False


def test_engage_halts_execution():
    emergency_stop.engage("user panicked", source="human")
    assert emergency_stop.is_engaged() is True
    with pytest.raises(emergency_stop.EmergencyStopError):
        emergency_stop.check("run_shell")


def test_agent_cannot_clear_the_stop():
    emergency_stop.engage("test", source="human")
    assert emergency_stop.clear(source="agent") is False
    assert emergency_stop.clear(source="model") is False
    assert emergency_stop.clear() is False
    assert emergency_stop.is_engaged() is True


def test_human_can_clear_the_stop():
    emergency_stop.engage("test", source="human")
    assert emergency_stop.clear(source=emergency_stop.HUMAN) is True
    assert emergency_stop.is_engaged() is False


def test_agent_may_engage_the_stop():
    emergency_stop.engage("saw something dangerous", source="agent")
    assert emergency_stop.is_engaged() is True


def test_stop_survives_process_restart():
    emergency_stop.engage("persist me", source="human")
    emergency_stop._flag.clear()          # simulate a fresh process
    assert emergency_stop.is_engaged() is True


def test_stop_phrase_detection():
    assert emergency_stop.phrase_engages("PinPoint STOP EVERYTHING right now")
    assert emergency_stop.phrase_engages("emergency stop")
    assert emergency_stop.phrase_engages("keep going, this is great") is None


def test_status_reports_reason():
    emergency_stop.engage("disk filling up", source="human")
    assert "disk filling up" in emergency_stop.status()["reason"]


# ── Audit ─────────────────────────────────────────────────────────────────────

def test_audit_records_and_reads_back():
    audit.record("tool_call", tool="write_file", params={"filename": "a"}, result="ok")
    entries = audit.query(tool="write_file")
    assert entries and entries[-1]["result"] == "ok"


def test_audit_redacts_secret_keys():
    audit.record("tool_call", tool="send_message", params={"api_key": "sk-verysecret1234567"})
    assert audit.query(tool="send_message")[-1]["params"]["api_key"] == "[REDACTED]"


def test_audit_redacts_secrets_hidden_in_free_text():
    audit.record("tool_call", tool="run_shell",
                 params={"command": "curl -u ghp_abcdefghijklmnopqrst1234"})
    written = audit.query(tool="run_shell")[-1]["params"]["command"]
    assert "ghp_abcdefghijklmnopqrst1234" not in written


def test_audit_survives_unserializable_values():
    audit.record("tool_call", tool="odd", params={"obj": object()})
    assert audit.query(tool="odd")


def test_audit_summary_counts():
    audit.record("tool_call", tool="a", result="ok")
    audit.record("tool_call", tool="a", error="boom")
    summary = audit.summarize()
    assert summary["total"] == 2 and summary["errors"] == 1 and summary["by_tool"]["a"] == 2


def test_permission_denials_are_audited():
    permissions.set_profile(permissions.AUTONOMOUS, source="agent")   # refused
    assert any(e["action"] == "profile_change_refused" for e in audit.query())


def test_emergency_stop_clear_refusal_is_audited():
    emergency_stop.engage("x", source="human")
    emergency_stop.clear(source="agent")
    assert any(e["action"] == "emergency_stop_clear_refused" for e in audit.query())
