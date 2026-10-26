"""Stage 3: tool registry.

The sync test is the important one — it fails when a tool is added to
agent.TOOLS without declaring its risk, so the policy engine can never be
silently bypassed by a new tool nobody classified.
"""

import pytest

from pinpoint.security import permissions
from pinpoint.tools import registry


def test_every_agent_tool_is_registered():
    agent = pytest.importorskip("agent")
    declared = {t["function"]["name"] for t in agent.TOOLS}
    missing = sorted(declared - set(registry.names()))
    assert not missing, f"tools missing from the registry: {missing}"


def test_lookup_returns_spec():
    spec = registry.get("write_file")
    assert spec is not None and spec.verification == registry.V_FILE_EXISTS


def test_unknown_tool_returns_none():
    assert registry.get("no_such_tool") is None


def test_require_defaults_unknown_tool_to_risky():
    spec = registry.require("no_such_tool")
    assert spec.level == permissions.YELLOW and spec.reversible is False


def test_capability_search():
    names = {s.name for s in registry.with_capability(registry.C_COMMUNICATION)}
    assert {"send_message", "make_call"} <= names


def test_text_search_matches_description():
    assert any(s.name == "run_tests" for s in registry.search("test suite"))


def test_search_empty_returns_nothing():
    assert registry.search("") == []


def test_catalog_is_planner_readable():
    text = registry.catalog(registry.C_FILESYSTEM)
    assert "write_file" in text and "GREEN" in text or "YELLOW" in text


def test_destructive_tools_are_flagged_irreversible():
    for name in ("delete_file", "run_shell", "pip_install"):
        assert registry.get(name).reversible is False


def test_communication_tools_are_external_and_confirmed():
    for name in ("send_message", "make_call", "send_email"):
        spec = registry.get(name)
        assert spec.external_side_effect is True
        assert spec.verification == registry.V_PROVIDER_CONFIRMATION


def test_pure_reasoning_tools_have_no_side_effects():
    for name in ("think", "brainstorm", "critique"):
        spec = registry.get(name)
        assert spec.level == permissions.GREEN
        assert spec.external_side_effect is False and spec.costs_money is False


def test_policy_engine_reads_registry_attributes():
    # send_message is registered external+billable, so AUTONOMOUS must still ask.
    permissions.set_profile(permissions.AUTONOMOUS, source=permissions.HUMAN)
    assert permissions.check("send_message", {"to": "a", "body": "b"}).requires_approval


def test_registry_level_drives_classification():
    spec = registry.register(registry.ToolSpec(name="tmp_probe_tool",
                                               level=permissions.GREEN))
    try:
        assert permissions.check("tmp_probe_tool", {}).level == permissions.GREEN
    finally:
        registry._REGISTRY.pop(spec.name, None)


def test_specs_serialize():
    d = registry.get("run_shell").to_dict()
    assert d["name"] == "run_shell" and d["timeout"] > 0


def test_summary_marks_irreversible_and_external():
    assert "irreversible" in registry.get("delete_file").summary()
    assert "external" in registry.get("send_message").summary()


def test_gui_tools_declare_runtime_requirements():
    assert "pyautogui" in registry.get("type_text").requires
