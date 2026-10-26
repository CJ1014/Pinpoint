"""Stage 7: tiered memory and preference learning."""

import pytest

from pinpoint.memory import preferences as PR, store as S
from pinpoint.security import permissions


@pytest.fixture
def mem():
    return S.MemoryStore()


@pytest.fixture
def prefs():
    return PR.PreferenceStore()


# ── Storing ───────────────────────────────────────────────────────────────────

def test_remember_returns_a_record(mem):
    assert mem.remember("the build uses gradle").content == "the build uses gradle"


def test_records_carry_provenance(mem):
    record = mem.remember("x", source="tool_result", confidence=0.4)
    assert record.source == "tool_result" and record.confidence == 0.4
    assert record.timestamp and record.relevance is not None


def test_empty_content_is_not_stored(mem):
    assert mem.remember("   ") is None


def test_long_term_requires_explicit_permission(mem):
    assert mem.remember("CJ is 13", tier=S.LONG_TERM, relevance=0.9) is None
    assert mem.remember("CJ is 13", tier=S.LONG_TERM, relevance=0.9, permitted=True)


def test_low_relevance_is_refused_by_episodic(mem):
    assert mem.remember("trivia", tier=S.EPISODIC, relevance=0.1) is None


def test_working_tier_accepts_anything(mem):
    assert mem.remember("ran ls", tier=S.WORKING, relevance=0.0) is not None


def test_memory_persists_across_instances(mem):
    mem.remember("persisted fact", tier=S.EPISODIC, relevance=0.8)
    assert S.MemoryStore().all(S.EPISODIC)


# ── Recall ────────────────────────────────────────────────────────────────────

def test_recall_ranks_by_query_overlap(mem):
    mem.remember("the gradle build needs java 17", tier=S.EPISODIC, relevance=0.6)
    mem.remember("the cat sat on the mat", tier=S.EPISODIC, relevance=0.6)
    assert "gradle" in mem.recall("gradle build")[0].content


def test_recall_filters_by_tier(mem):
    mem.remember("working note")
    mem.remember("episode", tier=S.EPISODIC, relevance=0.8)
    assert all(r.tier == S.EPISODIC for r in mem.recall(tier=S.EPISODIC))


def test_recall_filters_by_category(mem):
    mem.remember("a", category="build")
    mem.remember("b", category="chat")
    assert len(mem.recall(category="build")) == 1


def test_recall_increments_access_count(mem):
    record = mem.remember("touch me", tier=S.EPISODIC, relevance=0.8)
    mem.recall("touch")
    assert mem.get(record.id).access_count == 1


def test_recall_respects_limit(mem):
    for i in range(10):
        mem.remember(f"note {i}")
    assert len(mem.recall(limit=3)) == 3


def test_context_block_shows_provenance(mem):
    mem.remember("gradle needs java 17", tier=S.EPISODIC, relevance=0.8,
                 source="observation")
    block = mem.context_block("gradle")
    assert "observation" in block and "confidence" in block


# ── Retention ─────────────────────────────────────────────────────────────────

def test_expired_records_are_pruned(mem):
    record = mem.remember("stale", tier=S.EPISODIC, relevance=0.8)
    record.timestamp = "2000-01-01T00:00:00+00:00"
    mem.save()
    assert S.MemoryStore().prune()["expired"] == 1


def test_expired_records_are_not_recalled(mem):
    record = mem.remember("stale", tier=S.WORKING)
    record.timestamp = "2000-01-01T00:00:00+00:00"
    mem.save()
    assert mem.recall("stale") == []


def test_procedural_memory_does_not_expire(mem):
    record = mem.record_procedure("fix a build", ["read logs", "install dep"])
    record.timestamp = "2000-01-01T00:00:00+00:00"
    mem.save()
    assert not S.MemoryStore().get(record.id).expired()


def test_capacity_keeps_the_most_valuable(monkeypatch, mem):
    monkeypatch.setitem(S.POLICIES, S.EPISODIC,
                        S.RetentionPolicy(ttl_days=90.0, max_items=2, min_relevance=0.0))
    mem.remember("low", tier=S.EPISODIC, relevance=0.1)
    mem.remember("high", tier=S.EPISODIC, relevance=0.95)
    mem.remember("medium", tier=S.EPISODIC, relevance=0.6)
    kept = {r.content for r in mem.all(S.EPISODIC)}
    assert "high" in kept and "low" not in kept


def test_clear_working_leaves_other_tiers(mem):
    mem.remember("working")
    mem.remember("episode", tier=S.EPISODIC, relevance=0.8)
    mem.clear_working()
    assert len(mem.all(S.WORKING)) == 0 and len(mem.all(S.EPISODIC)) == 1


def test_forget_removes_a_record(mem):
    record = mem.remember("temporary")
    assert mem.forget(record.id) and mem.get(record.id) is None


# ── Promotion and consolidation ───────────────────────────────────────────────

def test_promotion_to_long_term_needs_permission(mem):
    record = mem.remember("CJ prefers dark mode", relevance=0.9)
    assert mem.promote(record.id, S.LONG_TERM) is None
    assert mem.promote(record.id, S.LONG_TERM, permitted=True).tier == S.LONG_TERM


def test_consolidation_promotes_the_best_and_drops_the_rest(mem):
    mem.remember("noise", relevance=0.05)
    mem.remember("the build breaks on java 21", relevance=0.9)
    mem.consolidate(session="1", summary="fixed the build")
    assert len(mem.all(S.WORKING)) == 0
    contents = " ".join(r.content for r in mem.all(S.EPISODIC))
    assert "java 21" in contents and "noise" not in contents


def test_consolidation_records_the_session_summary(mem):
    mem.consolidate(session="7", summary="investigated the crash")
    assert any("investigated the crash" in r.content for r in mem.all(S.EPISODIC))


# ── Procedural memory ─────────────────────────────────────────────────────────

def test_procedures_are_recallable_by_goal(mem):
    mem.record_procedure("fix a gradle build",
                         ["read the error", "check the dependency", "rerun"],
                         goal_kind="fix")
    assert mem.procedures_for("gradle build is broken")


def test_failed_procedures_rank_below_successful_ones(mem):
    good = mem.record_procedure("approach A", ["x"], outcome="worked")
    bad = mem.record_procedure("approach B", ["y"], outcome="failed")
    assert good.relevance > bad.relevance


def test_stats_report_tiers(mem):
    mem.remember("a")
    mem.remember("b", tier=S.EPISODIC, relevance=0.8)
    stats = mem.stats()
    assert stats["by_tier"][S.WORKING] == 1 and stats["by_tier"][S.EPISODIC] == 1


# ── Preferences ───────────────────────────────────────────────────────────────

def test_explicit_style_preference_is_learned(prefs):
    found = prefs.learn("Always use Modrinth when possible")
    assert found and "Modrinth" in found[0].value


def test_autonomy_preference_is_learned(prefs):
    prefs.learn("Don't ask me before running tests")
    assert prefs.grants_autonomy_for("run_tests") is True


def test_approval_preference_is_learned(prefs):
    prefs.learn("Never send messages without asking")
    assert prefs.wants_approval_for("send_message") is True


def test_ask_first_beats_dont_ask_in_the_same_sentence(prefs):
    prefs.learn("don't ask me before running tests, but always ask before you delete anything")
    assert prefs.grants_autonomy_for("run_tests") is True
    assert prefs.wants_approval_for("delete_file") is True


def test_preferences_carry_source_and_confidence(prefs):
    preference = prefs.learn("Always use TypeScript")[0]
    assert preference.source == PR.USER and preference.confidence > 0.5
    assert preference.timestamp and preference.scope


def test_agent_guesses_are_low_confidence(prefs):
    preference = prefs.learn("Always use TypeScript", source=PR.AGENT)[0]
    assert preference.confidence < 0.5


def test_nothing_is_inferred_from_ordinary_conversation(prefs):
    assert prefs.learn("I built three games this week and they were fun") == []


def test_a_newer_statement_supersedes_the_old_one(prefs):
    prefs.learn("Don't ask me before running tests")
    prefs.learn("Always ask before running tests")
    assert prefs.wants_approval_for("run_tests") is True


def test_user_preference_becomes_a_real_grant(prefs):
    permissions.set_profile(permissions.SAFE, source=permissions.HUMAN)
    prefs.learn("Don't ask me before running installs")
    prefs.apply_permissions()
    assert "pip_install" in permissions.list_grants()


def test_agent_stated_preference_cannot_grant_permission(prefs):
    permissions.set_profile(permissions.SAFE, source=permissions.HUMAN)
    prefs.learn("Don't ask me before running installs", source=PR.AGENT)
    prefs.apply_permissions()
    assert "pip_install" not in permissions.list_grants()


def test_ask_first_preference_revokes_an_existing_grant(prefs):
    permissions.grant("send_message", source=permissions.HUMAN, scope="always")
    prefs.learn("Never send messages without asking")
    prefs.apply_permissions()
    assert "send_message" not in permissions.list_grants()


def test_preferences_persist(prefs):
    prefs.learn("Always use Modrinth")
    assert PR.PreferenceStore().all()


def test_context_block_lists_preferences(prefs):
    prefs.learn("Always use Modrinth")
    assert "Modrinth" in prefs.context_block()


def test_forget_a_preference(prefs):
    preference = prefs.learn("Always use Modrinth")[0]
    assert prefs.forget(preference.id) and not prefs.all()
