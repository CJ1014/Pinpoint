"""Stage 5a: intent parsing."""

from pinpoint.agent import intent as I


def parse(text):
    return I.parse(text)


# ── Kind detection ────────────────────────────────────────────────────────────

def test_fix_request():
    assert parse("Investigate why my project won't build and fix it").kind in (I.FIX, I.INVESTIGATE)


def test_broken_build_is_a_fix():
    assert parse("my project won't build").kind == I.FIX


def test_restore_server_is_a_fix():
    assert parse("Get my Minecraft server running again").kind == I.FIX


def test_build_request():
    assert parse("Build me a todo app in HTML").kind == I.BUILD


def test_monitor_request():
    assert parse("Monitor my server and restart it if it crashes").kind == I.MONITOR


def test_message_request():
    assert parse("Text John that I'm running late").kind == I.COMMUNICATE


def test_call_request():
    assert parse("Call John and ask whether he's still coming").kind == I.COMMUNICATE


def test_research_request():
    assert parse("Find the latest documentation for this library").kind == I.RESEARCH


def test_reminder_request():
    assert parse("Remind me tomorrow at 9").kind == I.SCHEDULE


def test_unrecognised_request_is_unknown_not_guessed():
    assert parse("hmm").kind == I.UNKNOWN


# ── Priority and autonomy ─────────────────────────────────────────────────────

def test_urgent_priority():
    assert parse("fix the build ASAP").priority == I.PRIORITY_URGENT


def test_low_priority():
    assert parse("clean up the tests when you get a chance").priority == I.PRIORITY_LOW


def test_high_autonomy_from_keep_working():
    assert parse("Keep working on this project until the tests pass").autonomy == I.AUTONOMY_HIGH


def test_low_autonomy_from_ask_me_first():
    assert parse("update the deps but ask me before you install anything").autonomy == I.AUTONOMY_LOW


def test_ask_me_first_beats_keep_going():
    text = "keep working on it, but ask me before sending anything"
    assert parse(text).autonomy == I.AUTONOMY_LOW


# ── Conditions ────────────────────────────────────────────────────────────────

def test_success_conditions_are_populated():
    assert parse("fix my build").success_conditions


def test_communicate_requires_provider_confirmation():
    conditions = " ".join(parse("Text John that I'm late").success_conditions)
    assert "provider confirms" in conditions


def test_until_condition_becomes_success_condition():
    objective = parse("Keep working on this until the tests pass")
    assert "the tests pass" in objective.until_condition
    assert objective.long_running is True


def test_deadline_extraction():
    assert "9" in parse("Remind me tomorrow at 9 am").deadline or \
           parse("Remind me tomorrow at 9 am").deadline


def test_failure_conditions_exist():
    assert parse("fix my build").failure_conditions


# ── Constraints ───────────────────────────────────────────────────────────────

def test_negative_constraint_captured():
    objective = parse("Update my project but don't touch the database config")
    assert any("database config" in c for c in objective.constraints)


def test_always_constraint_captured():
    assert parse("Install the mod, always use Modrinth when possible").constraints


def test_multiple_constraints():
    objective = parse("Fix it without installing anything and don't change the API")
    assert len(objective.constraints) >= 2


# ── Entities and unknowns ─────────────────────────────────────────────────────

def test_recipient_extracted():
    assert "John" in parse("Text John that I'm running late").entities["recipients"]


def test_me_is_not_a_recipient():
    assert "recipients" not in parse("Remind me tomorrow at 9").entities


def test_url_extracted():
    assert parse("read https://example.com/docs").entities["urls"]


def test_port_extracted():
    assert "8080" in parse("check the server on port 8080").entities["ports"]


def test_unknown_recipient_identity_is_recorded_not_guessed():
    objective = parse("Text John that I'm running late")
    assert any("which contact 'John'" in u for u in objective.unknowns)


def test_missing_recipient_is_an_unknown():
    assert any("who the recipient is" in u for u in parse("send a text message").unknowns)


def test_ambiguous_recipient_blocks_with_a_question():
    question = I.needs_clarification(parse("Text John that I'm late"))
    assert question and "John" in question


def test_missing_message_content_is_a_question():
    objective = parse("Text Sarah")
    assert I.needs_clarification(objective) is not None


def test_fix_with_no_target_does_not_block():
    # The agent can go and find out which project is broken itself.
    assert I.needs_clarification(parse("figure out why the build is broken")) is None


def test_vague_fix_records_the_unknown_target():
    assert any("which project" in u for u in parse("fix the build").unknowns)


# ── Plumbing ──────────────────────────────────────────────────────────────────

def test_pinpoint_prefix_is_stripped():
    assert parse("PinPoint, build me a game").goal.lower().startswith("build")


def test_objective_serializes():
    assert parse("fix my build").to_dict()["kind"] == I.FIX


def test_render_lists_unknowns():
    assert "must not be guessed" in parse("Text John hello").render()


def test_empty_input_is_safe():
    objective = parse("")
    assert objective.kind == I.UNKNOWN and objective.goal == ""
