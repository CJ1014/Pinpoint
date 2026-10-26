"""The seam between PinPoint v2 and the v3 subsystems.

``agent.py`` and ``tools.py`` call into here rather than importing half the
package, which keeps the changes to those two large files small and reviewable.
Every function is defensive: if a v3 subsystem fails, v2 behaviour continues
unchanged rather than the agent falling over.
"""

from typing import Any, Dict, List, Optional

from pinpoint.security import audit, emergency_stop, permissions

# Tools whose failure to be classified would matter — used by the sync report.
_UNCLASSIFIED_DEFAULT = "YELLOW"


# ── Called from tools.dispatch ────────────────────────────────────────────────

def hard_block(tool: str, params: dict) -> Optional[str]:
    """Refusals that no approval mode, profile, or grant can override.

    This is deliberately narrower than the full policy check: it adds the
    emergency stop and RED classification on top of v2's existing guardrail,
    without changing the approval experience for everything else.
    """
    try:
        if emergency_stop.is_engaged():
            info = emergency_stop.status()
            return (f"BLOCKED: emergency stop is engaged ({info.get('reason', '')}). "
                    f"Nothing runs until CJ clears it. Stop what you're doing and "
                    f"tell him where things stand.")

        decision = permissions.check(tool, params or {})
        if decision.blocked:
            return (f"BLOCKED: {decision.reason}. This is a hard limit — there is no "
                    f"mode or approval that permits it. Do it a different way.")
    except Exception:
        return None      # a broken policy check must not block ordinary work
    return None


def record_action(tool: str, params: dict, result: str) -> None:
    """Mirror a v2 tool call into the structured audit trail."""
    try:
        audit.record("tool_call", tool=tool, params=params or {},
                     result=result if isinstance(result, str) else str(result))
    except Exception:
        pass


def observe(tool: str, params: dict, result: str) -> Optional[str]:
    """Independently verify a v2 tool result.

    Returns a note to feed back to the model when the effect could not be
    confirmed, or None when there is nothing worth saying.
    """
    try:
        from pinpoint.action import verify
        from pinpoint.tools import registry

        spec = registry.get(tool)
        if spec is None or spec.verification == registry.V_NONE:
            return None
        outcome = verify.verify(tool, params or {}, result or "",
                                method=spec.verification)
        if outcome.verified is False:
            return (f"[REALITY] {tool} did not do what its output suggests: "
                    f"{outcome.detail}. Treat it as failed, not done.")
        if outcome.verified is None and spec.verification in (
                registry.V_PROVIDER_CONFIRMATION, registry.V_EXIT_CODE):
            return (f"[REALITY] {tool} ran but the effect is unconfirmed: "
                    f"{outcome.detail}. Do not report it as done.")
    except Exception:
        return None
    return None


# ── Called from agent.run ─────────────────────────────────────────────────────

def session_start(session_num: Any = "", order: str = "") -> str:
    """Prepare v3 state for a session and return a prompt block for the model."""
    blocks: List[str] = []
    try:
        permissions.clear_session_grants()
    except Exception:
        pass

    try:
        from pinpoint.memory import preferences
        applied = preferences.preferences().apply_permissions()
        block = preferences.preferences().context_block()
        if block:
            blocks.append(block)
        if applied:
            blocks.append("Standing permissions from those preferences: "
                          + ", ".join(f"{tool} ({state})"
                                      for tool, state in applied.items()))
    except Exception:
        pass

    try:
        from pinpoint import toolapi
        blocks.append(toolapi.capability_report())
        comms = toolapi.communication_status()
        if "unavailable" in comms:
            blocks.append(comms)
    except Exception:
        pass

    if order:
        try:
            blocks.append(objective_block(order))
        except Exception:
            pass

    try:
        audit.record("session_start", session=session_num, goal=order,
                     params={"profile": permissions.get_profile()})
    except Exception:
        pass

    return "\n\n".join(b for b in blocks if b)


def objective_block(request: str) -> str:
    """Parse a request into a structured objective + plan, rendered for the model."""
    from pinpoint.agent import intent, planner

    objective = intent.parse(request)
    question = intent.needs_clarification(objective)
    plan = planner.plan_for(objective)

    lines = ["[OBJECTIVE]", objective.render(), "", "[PLAN]", plan.render()]
    if question:
        lines += ["", f"[ASK FIRST] Before doing anything else, ask CJ: {question}"]
    if objective.unknowns:
        lines += ["", "Do not invent answers to the unknowns above. "
                      "Find them out or ask."]
    return "\n".join(lines)


def handle_user_text(text: str) -> Dict[str, Any]:
    """Process something CJ said. Returns what the agent needs to react to.

    Two things happen here that cannot wait for the model: an emergency stop
    phrase halts execution immediately, and explicitly stated preferences are
    recorded with the user's authority behind them.
    """
    outcome: Dict[str, Any] = {"stopped": False, "preferences": [], "note": ""}
    if not text:
        return outcome

    try:
        phrase = emergency_stop.phrase_engages(text)
        if phrase:
            emergency_stop.engage(f"CJ said '{phrase}'", source=emergency_stop.HUMAN)
            outcome["stopped"] = True
            outcome["note"] = ("Everything is halted. Nothing else will run until "
                               "you tell me to resume.")
            return outcome
    except Exception:
        pass

    try:
        from pinpoint.memory import preferences
        store = preferences.preferences()
        learned = store.learn(text, source=preferences.USER)
        if learned:
            store.apply_permissions()
            outcome["preferences"] = [p.statement for p in learned]
            outcome["note"] = "Noted: " + "; ".join(outcome["preferences"][:2])
    except Exception:
        pass

    return outcome


def resume_stop(source: str = "human") -> bool:
    """Clear the emergency stop. Only ever called from a human-facing path."""
    return emergency_stop.clear(source=emergency_stop.HUMAN if source == "human"
                                else source)


def due_reminders() -> List[str]:
    """Anything scheduled that is due now — checked cheaply, every iteration."""
    try:
        from pinpoint.monitoring import scheduler
        return [event.detail for event in scheduler.scheduler().fire_due()]
    except Exception:
        return []


def monitor_notices() -> List[str]:
    """Anything the watchers have noticed that the agent should react to."""
    try:
        from pinpoint import toolapi
        from pinpoint.monitoring import events

        notices = []
        for decision in toolapi.monitor().poll_once():
            if decision.outcome in (events.ACT, events.NOTIFY):
                notices.append(decision.event.summary())
        return notices
    except Exception:
        return []


def session_end(session_num: Any = "", summary: str = "") -> str:
    """Consolidate memory and report what was learned."""
    try:
        from pinpoint.memory import store
        promoted = store.store().consolidate(session=str(session_num), summary=summary)
        store.store().prune()
        audit.record("session_end", session=session_num,
                     params={"summary": summary[:200], "promoted": len(promoted)})
        return f"Kept {len(promoted)} things from this session."
    except Exception:
        return ""


def status_report() -> str:
    """A short operational summary — profile, stop state, watchers, schedule."""
    from pinpoint import toolapi

    parts = [toolapi.emergency_status(), toolapi.permission_status()]
    try:
        watchers = toolapi.list_watchers()
        if "Not watching" not in watchers:
            parts.append(watchers)
        scheduled = toolapi.list_scheduled()
        if "Nothing scheduled" not in scheduled:
            parts.append("Scheduled:\n" + scheduled)
    except Exception:
        pass
    return "\n".join(parts)


def registry_gaps() -> List[str]:
    """Tools the agent exposes that the registry does not classify."""
    try:
        import agent

        from pinpoint.tools import registry
        declared = {t["function"]["name"] for t in agent.TOOLS}
        return sorted(declared - set(registry.names()))
    except Exception:
        return []
