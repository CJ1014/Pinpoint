"""How PinPoint talks to CJ.

Internal logs stay detailed. User-facing text does not: no ``[TOOL CALL]``,
no ``[VERIFICATION]``, no narrating the machinery. Say what happened, say what
is actually known, and say plainly when something could not be confirmed.

Confidence is expressed by hedging honestly, not by adding percentages to
sentences. "I think the cause is X, but I haven't proved it" beats "root cause
identified (confidence 72%)".
"""

from typing import List, Optional

from pinpoint.action import result as R
from pinpoint.agent import planner as P

# Markers the internal pipeline uses that must never reach the user.
_INTERNAL_MARKERS = ("[TOOL CALL]", "[TOOL RESULT]", "[VERIFICATION",
                     "[GOAL TREE]", "[VALUES CHECKPOINT]", "[GROUNDING CHECK]",
                     "[MULTI-FRAME]", "[CAPABILITY]", "[AUTO-REFLECT]",
                     "OBSERVED:", "INFERRED:", "UNKNOWN:", "ASSUMED:")


def clean(text: str) -> str:
    """Strip internal scaffolding out of something destined for the user."""
    lines = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if any(stripped.startswith(marker) for marker in _INTERNAL_MARKERS):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def completed(goal: str, what_changed: List[str], verified: bool = True) -> str:
    """Report a finished job."""
    lead = "Done." if verified else "I think that's done, but I couldn't verify it."
    if not what_changed:
        return f"{lead} {goal}"
    if len(what_changed) == 1:
        return f"{lead} {what_changed[0]}"
    body = "\n".join(f"  - {item}" for item in what_changed[:6])
    return f"{lead}\n{body}"


def blocked(goal: str, reason: str, tried: Optional[List[str]] = None,
            need: str = "") -> str:
    """Report being stuck, without burying the ask."""
    lines = [f"I couldn't finish {goal.lower().rstrip('.')}.", f"{reason.rstrip('.')}."]
    if tried:
        lines.append("I tried: " + "; ".join(tried[:3]) + ".")
    lines.append(need or "Tell me how you'd like to handle it and I'll pick it back up.")
    return " ".join(lines)


def unverified(action: str, detail: str = "") -> str:
    """Say clearly that something is not confirmed."""
    tail = f" ({detail})" if detail else ""
    return (f"I ran {action}, but I can't confirm it actually worked{tail}. "
            f"I'm not going to call it done until I can.")


def unavailable(what: str, workaround: str = "") -> str:
    """Say what can't be done here, then offer the nearest real thing."""
    line = f"I can't {what} with the tools I have here."
    return f"{line} {workaround}" if workaround else line


def asking(question: str) -> str:
    return question if question.endswith("?") else question + "?"


def progress(plan: P.Plan) -> str:
    """A one-line status a human would actually want."""
    stats = plan.progress()
    current = plan.next_task()
    line = f"{stats['done']} of {stats['total']} steps done"
    if stats["blocked"] or stats["failed"]:
        line += f", {stats['blocked'] + stats['failed']} stuck"
    if current:
        line += f". Working on: {current.description.lower()}"
    return line + "."


def summarize_results(results: List[R.ActionResult]) -> List[str]:
    """Turn verified action results into things a person would say happened."""
    lines: List[str] = []
    for result in results:
        if result.status == R.SUCCESS:
            if result.tool in ("write_file", "write_anywhere"):
                target = result.params.get("filename") or result.params.get("path", "")
                lines.append(f"wrote {target}")
            elif result.tool == "delete_file":
                lines.append(f"deleted {result.params.get('path', '')}")
            elif result.tool == "run_tests":
                lines.append("the tests pass")
            elif result.tool in ("run_shell", "run_python"):
                lines.append(f"ran {result.params.get('command') or result.params.get('filename', '')}")
            elif result.tool == "send_message":
                lines.append(f"the message to {result.params.get('to', '')} was accepted "
                             f"by the provider")
        elif result.status == R.UNVERIFIED:
            lines.append(f"{result.tool} ran but I couldn't confirm the result")
        elif result.status == R.UNAVAILABLE:
            lines.append(f"{result.tool} isn't available here")
    # Preserve order, drop repeats.
    return list(dict.fromkeys(lines))


def report(run) -> str:
    """The end-of-run message. ``run`` is an orchestrator RunReport."""
    goal = run.objective.goal if run.objective else "that"
    changes = summarize_results(run.results)

    if run.status == "needs_input":
        return asking(run.question)

    if run.status == "stopped":
        return "Stopped. I've saved where I got to, so we can pick it up when you're ready."

    if run.status == "completed":
        unverified_actions = [r for r in run.results if r.status == R.UNVERIFIED]
        text = completed(goal, changes, verified=not unverified_actions)
        if unverified_actions:
            text += ("\n\nOne thing I couldn't check: "
                     + unverified_actions[0].verification.get("detail", "the effect"))
        return text

    reasons = [b.get("reason", "") for b in run.blockers if b.get("reason")]
    return blocked(goal, reasons[0] if reasons else "I ran out of ways to make progress",
                   tried=changes, need=run.question)
