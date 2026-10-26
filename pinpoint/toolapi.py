"""String-returning wrappers that ``tools.dispatch`` can call directly.

Everything here returns the kind of plain text the v2 tool layer expects, while
the real work happens in the v3 subsystems. Where a result needs to be
verifiable, the string carries the evidence — a provider confirmation id, a
concrete count — because the verification layer reads these strings.
"""

from typing import Optional

from pinpoint.communication import contacts as C, manager as CM
from pinpoint.communication.providers import base as CB
from pinpoint.monitoring import events as E, scheduler as SCH, watchers as W

# One monitor per process; watchers registered by the agent live here.
_monitor: Optional[W.Monitor] = None


def monitor() -> W.Monitor:
    global _monitor
    if _monitor is None:
        _monitor = W.Monitor()
    return _monitor


# ── Communication ─────────────────────────────────────────────────────────────

def send_message(to: str, body: str) -> str:
    return CM.send_message(to, body).render()


def send_email(to: str, subject: str, body: str) -> str:
    return CM.send_email(to, subject, body).render()


def make_call(to: str, purpose: str = "", script: str = "",
              on_behalf_of: str = "CJ") -> str:
    return CM.make_call(to, purpose=purpose, script=script,
                        on_behalf_of=on_behalf_of).render()


def get_call_status(call_id: str) -> str:
    status = CM.get_call_status(call_id)
    if "error" in status:
        return f"Error: {status['error']}"
    return (f"Call {status.get('id')}: {status.get('status')} "
            f"({status.get('duration') or '?'}s)")


def get_messages(limit: int = 10) -> str:
    messages = CM.get_messages(limit)
    if not messages:
        return ("No messages — either the inbox is empty or no messaging provider "
                "is configured. " + CM.status()[CB.SMS]["detail"])
    return "\n".join(f"[{m.get('sent_at', '')}] {m.get('from')}: {m.get('body', '')[:200]}"
                     for m in messages)


def resolve_contact(name: str) -> str:
    resolution = CM.resolve(name)
    if resolution.ok:
        return f"Resolved '{name}' to {resolution.contact.label()}"
    return resolution.question


def add_contact(name: str, phone: str = "", email: str = "", note: str = "") -> str:
    if not (phone or email):
        return f"Error: a contact needs at least a phone number or an email address."
    contact = C.contacts().add(C.Contact(name=name, phone=phone, email=email, note=note))
    return f"Added contact {contact.label()}"


def communication_status() -> str:
    lines = ["Communication channels:"]
    for kind, info in CM.status().items():
        mark = "available" if info["available"] else "unavailable"
        lines.append(f"  {kind}: {mark} — {info['detail']}")
    return "\n".join(lines)


# ── Scheduling ────────────────────────────────────────────────────────────────

def schedule_task(what: str, when: str = "") -> str:
    task = SCH.scheduler().schedule(what, when or what)
    if task is None:
        return (f"Error: I couldn't work out a time from '{when or what}'. "
                f"Give me something concrete like 'tomorrow at 9am', "
                f"'in 20 minutes', or 'every day at 8am'.")
    return f"Scheduled: {task.describe()} (id {task.id})"


def list_scheduled() -> str:
    return SCH.scheduler().describe()


def cancel_scheduled(task_id: str) -> str:
    if SCH.scheduler().cancel(task_id):
        return f"Cancelled scheduled task {task_id}"
    return f"Error: no scheduled task with id {task_id}"


def check_schedule() -> str:
    """Fire anything due. Returns what fired."""
    fired = SCH.scheduler().fire_due()
    if not fired:
        return "Nothing is due."
    return "\n".join(f"DUE: {event.detail}" for event in fired)


# ── Monitoring ────────────────────────────────────────────────────────────────

def watch(kind: str, target: str, response: str = "", auto: bool = False) -> str:
    """Start watching something. ``kind`` is port / process / file / http / command."""
    kind = (kind or "").strip().lower()
    try:
        if kind == "port":
            host, _, port = target.rpartition(":")
            watcher = W.PortWatcher(host or "localhost", int(port))
            event_types = [E.PORT_DOWN, E.PORT_UP]
        elif kind == "process":
            watcher = W.ProcessWatcher(target)
            event_types = [E.PROCESS_CRASHED, E.PROCESS_STARTED]
        elif kind == "file":
            watcher = W.FileWatcher(target)
            event_types = [E.FILE_CHANGED]
        elif kind == "http":
            watcher = W.HttpWatcher(target)
            event_types = [E.HTTP_DOWN, E.HTTP_UP]
        elif kind == "command":
            watcher = W.CommandWatcher(target)
            event_types = [E.COMMAND_FAILED, E.COMMAND_RECOVERED]
        else:
            return (f"Error: '{kind}' is not a kind of watcher. "
                    f"Use port, process, file, http, or command.")
    except (ValueError, TypeError) as exc:
        return f"Error: couldn't set up that watcher — {exc}"

    active = monitor()
    active.add(watcher)
    active.pipeline.subscribe(
        event_types, goal=response or f"watch {target}",
        response=response, policy=E.ACT if auto else E.NOTIFY,
        sources=[watcher.name])
    watcher.poll()      # establish the baseline immediately
    active.start(interval=30.0)
    return (f"Watching {kind} {target}. Currently: {watcher.state}. "
            f"{'I will act on changes.' if auto else 'I will tell you about changes.'}")


def list_watchers() -> str:
    status = monitor().status()
    if not status["watchers"]:
        return "Not watching anything."
    lines = [f"Monitor running: {status['running']}"]
    lines += [f"  {line}" for line in status["watchers"]]
    return "\n".join(lines)


def check_watchers() -> str:
    """Poll every watcher once and report what changed."""
    decisions = monitor().poll_once()
    if not decisions:
        return "Nothing has changed."
    return "\n".join(f"{d.outcome.upper()}: {d.event.summary()}" for d in decisions)


def stop_watching(key: str = "") -> str:
    active = monitor()
    if not key:
        active.stop()
        return "Stopped the monitor."
    return ("Stopped watching " + key) if active.remove(key) else \
        f"Error: not watching '{key}'"


# ── Honesty about capability ──────────────────────────────────────────────────

def capability_report() -> str:
    """What can actually be done in this environment, and the workaround if not."""
    from pinpoint.action import capabilities

    lines = ["What I can actually do here:"]
    for name, info in capabilities.report().items():
        if info["available"] is True:
            lines.append(f"  {name}: yes")
        elif info["available"] is False:
            lines.append(f"  {name}: no — {info['detail']}. {info['workaround']}")
        else:
            lines.append(f"  {name}: unknown until tried")
    return "\n".join(lines)


def emergency_status() -> str:
    from pinpoint.security import emergency_stop

    status = emergency_stop.status()
    if not status.get("engaged"):
        return "Emergency stop is not engaged."
    return (f"EMERGENCY STOP is engaged: {status.get('reason', '')} "
            f"(set by {status.get('source', '?')}). Only CJ can clear it.")


def permission_status() -> str:
    from pinpoint.security import approval, permissions

    grants = permissions.list_grants()
    lines = [f"Autonomy profile: {permissions.get_profile()}"]
    if grants:
        lines.append("Standing permissions: " + ", ".join(grants))
    stats = approval.stats()
    lines.append(f"Approvals this session: {stats['approved']} granted, "
                 f"{stats['denied']} declined")
    return "\n".join(lines)
