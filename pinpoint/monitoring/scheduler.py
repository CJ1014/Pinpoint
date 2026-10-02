"""Time-based tasks.

"Remind me tomorrow at 9" becomes a stored task with an absolute due time, so
it survives a restart. Parsing is deterministic and refuses what it cannot
read: an unparseable time returns nothing rather than a plausible-looking guess
at when the user meant.
"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from pinpoint import jsonstore, paths
from pinpoint.monitoring import events as E

ONCE = "once"
DAILY = "daily"
HOURLY = "hourly"
INTERVAL = "interval"

_UNIT_SECONDS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400,
                 "week": 604800}

_IN = re.compile(r"\bin\s+(\d+)\s*(second|minute|hour|day|week)s?\b", re.I)
_EVERY_N = re.compile(r"\bevery\s+(\d+)\s*(second|minute|hour|day)s?\b", re.I)
_EVERY_DAY = re.compile(r"\bevery\s+day(?:\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?)?\b", re.I)
_EVERY_HOUR = re.compile(r"\bevery\s+hour\b", re.I)
_AT = re.compile(r"\b(tomorrow|today|tonight)?\s*(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.I)
_AT_24 = re.compile(r"\bat\s+(\d{1,2}):(\d{2})\b")
_TOMORROW_BARE = re.compile(r"\btomorrow\b", re.I)
_ISO = re.compile(r"\b(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?)\b")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def parse_when(text: str, now: Optional[datetime] = None) -> Optional[Tuple]:
    """Parse a time phrase into ``(due, recurrence, interval_seconds)``.

    Returns None when the phrase contains no time it can read.
    """
    now = now or _now()
    blob = text or ""

    match = _ISO.search(blob)
    if match:
        try:
            parsed = datetime.fromisoformat(match.group(1).replace(" ", "T"))
            if not parsed.tzinfo:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed, ONCE, 0
        except ValueError:
            pass

    match = _EVERY_N.search(blob)
    if match:
        seconds = int(match.group(1)) * _UNIT_SECONDS[match.group(2).lower()]
        return now + timedelta(seconds=seconds), INTERVAL, seconds

    match = _EVERY_DAY.search(blob)
    if match:
        hour = int(match.group(1)) if match.group(1) else 9
        minute = int(match.group(2) or 0)
        hour = _to_24h(hour, match.group(3))
        due = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if due <= now:
            due += timedelta(days=1)
        return due, DAILY, 86400

    if _EVERY_HOUR.search(blob):
        return now + timedelta(hours=1), HOURLY, 3600

    match = _IN.search(blob)
    if match:
        seconds = int(match.group(1)) * _UNIT_SECONDS[match.group(2).lower()]
        return now + timedelta(seconds=seconds), ONCE, 0

    match = _AT.search(blob)
    if match:
        day_word = (match.group(1) or "").lower()
        hour = _to_24h(int(match.group(2)), match.group(4))
        minute = int(match.group(3) or 0)
        due = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if day_word == "tomorrow":
            due += timedelta(days=1)
        elif due <= now:
            due += timedelta(days=1)
        return due, ONCE, 0

    match = _AT_24.search(blob)
    if match:
        due = now.replace(hour=int(match.group(1)) % 24, minute=int(match.group(2)),
                          second=0, microsecond=0)
        if _TOMORROW_BARE.search(blob) or due <= now:
            due += timedelta(days=1)
        return due, ONCE, 0

    if _TOMORROW_BARE.search(blob):
        due = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0,
                                                microsecond=0)
        return due, ONCE, 0

    return None


def _to_24h(hour: int, meridiem: Optional[str]) -> int:
    meridiem = (meridiem or "").lower()
    if meridiem == "pm" and hour < 12:
        return hour + 12
    if meridiem == "am" and hour == 12:
        return 0
    return hour % 24


@dataclass
class ScheduledTask:
    """One thing to do at a time."""
    what: str
    due: str
    recurrence: str = ONCE
    interval_seconds: int = 0
    source_text: str = ""
    active: bool = True
    runs: int = 0
    last_run: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created_at: str = field(default_factory=lambda: _now().isoformat())

    def due_at(self) -> datetime:
        try:
            parsed = datetime.fromisoformat(self.due)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return _now() + timedelta(days=3650)

    def is_due(self, now: Optional[datetime] = None) -> bool:
        return self.active and self.due_at() <= (now or _now())

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict) -> "ScheduledTask":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def describe(self) -> str:
        when = self.due_at().strftime("%Y-%m-%d %H:%M UTC")
        repeat = "" if self.recurrence == ONCE else f" (repeats {self.recurrence})"
        return f"{self.what} — {when}{repeat}"


class Scheduler:
    """Persistent schedule of time-based tasks."""

    def __init__(self, path: str = ""):
        self._path = path or paths.output_dir("schedule.json")
        self._tasks: List[ScheduledTask] = []
        self._loaded = False

    def load(self) -> "Scheduler":
        if not self._loaded:
            data = jsonstore.read_json(self._path, {"tasks": []})
            self._tasks = [ScheduledTask.from_dict(d) for d in data.get("tasks", [])]
            self._loaded = True
        return self

    def save(self) -> bool:
        return jsonstore.write_json(
            self._path, {"tasks": [t.to_dict() for t in self._tasks]})

    # ── scheduling ───────────────────────────────────────────────────────────

    def schedule(self, what: str, when_text: str = "",
                 due: Optional[datetime] = None,
                 now: Optional[datetime] = None) -> Optional[ScheduledTask]:
        """Schedule something. Returns None when the time cannot be read."""
        recurrence, interval = ONCE, 0
        if due is None:
            parsed = parse_when(when_text or what, now)
            if parsed is None:
                return None
            due, recurrence, interval = parsed

        self.load()
        task = ScheduledTask(what=what.strip(), due=due.isoformat(),
                             recurrence=recurrence, interval_seconds=interval,
                             source_text=when_text or what)
        self._tasks.append(task)
        self.save()
        return task

    def cancel(self, task_id: str) -> bool:
        self.load()
        for task in self._tasks:
            if task.id == task_id:
                task.active = False
                self.save()
                return True
        return False

    def remove(self, task_id: str) -> bool:
        self.load()
        before = len(self._tasks)
        self._tasks = [t for t in self._tasks if t.id != task_id]
        if len(self._tasks) != before:
            self.save()
            return True
        return False

    # ── firing ───────────────────────────────────────────────────────────────

    def due_now(self, now: Optional[datetime] = None) -> List[ScheduledTask]:
        self.load()
        return [t for t in self._tasks if t.is_due(now)]

    def fire_due(self, now: Optional[datetime] = None) -> List[E.Event]:
        """Return an event per due task, and roll recurring tasks forward."""
        now = now or _now()
        fired: List[E.Event] = []
        for task in self.due_now(now):
            fired.append(E.Event(
                type=E.SCHEDULED, source=task.id, detail=task.what,
                severity=E.INFO,
                data={"task_id": task.id, "recurrence": task.recurrence}))
            task.runs += 1
            task.last_run = now.isoformat()
            if task.recurrence == ONCE:
                task.active = False
            else:
                step = task.interval_seconds or 86400
                next_due = task.due_at() + timedelta(seconds=step)
                while next_due <= now:
                    next_due += timedelta(seconds=step)
                task.due = next_due.isoformat()
        if fired:
            self.save()
        return fired

    # ── reading ──────────────────────────────────────────────────────────────

    def upcoming(self, limit: int = 10) -> List[ScheduledTask]:
        self.load()
        active = [t for t in self._tasks if t.active]
        return sorted(active, key=lambda t: t.due_at())[:limit]

    def all(self) -> List[ScheduledTask]:
        self.load()
        return list(self._tasks)

    def describe(self) -> str:
        upcoming = self.upcoming()
        if not upcoming:
            return "Nothing scheduled."
        return "\n".join(f"  - {t.describe()}" for t in upcoming)


_default: Optional[Scheduler] = None


def scheduler() -> Scheduler:
    """Shared instance, rebuilt when PINPOINT_HOME changes (tests)."""
    global _default
    expected = paths.output_dir("schedule.json")
    if _default is None or _default._path != expected:
        _default = Scheduler(expected)
    return _default
