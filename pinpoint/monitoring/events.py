"""Event pipeline.

    EVENT → FILTER → RELEVANCE → ACTIVE GOAL → POLICY → ACT / NOTIFY / IGNORE

Every stage is deterministic. Filtering drops duplicates and noise below the
severity floor; relevance matches the event against standing subscriptions;
the goal check asks whether anything currently cares; the policy check asks the
permission engine whether the proposed response could even run unattended.

Only after all four does anything expensive happen.
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

# Event types.
PROCESS_CRASHED = "process_crashed"
PROCESS_STARTED = "process_started"
PORT_DOWN = "port_down"
PORT_UP = "port_up"
FILE_CHANGED = "file_changed"
HTTP_DOWN = "http_down"
HTTP_UP = "http_up"
COMMAND_FAILED = "command_failed"
COMMAND_RECOVERED = "command_recovered"
BUILD_FINISHED = "build_finished"
TEST_FAILED = "test_failed"
MESSAGE_RECEIVED = "message_received"
SCHEDULED = "scheduled"
RESOURCE_THRESHOLD = "resource_threshold"

# Severities, ordered.
INFO, WARNING, CRITICAL = "info", "warning", "critical"
_SEVERITY_RANK = {INFO: 0, WARNING: 1, CRITICAL: 2}

# Pipeline outcomes.
ACT = "act"
NOTIFY = "notify"
IGNORE = "ignore"


@dataclass
class Event:
    """Something that happened in the world."""
    type: str
    source: str = ""
    detail: str = ""
    severity: str = INFO
    data: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def key(self) -> str:
        """Identity for deduplication — type plus source, not the whole payload."""
        return f"{self.type}:{self.source}"

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def summary(self) -> str:
        return f"[{self.severity}] {self.type} on {self.source or '?'}: {self.detail}"


@dataclass
class Subscription:
    """A standing interest in some class of event."""
    event_types: List[str]
    goal: str = ""
    response: str = ""                       # what to do, in words
    policy: str = NOTIFY                     # ACT | NOTIFY | IGNORE
    min_severity: str = INFO
    sources: List[str] = field(default_factory=list)   # empty = any source
    tool: str = ""                           # the tool an ACT response would use
    handler: Optional[Callable] = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    active: bool = True

    def matches(self, event: Event) -> bool:
        if not self.active:
            return False
        if self.event_types and event.type not in self.event_types:
            return False
        if self.sources and event.source not in self.sources:
            return False
        return _SEVERITY_RANK.get(event.severity, 0) >= _SEVERITY_RANK.get(
            self.min_severity, 0)

    def to_dict(self) -> dict:
        return {"id": self.id, "event_types": self.event_types, "goal": self.goal,
                "response": self.response, "policy": self.policy,
                "min_severity": self.min_severity, "sources": self.sources,
                "tool": self.tool, "active": self.active}


@dataclass
class Decision:
    """What the pipeline concluded about one event."""
    outcome: str
    event: Event
    reason: str = ""
    subscription: Optional[Subscription] = None
    response: str = ""
    requires_approval: bool = False

    def to_dict(self) -> dict:
        return {"outcome": self.outcome, "event": self.event.to_dict(),
                "reason": self.reason, "response": self.response,
                "requires_approval": self.requires_approval,
                "subscription": self.subscription.id if self.subscription else ""}


class EventPipeline:
    """Deterministic filter → relevance → goal → policy chain."""

    def __init__(self, dedupe_seconds: float = 30.0, min_severity: str = INFO):
        self.subscriptions: List[Subscription] = []
        self.dedupe_seconds = dedupe_seconds
        self.min_severity = min_severity
        self.decisions: List[Decision] = []
        self._seen: Dict[str, float] = {}
        self._clock: Callable[[], float] = time.monotonic

    # ── subscriptions ────────────────────────────────────────────────────────

    def subscribe(self, event_types, *, goal: str = "", response: str = "",
                  policy: str = NOTIFY, min_severity: str = INFO,
                  sources: Optional[List[str]] = None, tool: str = "",
                  handler: Optional[Callable] = None) -> Subscription:
        subscription = Subscription(
            event_types=list(event_types), goal=goal, response=response,
            policy=policy, min_severity=min_severity, sources=list(sources or []),
            tool=tool, handler=handler)
        self.subscriptions.append(subscription)
        return subscription

    def unsubscribe(self, subscription_id: str) -> bool:
        before = len(self.subscriptions)
        self.subscriptions = [s for s in self.subscriptions if s.id != subscription_id]
        return len(self.subscriptions) != before

    def active_subscriptions(self) -> List[Subscription]:
        return [s for s in self.subscriptions if s.active]

    # ── the pipeline ─────────────────────────────────────────────────────────

    def _is_duplicate(self, event: Event) -> bool:
        now = self._clock()
        last = self._seen.get(event.key())
        self._seen[event.key()] = now
        return last is not None and (now - last) < self.dedupe_seconds

    def decide(self, event: Event) -> Decision:
        """Run one event through every stage. No model is consulted."""
        # 1. FILTER — severity floor.
        if _SEVERITY_RANK.get(event.severity, 0) < _SEVERITY_RANK.get(self.min_severity, 0):
            return self._record(Decision(IGNORE, event, "below the severity floor"))

        # 1b. FILTER — duplicate suppression.
        if self._is_duplicate(event):
            return self._record(Decision(IGNORE, event,
                                         f"duplicate within {self.dedupe_seconds:.0f}s"))

        # 2. RELEVANCE — does anything care?
        matches = [s for s in self.active_subscriptions() if s.matches(event)]
        if not matches:
            return self._record(Decision(IGNORE, event, "nothing is subscribed to it"))

        # 3. ACTIVE GOAL — prefer a subscription tied to a live goal.
        subscription = next((s for s in matches if s.goal), matches[0])

        if subscription.policy == IGNORE:
            return self._record(Decision(IGNORE, event, "the subscription ignores it",
                                         subscription))

        if subscription.policy == NOTIFY:
            return self._record(Decision(NOTIFY, event, "the subscription notifies only",
                                         subscription, subscription.response))

        # 4. POLICY — could the response even run unattended?
        requires_approval = False
        if subscription.tool:
            from pinpoint.security import permissions
            decision = permissions.check(subscription.tool, {})
            if decision.blocked:
                return self._record(Decision(
                    NOTIFY, event,
                    f"the response '{subscription.tool}' is blocked by policy — "
                    f"telling the human instead", subscription, subscription.response))
            requires_approval = decision.requires_approval

        return self._record(Decision(
            ACT, event, "subscribed, permitted, and tied to an active goal",
            subscription, subscription.response, requires_approval))

    def _record(self, decision: Decision) -> Decision:
        self.decisions.append(decision)
        return decision

    def handle(self, event: Event) -> Decision:
        """Decide, then run the subscription's handler when the outcome is ACT."""
        decision = self.decide(event)
        if decision.outcome == ACT and decision.subscription \
                and decision.subscription.handler:
            try:
                decision.subscription.handler(event)
            except Exception as exc:
                decision.reason += f" (handler failed: {exc})"
        return decision

    def stats(self) -> dict:
        counts: Dict[str, int] = {}
        for decision in self.decisions:
            counts[decision.outcome] = counts.get(decision.outcome, 0) + 1
        return {"events": len(self.decisions), "by_outcome": counts,
                "subscriptions": len(self.active_subscriptions())}
