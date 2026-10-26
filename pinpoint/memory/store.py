"""Tiered memory.

Six tiers with different lifetimes and different rules about what earns a
place in them:

* **working**    — the current execution state. Cleared when the session ends.
* **short_term** — the current conversation. Hours.
* **episodic**   — past sessions and significant events. Capped and aged out.
* **procedural** — workflows that actually worked, so they can be reused.
* **long_term**  — stable facts and preferences, and *only* with explicit
  permission to persist.
* **failure**    — delegated to :mod:`pinpoint.memory.failures`.

Nothing is stored just because it happened. Every record carries relevance,
confidence, source, timestamp and a retention policy, and a record that clears
neither the relevance gate nor an explicit permission is simply not kept.
"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from pinpoint import jsonstore, paths

WORKING = "working"
SHORT_TERM = "short_term"
EPISODIC = "episodic"
PROCEDURAL = "procedural"
LONG_TERM = "long_term"
TIERS = (WORKING, SHORT_TERM, EPISODIC, PROCEDURAL, LONG_TERM)


@dataclass
class RetentionPolicy:
    """How long a tier keeps things, and how much it keeps."""
    ttl_days: Optional[float]     # None = no expiry
    max_items: int
    min_relevance: float
    requires_permission: bool = False


POLICIES: Dict[str, RetentionPolicy] = {
    WORKING: RetentionPolicy(ttl_days=1.0, max_items=300, min_relevance=0.0),
    SHORT_TERM: RetentionPolicy(ttl_days=0.5, max_items=200, min_relevance=0.0),
    EPISODIC: RetentionPolicy(ttl_days=90.0, max_items=400, min_relevance=0.35),
    PROCEDURAL: RetentionPolicy(ttl_days=None, max_items=200, min_relevance=0.4),
    LONG_TERM: RetentionPolicy(ttl_days=None, max_items=300, min_relevance=0.5,
                               requires_permission=True),
}

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "your", "you",
    "was", "were", "have", "has", "had", "not", "but", "are", "its", "it's",
    "what", "when", "which", "there", "their", "then", "than", "them", "they",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso() -> str:
    return _now().isoformat()


def _parse(timestamp: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(timestamp)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return _now()


def _keywords(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]{3,}", (text or "").lower())
            if w not in _STOPWORDS}


@dataclass
class MemoryRecord:
    """One remembered thing, with the metadata that decides its fate."""
    content: str
    tier: str = WORKING
    category: str = "general"
    relevance: float = 0.5
    confidence: float = 0.7
    source: str = "agent"
    tags: List[str] = field(default_factory=list)
    session: str = ""
    ttl_days: Optional[float] = None
    access_count: int = 0
    last_accessed: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    timestamp: str = field(default_factory=_iso)

    def age_days(self) -> float:
        return (_now() - _parse(self.timestamp)).total_seconds() / 86400.0

    def expired(self) -> bool:
        ttl = self.ttl_days if self.ttl_days is not None else \
            POLICIES[self.tier].ttl_days if self.tier in POLICIES else None
        return ttl is not None and self.age_days() > ttl

    def score(self, query: str = "") -> float:
        """Ranking: relevance × confidence, decayed by age, boosted by overlap."""
        base = self.relevance * self.confidence
        decay = 1.0 / (1.0 + self.age_days() / 30.0)
        overlap = 0.0
        if query:
            terms = _keywords(query)
            mine = _keywords(self.content) | {t.lower() for t in self.tags}
            if terms and mine:
                overlap = len(terms & mine) / len(terms)
        usage = min(0.2, self.access_count * 0.02)
        return base * (0.5 + 0.5 * decay) + overlap + usage

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class MemoryStore:
    """The tiered store. Persists to its own file; v2's memory.json is untouched."""

    def __init__(self, path: str = ""):
        self._path = path or paths.output_dir("memory_tiers.json")
        self._records: List[MemoryRecord] = []
        self._loaded = False

    # ── persistence ──────────────────────────────────────────────────────────

    def load(self) -> "MemoryStore":
        if not self._loaded:
            data = jsonstore.read_json(self._path, {"records": []})
            self._records = [MemoryRecord.from_dict(d) for d in data.get("records", [])]
            self._loaded = True
        return self

    def save(self) -> bool:
        return jsonstore.write_json(self._path, {
            "records": [r.to_dict() for r in self._records],
            "saved_at": _iso(),
        })

    # ── writing ──────────────────────────────────────────────────────────────

    def remember(self, content: str, *, tier: str = WORKING, category: str = "general",
                 relevance: float = 0.5, confidence: float = 0.7,
                 source: str = "agent", tags: Optional[List[str]] = None,
                 session: str = "", ttl_days: Optional[float] = None,
                 permitted: bool = False) -> Optional[MemoryRecord]:
        """Store something, if it earns its place. Returns None when refused."""
        self.load()
        if tier not in POLICIES:
            tier = WORKING
        policy = POLICIES[tier]

        if policy.requires_permission and not permitted:
            return None
        if relevance < policy.min_relevance:
            return None
        if not (content or "").strip():
            return None

        record = MemoryRecord(
            content=content.strip(), tier=tier, category=category,
            relevance=max(0.0, min(1.0, relevance)),
            confidence=max(0.0, min(1.0, confidence)),
            source=source, tags=list(tags or []), session=session, ttl_days=ttl_days,
        )
        self._records.append(record)
        self._enforce_capacity(tier)
        self.save()
        return record

    def _enforce_capacity(self, tier: str) -> None:
        """Keep the most valuable records when a tier overflows."""
        policy = POLICIES[tier]
        in_tier = [r for r in self._records if r.tier == tier]
        if len(in_tier) <= policy.max_items:
            return
        keep = sorted(in_tier, key=lambda r: -r.score())[:policy.max_items]
        keep_ids = {r.id for r in keep}
        self._records = [r for r in self._records
                         if r.tier != tier or r.id in keep_ids]

    # ── reading ──────────────────────────────────────────────────────────────

    def recall(self, query: str = "", *, tier: str = "", category: str = "",
               limit: int = 10, min_relevance: float = 0.0,
               touch: bool = True) -> List[MemoryRecord]:
        """Ranked recall. Accessing a record makes it slightly stickier."""
        self.load()
        candidates = [r for r in self._records if not r.expired()]
        if tier:
            candidates = [r for r in candidates if r.tier == tier]
        if category:
            candidates = [r for r in candidates if r.category == category]
        if min_relevance:
            candidates = [r for r in candidates if r.relevance >= min_relevance]

        ranked = sorted(candidates, key=lambda r: -r.score(query))[:limit]
        if touch and ranked:
            for record in ranked:
                record.access_count += 1
                record.last_accessed = _iso()
            self.save()
        return ranked

    def get(self, record_id: str) -> Optional[MemoryRecord]:
        self.load()
        return next((r for r in self._records if r.id == record_id), None)

    def all(self, tier: str = "") -> List[MemoryRecord]:
        self.load()
        return [r for r in self._records if not tier or r.tier == tier]

    # ── lifecycle ────────────────────────────────────────────────────────────

    def forget(self, record_id: str) -> bool:
        self.load()
        before = len(self._records)
        self._records = [r for r in self._records if r.id != record_id]
        if len(self._records) != before:
            self.save()
            return True
        return False

    def clear_working(self) -> int:
        """Drop working memory. Called when a session ends."""
        self.load()
        before = len(self._records)
        self._records = [r for r in self._records if r.tier != WORKING]
        self.save()
        return before - len(self._records)

    def prune(self) -> dict:
        """Apply retention policy: drop expired records, enforce capacity."""
        self.load()
        before = len(self._records)
        self._records = [r for r in self._records if not r.expired()]
        expired = before - len(self._records)
        for tier in TIERS:
            self._enforce_capacity(tier)
        self.save()
        return {"expired": expired, "remaining": len(self._records)}

    def promote(self, record_id: str, tier: str, *, permitted: bool = False,
                relevance: Optional[float] = None) -> Optional[MemoryRecord]:
        """Move a record up a tier. long_term still requires explicit permission."""
        record = self.get(record_id)
        if record is None or tier not in POLICIES:
            return None
        policy = POLICIES[tier]
        if policy.requires_permission and not permitted:
            return None
        if relevance is not None:
            record.relevance = max(0.0, min(1.0, relevance))
        if record.relevance < policy.min_relevance:
            return None
        record.tier = tier
        record.ttl_days = None
        self.save()
        return record

    def consolidate(self, session: str = "", summary: str = "",
                    keep_top: int = 5) -> List[MemoryRecord]:
        """End-of-session sweep: the best of working memory becomes episodic.

        Everything else in working memory is discarded — that is the point of
        having a working tier at all.
        """
        self.load()
        working = [r for r in self._records if r.tier == WORKING]
        promoted: List[MemoryRecord] = []
        for record in sorted(working, key=lambda r: -r.score())[:keep_top]:
            if record.relevance >= POLICIES[EPISODIC].min_relevance:
                record.tier = EPISODIC
                record.ttl_days = None
                promoted.append(record)
        if summary:
            episode = self.remember(summary, tier=EPISODIC, category="session",
                                    relevance=0.7, confidence=0.9,
                                    source="session", session=session)
            if episode:
                promoted.append(episode)
        self.clear_working()
        return promoted

    # ── procedural memory ────────────────────────────────────────────────────

    def record_procedure(self, name: str, steps: List[str], *, goal_kind: str = "",
                         outcome: str = "worked", confidence: float = 0.7
                         ) -> Optional[MemoryRecord]:
        """Store a workflow that worked, so it can be reused."""
        content = f"{name}: " + " → ".join(steps)
        return self.remember(content, tier=PROCEDURAL, category="procedure",
                             relevance=0.7 if outcome == "worked" else 0.4,
                             confidence=confidence, source="experience",
                             tags=[goal_kind, name] if goal_kind else [name])

    def procedures_for(self, goal: str, limit: int = 3) -> List[MemoryRecord]:
        return self.recall(goal, tier=PROCEDURAL, limit=limit, touch=False)

    # ── reporting ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        self.load()
        by_tier = {tier: 0 for tier in TIERS}
        for record in self._records:
            by_tier[record.tier] = by_tier.get(record.tier, 0) + 1
        return {
            "total": len(self._records),
            "by_tier": by_tier,
            "expired_pending": sum(1 for r in self._records if r.expired()),
        }

    def context_block(self, query: str, limit: int = 6) -> str:
        """Memory formatted for injection into a prompt, with provenance."""
        records = self.recall(query, limit=limit, touch=False)
        if not records:
            return ""
        lines = ["What I know that may bear on this:"]
        for record in records:
            lines.append(f"  - [{record.tier}/{record.source}, "
                         f"confidence {record.confidence:.0%}] {record.content[:180]}")
        return "\n".join(lines)


_default: Optional[MemoryStore] = None


def store() -> MemoryStore:
    """Shared instance, rebuilt when PINPOINT_HOME changes (tests)."""
    global _default
    expected = paths.output_dir("memory_tiers.json")
    if _default is None or _default._path != expected:
        _default = MemoryStore(expected)
    return _default
