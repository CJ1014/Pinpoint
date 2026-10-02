"""Contacts and recipient resolution.

Resolution never guesses. One match proceeds; several matches produce a
question naming the candidates; no match says so. Sending a message to the
wrong person is not recoverable, which is why ambiguity stops the pipeline
rather than being resolved by confidence score.
"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from pinpoint import jsonstore, paths

RESOLVED = "resolved"
AMBIGUOUS = "ambiguous"
NOT_FOUND = "not_found"

_PHONE = re.compile(r"^\+?[\d][\d\s().-]{6,}$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.I)


@dataclass
class Contact:
    name: str
    phone: str = ""
    email: str = ""
    aliases: List[str] = field(default_factory=list)
    note: str = ""
    source: str = "user"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    added_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict) -> "Contact":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def label(self) -> str:
        """A description that actually distinguishes this contact from a namesake."""
        bits = []
        if self.note:
            bits.append(self.note)
        if self.phone:
            bits.append(f"…{self.phone[-4:]}")
        elif self.email:
            bits.append(self.email)
        return f"{self.name} ({', '.join(bits)})" if bits else self.name

    def address_for(self, kind: str) -> str:
        return self.email if kind == "email" else self.phone


@dataclass
class Resolution:
    """The outcome of looking a recipient up."""
    status: str
    contact: Optional[Contact] = None
    candidates: List[Contact] = field(default_factory=list)
    question: str = ""

    @property
    def ok(self) -> bool:
        return self.status == RESOLVED


def looks_like_address(text: str) -> bool:
    """True when the user gave a literal number or email rather than a name."""
    value = (text or "").strip()
    return bool(_PHONE.match(value) or _EMAIL.match(value))


class ContactBook:
    """The contacts PinPoint is allowed to know about."""

    def __init__(self, path: str = ""):
        self._path = path or paths.state_path("contacts.json")
        self._items: List[Contact] = []
        self._loaded = False

    def load(self) -> "ContactBook":
        if not self._loaded:
            data = jsonstore.read_json(self._path, {"contacts": []})
            self._items = [Contact.from_dict(d) for d in data.get("contacts", [])]
            self._loaded = True
        return self

    def save(self) -> bool:
        return jsonstore.write_json(
            self._path, {"contacts": [c.to_dict() for c in self._items]})

    def add(self, contact: Contact) -> Contact:
        self.load()
        self._items.append(contact)
        self.save()
        return contact

    def all(self) -> List[Contact]:
        self.load()
        return list(self._items)

    def remove(self, contact_id: str) -> bool:
        self.load()
        before = len(self._items)
        self._items = [c for c in self._items if c.id != contact_id]
        if len(self._items) != before:
            self.save()
            return True
        return False

    # ── resolution ───────────────────────────────────────────────────────────

    def _matches(self, name: str) -> List[Contact]:
        needle = (name or "").strip().lower()
        if not needle:
            return []
        exact = [c for c in self.all()
                 if c.name.lower() == needle
                 or needle in [a.lower() for a in c.aliases]]
        if exact:
            return exact
        # First-name match — the common case for "text John".
        first = [c for c in self.all() if c.name.lower().split()[:1] == [needle]]
        if first:
            return first
        return [c for c in self.all() if needle in c.name.lower()]

    def resolve(self, name: str, kind: str = "sms") -> Resolution:
        """Resolve a name to exactly one contact, or ask."""
        if looks_like_address(name):
            return Resolution(RESOLVED, Contact(name=name.strip(),
                                                phone=name.strip() if not _EMAIL.match(name.strip()) else "",
                                                email=name.strip() if _EMAIL.match(name.strip()) else "",
                                                source="literal"))

        matches = self._matches(name)
        if not matches:
            return Resolution(
                NOT_FOUND,
                question=f"I don't have a contact for '{name}'. "
                         f"What's their number or email?")

        usable = [c for c in matches if c.address_for(kind)]
        if kind and not usable:
            channel = "email address" if kind == "email" else "phone number"
            return Resolution(
                NOT_FOUND,
                question=f"I have {matches[0].name} but no {channel} for them. "
                         f"What should I use?")

        pool = usable or matches
        if len(pool) > 1:
            listing = ", ".join(c.label() for c in pool[:4])
            return Resolution(
                AMBIGUOUS, candidates=pool,
                question=f"I found {len(pool)} contacts matching '{name}': "
                         f"{listing}. Which one do you mean?")

        return Resolution(RESOLVED, contact=pool[0])


_default: Optional[ContactBook] = None


def contacts() -> ContactBook:
    """Shared instance, rebuilt when PINPOINT_HOME changes (tests)."""
    global _default
    expected = paths.state_path("contacts.json")
    if _default is None or _default._path != expected:
        _default = ContactBook(expected)
    return _default
