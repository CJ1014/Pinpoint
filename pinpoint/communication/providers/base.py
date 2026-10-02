"""Provider interface and the result type every provider must return.

The contract that matters: ``SendResult.ok`` is True only when the provider
handed back an identifier. "The HTTP request didn't raise" is not delivery, and
a provider that cannot produce a confirmation must say so.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Provider kinds.
SMS = "sms"
EMAIL = "email"
VOICE = "voice"


@dataclass
class SendResult:
    """Outcome of one attempt to reach the outside world."""
    ok: bool = False
    provider: str = ""
    kind: str = ""
    confirmation_id: str = ""
    status: str = ""
    to: str = ""
    error: str = ""
    question: str = ""          # set when the send needs the user to decide something
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def needs_clarification(self) -> bool:
        return bool(self.question)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "provider": self.provider, "kind": self.kind,
                "confirmation_id": self.confirmation_id, "status": self.status,
                "to": self.to, "error": self.error, "question": self.question}

    def render(self) -> str:
        """The string a tool returns. Carries the confirmation id when there is one.

        The verification layer looks for that id — which is exactly why an
        unconfirmed send cannot be phrased as a success here.
        """
        if self.ok and self.confirmation_id:
            return (f"{self.kind} accepted by {self.provider}. "
                    f"confirmation_id: {self.confirmation_id} "
                    f"status: {self.status or 'queued'}")
        if self.question:
            return f"Nothing sent — I need to ask first: {self.question}"
        if self.error:
            return f"Error: {self.kind} not sent — {self.error}"
        return (f"Not confirmed: {self.provider or 'provider'} did not return a "
                f"confirmation id. Do not report this as delivered.")


class Provider:
    """Base class. Subclasses implement whichever methods they support."""

    name = "base"
    kind = ""

    def available(self) -> bool:
        """True when this provider is configured well enough to be used."""
        return False

    def missing_config(self) -> str:
        """Human-readable description of what is missing."""
        return f"{self.name} is not configured"

    # Each provider implements what it supports; the rest stay unsupported.
    def send_message(self, to: str, body: str) -> SendResult:
        return SendResult(kind=SMS, provider=self.name, to=to,
                          error=f"{self.name} does not support messaging")

    def send_email(self, to: str, subject: str, body: str) -> SendResult:
        return SendResult(kind=EMAIL, provider=self.name, to=to,
                          error=f"{self.name} does not support email")

    def make_call(self, to: str, script: str) -> SendResult:
        return SendResult(kind=VOICE, provider=self.name, to=to,
                          error=f"{self.name} does not support calling")

    def get_messages(self, limit: int = 10):
        return []

    def get_call_status(self, call_id: str) -> Dict[str, Any]:
        return {"error": f"{self.name} does not support calling"}


def unavailable(kind: str, detail: str) -> SendResult:
    """The result for 'this capability does not exist here'."""
    return SendResult(ok=False, kind=kind, provider="none", error=detail)
