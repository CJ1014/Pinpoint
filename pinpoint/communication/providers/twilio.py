"""Twilio provider for SMS and voice.

Credentials are read from the environment at call time:

* ``TWILIO_ACCOUNT_SID``
* ``TWILIO_AUTH_TOKEN``
* ``TWILIO_FROM_NUMBER``
* ``PINPOINT_VOICE_CALLBACK_URL`` (voice only — the TwiML endpoint Twilio
  fetches to drive the call)

Nothing is persisted. A send is reported as successful only when Twilio
returns a ``sid``.
"""

import os
from typing import Any, Dict, List

from pinpoint.communication.providers import base

API_ROOT = "https://api.twilio.com/2010-04-01"


class TwilioProvider(base.Provider):
    name = "twilio"

    def __init__(self, kind: str = base.SMS, timeout: float = 30.0):
        self.kind = kind
        self.timeout = timeout

    # ── configuration ────────────────────────────────────────────────────────

    @staticmethod
    def _credentials() -> tuple:
        return (os.environ.get("TWILIO_ACCOUNT_SID", ""),
                os.environ.get("TWILIO_AUTH_TOKEN", ""),
                os.environ.get("TWILIO_FROM_NUMBER", ""))

    def available(self) -> bool:
        sid, token, from_number = self._credentials()
        if not (sid and token and from_number):
            return False
        if self.kind == base.VOICE:
            return bool(os.environ.get("PINPOINT_VOICE_CALLBACK_URL"))
        return True

    def missing_config(self) -> str:
        sid, token, from_number = self._credentials()
        missing = [name for name, value in (
            ("TWILIO_ACCOUNT_SID", sid), ("TWILIO_AUTH_TOKEN", token),
            ("TWILIO_FROM_NUMBER", from_number)) if not value]
        if self.kind == base.VOICE and not os.environ.get("PINPOINT_VOICE_CALLBACK_URL"):
            missing.append("PINPOINT_VOICE_CALLBACK_URL")
        return ("twilio needs these environment variables: " + ", ".join(missing)
                if missing else "twilio is configured")

    # ── requests ─────────────────────────────────────────────────────────────

    def _post(self, resource: str, payload: Dict[str, str]) -> Dict[str, Any]:
        import httpx
        sid, token, _ = self._credentials()
        response = httpx.post(f"{API_ROOT}/Accounts/{sid}/{resource}",
                              data=payload, auth=(sid, token), timeout=self.timeout)
        try:
            body = response.json()
        except Exception:
            body = {"raw": response.text[:400]}
        body["_http_status"] = response.status_code
        return body

    def _get(self, resource: str, params: Dict[str, str] = None) -> Dict[str, Any]:
        import httpx
        sid, token, _ = self._credentials()
        response = httpx.get(f"{API_ROOT}/Accounts/{sid}/{resource}",
                             params=params or {}, auth=(sid, token),
                             timeout=self.timeout)
        try:
            body = response.json()
        except Exception:
            body = {"raw": response.text[:400]}
        body["_http_status"] = response.status_code
        return body

    # ── operations ───────────────────────────────────────────────────────────

    def send_message(self, to: str, body: str) -> base.SendResult:
        if not self.available():
            return base.SendResult(kind=base.SMS, provider=self.name, to=to,
                                   error=self.missing_config())
        _, _, from_number = self._credentials()
        try:
            data = self._post("Messages.json",
                              {"To": to, "From": from_number, "Body": body})
        except Exception as exc:
            return base.SendResult(kind=base.SMS, provider=self.name, to=to,
                                   error=f"{type(exc).__name__}: {exc}")

        confirmation = data.get("sid", "")
        if not confirmation:
            return base.SendResult(
                kind=base.SMS, provider=self.name, to=to,
                error=data.get("message", "provider returned no message sid"),
                raw=data)
        return base.SendResult(ok=True, kind=base.SMS, provider=self.name, to=to,
                               confirmation_id=confirmation,
                               status=data.get("status", "queued"), raw=data)

    def make_call(self, to: str, script: str) -> base.SendResult:
        if not self.available():
            return base.SendResult(kind=base.VOICE, provider=self.name, to=to,
                                   error=self.missing_config())
        _, _, from_number = self._credentials()
        callback = os.environ.get("PINPOINT_VOICE_CALLBACK_URL", "")
        try:
            data = self._post("Calls.json",
                              {"To": to, "From": from_number, "Url": callback})
        except Exception as exc:
            return base.SendResult(kind=base.VOICE, provider=self.name, to=to,
                                   error=f"{type(exc).__name__}: {exc}")

        confirmation = data.get("sid", "")
        if not confirmation:
            return base.SendResult(
                kind=base.VOICE, provider=self.name, to=to,
                error=data.get("message", "provider returned no call sid"), raw=data)
        return base.SendResult(ok=True, kind=base.VOICE, provider=self.name, to=to,
                               confirmation_id=confirmation,
                               status=data.get("status", "initiated"), raw=data)

    def get_messages(self, limit: int = 10) -> List[dict]:
        if not self.available():
            return []
        try:
            data = self._get("Messages.json", {"PageSize": str(limit)})
        except Exception:
            return []
        return [
            {"from": m.get("from"), "to": m.get("to"), "body": m.get("body"),
             "status": m.get("status"), "sent_at": m.get("date_sent"),
             "id": m.get("sid")}
            for m in data.get("messages", [])[:limit]
        ]

    def get_call_status(self, call_id: str) -> Dict[str, Any]:
        if not self.available():
            return {"error": self.missing_config()}
        try:
            data = self._get(f"Calls/{call_id}.json")
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        if "status" not in data:
            return {"error": data.get("message", "no status returned"), "id": call_id}
        return {"id": call_id, "status": data.get("status"),
                "duration": data.get("duration"), "to": data.get("to"),
                "ended_at": data.get("end_time")}
