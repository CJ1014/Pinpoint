"""Provider-agnostic communication manager.

Selection is by environment variable, so which provider is in use is a
deployment decision rather than something baked into the agent:

* ``PINPOINT_SMS_PROVIDER=twilio``
* ``PINPOINT_EMAIL_PROVIDER=smtp``
* ``PINPOINT_VOICE_PROVIDER=twilio``

Unset means the capability does not exist, and every call returns that plainly.
There is no simulation mode: a fake success here would be indistinguishable
from a real one to everything downstream, which is exactly the failure this
architecture exists to prevent.

Outbound calls always identify themselves as an automated assistant calling on
someone's behalf. That disclosure cannot be turned off.
"""

import os
from typing import Any, Dict, List, Optional

from pinpoint.communication import contacts as C
from pinpoint.communication.providers import base

_PROVIDER_ENV = {
    base.SMS: "PINPOINT_SMS_PROVIDER",
    base.EMAIL: "PINPOINT_EMAIL_PROVIDER",
    base.VOICE: "PINPOINT_VOICE_PROVIDER",
}

# Test seams — set to override provider selection.
_overrides: Dict[str, base.Provider] = {}


def set_provider(kind: str, provider: Optional[base.Provider]) -> None:
    """Install a provider directly, bypassing environment selection."""
    if provider is None:
        _overrides.pop(kind, None)
    else:
        _overrides[kind] = provider


def reset_providers() -> None:
    _overrides.clear()


def _build(kind: str) -> Optional[base.Provider]:
    name = (os.environ.get(_PROVIDER_ENV.get(kind, ""), "") or "").strip().lower()
    if not name or name == "none":
        return None
    if name == "twilio" and kind in (base.SMS, base.VOICE):
        from pinpoint.communication.providers.twilio import TwilioProvider
        return TwilioProvider(kind=kind)
    if name == "smtp" and kind == base.EMAIL:
        from pinpoint.communication.providers.smtp_email import SMTPProvider
        return SMTPProvider()
    return None


def get_provider(kind: str) -> Optional[base.Provider]:
    if kind in _overrides:
        return _overrides[kind]
    return _build(kind)


def provider_available(kind: str) -> bool:
    provider = get_provider(kind)
    return bool(provider and provider.available())


def _finalize(result: base.SendResult, address: str) -> base.SendResult:
    """Enforce the invariant no provider is trusted to enforce for itself.

    A provider that claims success without handing back an identifier has not
    confirmed anything, so the manager downgrades it rather than letting an
    unconfirmed send reach the agent looking delivered.
    """
    result.to = address
    if result.ok and not result.confirmation_id:
        result.ok = False
        result.error = (result.error
                        or f"{result.provider or 'the provider'} reported success but "
                           f"returned no confirmation id — treating it as unconfirmed")
    return result


def _unavailable(kind: str) -> base.SendResult:
    provider = get_provider(kind)
    if provider is None:
        env = _PROVIDER_ENV.get(kind, "")
        return base.unavailable(
            kind, f"no {kind} provider is configured — set {env} and its credentials")
    return base.unavailable(kind, provider.missing_config())


def status() -> Dict[str, dict]:
    """What can actually be done right now, and what is missing if not."""
    out = {}
    for kind in (base.SMS, base.EMAIL, base.VOICE):
        provider = get_provider(kind)
        out[kind] = {
            "provider": provider.name if provider else None,
            "available": bool(provider and provider.available()),
            "detail": provider.missing_config() if provider else
                      f"{_PROVIDER_ENV[kind]} is not set",
        }
    return out


# ── Recipient handling ────────────────────────────────────────────────────────

def resolve(name: str, kind: str = base.SMS) -> C.Resolution:
    return C.contacts().resolve(name, kind="email" if kind == base.EMAIL else "sms")


def _address(name: str, kind: str) -> tuple:
    """Return ``(address, clarification_result_or_None)``."""
    resolution = resolve(name, kind)
    if not resolution.ok:
        return "", base.SendResult(kind=kind, provider="", to=name,
                                   question=resolution.question)
    contact = resolution.contact
    address = contact.address_for("email" if kind == base.EMAIL else "sms")
    if not address:
        channel = "email address" if kind == base.EMAIL else "phone number"
        return "", base.SendResult(
            kind=kind, provider="", to=name,
            question=f"I have {contact.name} but no {channel} for them. "
                     f"What should I use?")
    return address, None


# ── Operations ────────────────────────────────────────────────────────────────

def send_message(to: str, body: str) -> base.SendResult:
    """Send a text. Resolves the recipient first, and asks rather than guessing."""
    if not (body or "").strip():
        return base.SendResult(kind=base.SMS, to=to,
                               question="What exactly do you want the message to say?")

    address, clarification = _address(to, base.SMS)
    if clarification is not None:
        return clarification

    if not provider_available(base.SMS):
        result = _unavailable(base.SMS)
        result.to = address
        return result

    return _finalize(get_provider(base.SMS).send_message(address, body), address)


def send_email(to: str, subject: str, body: str) -> base.SendResult:
    address, clarification = _address(to, base.EMAIL)
    if clarification is not None:
        return clarification

    if not provider_available(base.EMAIL):
        result = _unavailable(base.EMAIL)
        result.to = address
        return result

    return _finalize(get_provider(base.EMAIL).send_email(address, subject, body),
                     address)


def call_intro(on_behalf_of: str = "", purpose: str = "") -> str:
    """The disclosure every outbound call opens with."""
    who = on_behalf_of or "the person who asked me to call"
    line = f"Hi — this is an automated assistant calling on behalf of {who}."
    if purpose:
        line += f" {purpose.rstrip('.')}."
    return line


def make_call(to: str, purpose: str = "", script: str = "",
              on_behalf_of: str = "", disclose: bool = True) -> base.SendResult:
    """Place a call. The automated-assistant disclosure is not optional."""
    if not disclose:
        return base.SendResult(
            kind=base.VOICE, to=to,
            error="outbound calls must identify themselves as an automated "
                  "assistant — refusing to place a call that pretends otherwise")

    address, clarification = _address(to, base.VOICE)
    if clarification is not None:
        return clarification

    if not provider_available(base.VOICE):
        result = _unavailable(base.VOICE)
        result.to = address
        return result

    full_script = call_intro(on_behalf_of, purpose)
    if script:
        full_script += " " + script
    return _finalize(get_provider(base.VOICE).make_call(address, full_script), address)


def get_messages(limit: int = 10) -> List[dict]:
    if not provider_available(base.SMS):
        return []
    return get_provider(base.SMS).get_messages(limit)


def get_call_status(call_id: str) -> Dict[str, Any]:
    if not provider_available(base.VOICE):
        return {"error": _unavailable(base.VOICE).error}
    return get_provider(base.VOICE).get_call_status(call_id)
