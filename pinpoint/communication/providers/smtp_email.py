"""SMTP email provider.

Environment:

* ``SMTP_HOST``, ``SMTP_PORT`` (default 587)
* ``SMTP_USER``, ``SMTP_PASSWORD``
* ``SMTP_FROM`` (defaults to ``SMTP_USER``)

SMTP has no message id of its own, so the confirmation id is the RFC 5322
``Message-ID`` generated for the send — and it is only reported once the server
has accepted the message with no rejected recipients.
"""

import os
import smtplib
import uuid
from email.message import EmailMessage
from typing import Tuple

from pinpoint.communication.providers import base


class SMTPProvider(base.Provider):
    name = "smtp"
    kind = base.EMAIL

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    @staticmethod
    def _config() -> Tuple[str, int, str, str, str]:
        return (os.environ.get("SMTP_HOST", ""),
                int(os.environ.get("SMTP_PORT", "587") or 587),
                os.environ.get("SMTP_USER", ""),
                os.environ.get("SMTP_PASSWORD", ""),
                os.environ.get("SMTP_FROM") or os.environ.get("SMTP_USER", ""))

    def available(self) -> bool:
        host, _, user, password, sender = self._config()
        return bool(host and user and password and sender)

    def missing_config(self) -> str:
        host, _, user, password, sender = self._config()
        missing = [name for name, value in (
            ("SMTP_HOST", host), ("SMTP_USER", user),
            ("SMTP_PASSWORD", password), ("SMTP_FROM", sender)) if not value]
        return ("smtp needs these environment variables: " + ", ".join(missing)
                if missing else "smtp is configured")

    def send_email(self, to: str, subject: str, body: str) -> base.SendResult:
        if not self.available():
            return base.SendResult(kind=base.EMAIL, provider=self.name, to=to,
                                   error=self.missing_config())
        host, port, user, password, sender = self._config()
        message_id = f"<{uuid.uuid4().hex}@pinpoint.local>"

        message = EmailMessage()
        message["From"] = sender
        message["To"] = to
        message["Subject"] = subject
        message["Message-ID"] = message_id
        message.set_content(body)

        try:
            with smtplib.SMTP(host, port, timeout=self.timeout) as server:
                server.starttls()
                server.login(user, password)
                rejected = server.send_message(message)
        except Exception as exc:
            return base.SendResult(kind=base.EMAIL, provider=self.name, to=to,
                                   error=f"{type(exc).__name__}: {exc}")

        if rejected:
            return base.SendResult(
                kind=base.EMAIL, provider=self.name, to=to,
                error=f"the server rejected {len(rejected)} recipient(s): "
                      f"{list(rejected)[:3]}")
        return base.SendResult(ok=True, kind=base.EMAIL, provider=self.name, to=to,
                               confirmation_id=message_id.strip("<>"),
                               status="accepted")
