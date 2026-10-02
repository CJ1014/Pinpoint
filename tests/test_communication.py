"""Stage 8: contacts, providers, and the communication manager.

Two things must hold no matter what: an ambiguous recipient stops the send and
asks, and nothing is ever reported as delivered without a provider
confirmation.
"""

import pytest

from pinpoint.communication import contacts as C, manager as M
from pinpoint.communication.providers import base


class FakeProvider(base.Provider):
    """Stands in for a real provider. Never pretends on its own."""

    name = "fake"

    def __init__(self, kind=base.SMS, ok=True, confirm="SM123456", error="",
                 configured=True):
        self.kind = kind
        self._ok = ok
        self._confirm = confirm
        self._error = error
        self._configured = configured
        self.sent = []

    def available(self):
        return self._configured

    def missing_config(self):
        return "fake provider is not configured"

    def _result(self, to):
        if not self._ok:
            return base.SendResult(kind=self.kind, provider=self.name, to=to,
                                   error=self._error or "provider rejected it")
        return base.SendResult(ok=True, kind=self.kind, provider=self.name, to=to,
                               confirmation_id=self._confirm, status="queued")

    def send_message(self, to, body):
        self.sent.append((to, body))
        return self._result(to)

    def send_email(self, to, subject, body):
        self.sent.append((to, subject, body))
        return self._result(to)

    def make_call(self, to, script):
        self.sent.append((to, script))
        return self._result(to)

    def get_call_status(self, call_id):
        return {"id": call_id, "status": "completed", "duration": "42"}


@pytest.fixture(autouse=True)
def clean_providers():
    M.reset_providers()
    yield
    M.reset_providers()


@pytest.fixture
def book():
    store = C.ContactBook()
    store.add(C.Contact(name="Sarah Kim", phone="+15550001111",
                        email="sarah@example.com"))
    return store


# ── Contact resolution ────────────────────────────────────────────────────────

def test_single_match_resolves(book):
    assert book.resolve("Sarah").ok


def test_two_johns_are_ambiguous(book):
    book.add(C.Contact(name="John Smith", phone="+15550002222", note="school"))
    book.add(C.Contact(name="John Doe", phone="+15550003333", note="football"))
    resolution = book.resolve("John")
    assert resolution.status == C.AMBIGUOUS and len(resolution.candidates) == 2


def test_ambiguity_asks_a_distinguishing_question(book):
    book.add(C.Contact(name="John Smith", phone="+15550002222", note="school"))
    book.add(C.Contact(name="John Doe", phone="+15550003333", note="football"))
    question = book.resolve("John").question
    assert "school" in question and "football" in question


def test_unknown_contact_is_not_found(book):
    resolution = book.resolve("Mallory")
    assert resolution.status == C.NOT_FOUND and "don't have a contact" in resolution.question


def test_literal_phone_number_resolves_without_a_contact(book):
    assert book.resolve("+15559998888").ok


def test_literal_email_resolves_for_email(book):
    assert book.resolve("someone@example.com", kind="email").ok


def test_alias_matches(book):
    book.add(C.Contact(name="Robert Jones", phone="+15550004444", aliases=["Bob"]))
    assert book.resolve("Bob").contact.name == "Robert Jones"


def test_contact_without_the_needed_channel_asks(book):
    book.add(C.Contact(name="Pat Lee", phone="+15550005555"))   # no email
    resolution = book.resolve("Pat Lee", kind="email")
    assert resolution.status == C.NOT_FOUND and "email address" in resolution.question


def test_contacts_persist(book):
    assert C.ContactBook().all()


# ── Provider availability ─────────────────────────────────────────────────────

def test_no_provider_means_unavailable(monkeypatch):
    monkeypatch.delenv("PINPOINT_SMS_PROVIDER", raising=False)
    assert M.provider_available(base.SMS) is False


def test_unavailable_send_says_what_is_missing(book, monkeypatch):
    monkeypatch.delenv("PINPOINT_SMS_PROVIDER", raising=False)
    result = M.send_message("Sarah", "hello")
    assert result.ok is False and "PINPOINT_SMS_PROVIDER" in result.error


def test_configured_but_credential_less_provider_explains_itself(book):
    M.set_provider(base.SMS, FakeProvider(configured=False))
    result = M.send_message("Sarah", "hello")
    assert result.ok is False and "not configured" in result.error


def test_status_reports_each_channel(monkeypatch):
    monkeypatch.delenv("PINPOINT_SMS_PROVIDER", raising=False)
    monkeypatch.delenv("PINPOINT_VOICE_PROVIDER", raising=False)
    report = M.status()
    assert set(report) == {base.SMS, base.EMAIL, base.VOICE}
    assert report[base.SMS]["available"] is False


def test_twilio_reports_the_variables_it_needs(monkeypatch):
    from pinpoint.communication.providers.twilio import TwilioProvider
    for name in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER"):
        monkeypatch.delenv(name, raising=False)
    provider = TwilioProvider()
    assert provider.available() is False
    assert "TWILIO_ACCOUNT_SID" in provider.missing_config()


def test_smtp_reports_the_variables_it_needs(monkeypatch):
    from pinpoint.communication.providers.smtp_email import SMTPProvider
    for name in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM"):
        monkeypatch.delenv(name, raising=False)
    assert "SMTP_HOST" in SMTPProvider().missing_config()


# ── Sending ───────────────────────────────────────────────────────────────────

def test_successful_send_carries_a_confirmation(book):
    M.set_provider(base.SMS, FakeProvider())
    result = M.send_message("Sarah", "running late")
    assert result.ok and result.confirmation_id == "SM123456"


def test_rendered_result_exposes_the_confirmation_id(book):
    M.set_provider(base.SMS, FakeProvider())
    assert "confirmation_id: SM123456" in M.send_message("Sarah", "hi").render()


def test_provider_rejection_is_not_a_success(book):
    M.set_provider(base.SMS, FakeProvider(ok=False, error="21610 unsubscribed"))
    result = M.send_message("Sarah", "hi")
    assert result.ok is False and "21610" in result.error


def test_provider_claiming_success_without_an_id_is_downgraded(book):
    """A provider is not trusted to certify its own delivery."""
    M.set_provider(base.SMS, FakeProvider(confirm=""))
    result = M.send_message("Sarah", "hi")
    assert result.ok is False and "no confirmation id" in result.error


def test_unconfirmed_result_never_renders_as_delivered():
    unconfirmed = base.SendResult(ok=False, kind=base.SMS, provider="fake")
    assert "Do not report this as delivered" in unconfirmed.render()


def test_ambiguous_recipient_blocks_the_send(book):
    book.add(C.Contact(name="John Smith", phone="+1555000222", note="school"))
    book.add(C.Contact(name="John Doe", phone="+1555000333", note="football"))
    provider = FakeProvider()
    M.set_provider(base.SMS, provider)
    result = M.send_message("John", "running late")
    assert result.needs_clarification and provider.sent == []


def test_unknown_recipient_blocks_the_send(book):
    provider = FakeProvider()
    M.set_provider(base.SMS, provider)
    result = M.send_message("Mallory", "hi")
    assert result.needs_clarification and provider.sent == []


def test_empty_body_asks_what_to_say(book):
    M.set_provider(base.SMS, FakeProvider())
    assert M.send_message("Sarah", "   ").needs_clarification


def test_clarification_renders_as_a_question(book):
    M.set_provider(base.SMS, FakeProvider())
    assert "I need to ask first" in M.send_message("Mallory", "hi").render()


def test_email_send(book):
    M.set_provider(base.EMAIL, FakeProvider(kind=base.EMAIL, confirm="mid-1"))
    assert M.send_email("Sarah", "subject", "body").ok


# ── Calling ───────────────────────────────────────────────────────────────────

def test_call_places_through_the_provider(book):
    provider = FakeProvider(kind=base.VOICE, confirm="CA999")
    M.set_provider(base.VOICE, provider)
    result = M.make_call("Sarah", purpose="ask if she's still coming",
                         on_behalf_of="CJ")
    assert result.ok and result.confirmation_id == "CA999"


def test_call_always_discloses_it_is_automated(book):
    provider = FakeProvider(kind=base.VOICE)
    M.set_provider(base.VOICE, provider)
    M.make_call("Sarah", purpose="ask about tonight", on_behalf_of="CJ")
    script = provider.sent[0][1]
    assert "automated assistant" in script and "CJ" in script


def test_call_that_hides_being_automated_is_refused(book):
    provider = FakeProvider(kind=base.VOICE)
    M.set_provider(base.VOICE, provider)
    result = M.make_call("Sarah", on_behalf_of="CJ", disclose=False)
    assert result.ok is False and provider.sent == []
    assert "refusing" in result.error


def test_call_without_a_provider_does_not_pretend(book, monkeypatch):
    monkeypatch.delenv("PINPOINT_VOICE_PROVIDER", raising=False)
    result = M.make_call("Sarah", purpose="check in")
    assert result.ok is False and "no voice provider" in result.error


def test_call_status_passes_through(book):
    M.set_provider(base.VOICE, FakeProvider(kind=base.VOICE))
    assert M.get_call_status("CA999")["status"] == "completed"


def test_call_status_without_provider_reports_the_gap(monkeypatch):
    monkeypatch.delenv("PINPOINT_VOICE_PROVIDER", raising=False)
    assert "error" in M.get_call_status("CA999")


def test_ambiguous_recipient_blocks_a_call(book):
    book.add(C.Contact(name="John Smith", phone="+1555000222", note="school"))
    book.add(C.Contact(name="John Doe", phone="+1555000333", note="football"))
    provider = FakeProvider(kind=base.VOICE)
    M.set_provider(base.VOICE, provider)
    assert M.make_call("John").needs_clarification and provider.sent == []


# ── Credential hygiene ────────────────────────────────────────────────────────

def test_no_credentials_are_written_to_disk(book, monkeypatch, isolated_home):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "supersecrettoken12345")
    M.set_provider(base.SMS, FakeProvider())
    M.send_message("Sarah", "hi")
    for path in isolated_home.rglob("*"):
        if path.is_file():
            assert "supersecrettoken12345" not in path.read_text(errors="ignore")


def test_registry_marks_communication_as_external_and_billable():
    from pinpoint.tools import registry
    assert registry.get("send_message").external_side_effect
    assert registry.get("make_call").costs_money
