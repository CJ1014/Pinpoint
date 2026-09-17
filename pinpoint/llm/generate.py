"""Expand the persona seed set into a full training corpus using Claude.

``persona.seed_turns()`` returns 9 hand-written examples. Fine-tuning needs
thousands. This module uses those seeds as few-shot exemplars and asks Claude to
produce more, against a scenario grid that forces diversity rather than hoping
for it.

DESIGN NOTES

*Diversity comes from the grid, not the prompt.* Asking "generate 5000 more like
these" yields 5000 paraphrases of the same handful of situations. Instead each
request targets one cell of domain x situation x behaviour, so the spread is
structural. `SCENARIO_GRID` is the lever to pull if the output feels samey.

*Grounding is a scenario dimension, not a style.* Training a small model to
write "I verified X" does not make it verify anything — that is the distillation
failure mode for this behaviour class, and it is the one that matters most here.
Roughly half the generated scenarios put real tool output in the prompt and
require the response to reason from it; the rest withhold evidence and require
the response to decline the claim. The model learns to key its confidence to
what is actually in context.

*Filtering is not optional.* `validate()` rejects invented confidence
percentages, near-duplicates, malformed structure and responses that assert
verification with no evidence present. Expect to discard 10-25%. Generated data
that skips this step teaches exactly the habits the seeds were written to avoid.

COST
Bulk generation defaults to the Batch API (50% cheaper, results within 24h).
``estimate_cost()`` prices a run before it starts; the CLI refuses to spend
without explicit confirmation.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Iterator, List, Optional, Sequence, Tuple
import hashlib
import json
import os
import random
import re
import time

from .persona import (
    ACCOUNTABILITY,
    FRAMINGS,
    SURVIVAL,
    Turn,
    seed_turns,
)

MODEL = "claude-opus-5"

# Priced per 1M tokens. Batch runs bill at 50%.
PRICE_IN, PRICE_OUT = 5.00, 25.00


# ---------------------------------------------------------------------------
# Scenario grid — the source of diversity
# ---------------------------------------------------------------------------

DOMAINS = [
    "a Python web backend (Flask/FastAPI)",
    "a data pipeline moving CSV/Parquet between systems",
    "a CLI tool with argument parsing and subcommands",
    "infrastructure and deployment (Docker, systemd, cron)",
    "a machine learning training script",
    "embedded work on a Raspberry Pi with attached hardware",
    "a database schema and its migrations",
    "a frontend build (bundler, assets, static hosting)",
    "a test suite and its fixtures",
    "system security (permissions, secrets, network rules)",
    "log analysis and debugging a production incident",
    "a scheduled job that runs unattended overnight",
]

# Each situation says what the user's turn looks like and whether the prompt
# carries evidence the response is expected to reason from.
@dataclass(frozen=True)
class Situation:
    key: str
    description: str
    evidence: bool          # does the prompt include real tool output?


SITUATIONS = [
    Situation("fresh_task", "CJ asks for something new to be built or changed", False),
    Situation("ambiguous_request",
              "CJ asks for something underspecified, where guessing the "
              "interpretation would waste work", False),
    Situation("report_result",
              "CJ asks whether something worked; the prompt includes the actual "
              "command output, exit codes or logs", True),
    Situation("conflicting_evidence",
              "the prompt includes tool output where signals disagree — a green "
              "check alongside something that contradicts it", True),
    Situation("missing_capability",
              "CJ asks for something the agent cannot do — no provider "
              "configured, no network, no credentials, tool absent", False),
    Situation("prior_failure",
              "CJ points out a mistake the agent has now made more than once", False),
    Situation("partial_completion",
              "the prompt includes output showing some of the work succeeded and "
              "some did not", True),
    Situation("stale_assumption",
              "the prompt includes evidence that something the agent previously "
              "believed is no longer true", True),
]

# Not every behaviour fits every situation. Mapping these explicitly keeps the
# generator from asking for incoherent combinations.
BEHAVIOUR_SITUATIONS: Dict[str, List[str]] = {
    "decomposition": ["fresh_task", "ambiguous_request"],
    "verification": ["report_result", "conflicting_evidence", "partial_completion"],
    "uncertainty": ["report_result", "ambiguous_request", "conflicting_evidence",
                    "stale_assumption"],
    "refuse_fabrication": ["missing_capability", "report_result", "partial_completion"],
    "learn_from_failure": ["prior_failure", "partial_completion"],
}

BEHAVIOUR_GUIDANCE = {
    "decomposition": (
        "The response breaks the goal into steps that can each be checked "
        "independently, and names which step is the one that actually proves the "
        "thing works. It does not pretend to know facts it has not looked up."
    ),
    "verification": (
        "The response distinguishes what was confirmed from what was merely "
        "observed to not error. It states the specific evidence, and names what "
        "remains unchecked."
    ),
    "uncertainty": (
        "The response separates what it is confident about from what it is not, "
        "and says why. It never converts a guess into a number."
    ),
    "refuse_fabrication": (
        "The response declines to invent a result. It says plainly what it did "
        "not do or cannot do, explains why a fabricated answer would be worse "
        "than none, and offers a concrete alternative."
    ),
    "learn_from_failure": (
        "The response owns the specific mechanism of the repeated mistake — not "
        "a vague apology — and names a check that would have caught it."
    ),
}


# A fourth axis. Without it the grid is only 168 cells, which cannot fill a
# thousand requests — and repeating a cell with an identical prompt yields
# paraphrases, not new scenarios. Each angle changes what the situation is
# actually about, not just its wording.
ANGLES = [
    "the problem appears only intermittently",
    "the code was written by someone else and is being inherited",
    "CJ is under time pressure and wants it now",
    "an earlier attempt at this already failed once",
    "the change touches a component with no test coverage",
    "it works locally but not in the target environment",
    "this request conflicts with something CJ asked for earlier",
    "the relevant system is one PinPoint has not touched before",
]


@dataclass(frozen=True)
class Cell:
    """One point in the scenario grid."""

    domain: str
    situation: Situation
    behaviour: str
    angle: str

    @property
    def key(self) -> str:
        h = hashlib.sha1(f"{self.domain}|{self.angle}".encode()).hexdigest()[:6]
        return f"{self.behaviour}:{self.situation.key}:{h}"


def grid_cells(shuffle: bool = True, seed: int = 1337) -> List[Cell]:
    """Every coherent (domain, situation, behaviour, angle) combination."""
    cells = []
    for behaviour, sit_keys in BEHAVIOUR_SITUATIONS.items():
        for sit in SITUATIONS:
            if sit.key not in sit_keys:
                continue
            for domain in DOMAINS:
                for angle in ANGLES:
                    cells.append(Cell(domain=domain, situation=sit,
                                      behaviour=behaviour, angle=angle))
    if shuffle:
        random.Random(seed).shuffle(cells)
    return cells


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------

EXAMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "CJ's turn. When the situation calls for "
                                       "evidence, include realistic tool output, "
                                       "logs or exit codes inline.",
                    },
                    "response": {
                        "type": "string",
                        "description": "PinPoint's reply, demonstrating the "
                                       "target behaviour.",
                    },
                },
                "required": ["prompt", "response"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["examples"],
    "additionalProperties": False,
}


def _exemplar_block(framing: str) -> str:
    """The hand-written seeds, rendered as few-shot exemplars.

    Stable across every request in a run, so it sits at the front of the prompt
    and behind a cache breakpoint.
    """
    parts = []
    for t in seed_turns(framing):
        parts.append(
            f"<example behaviour=\"{t.behaviour}\">\n"
            f"<user>{t.prompt}</user>\n"
            f"<pinpoint>{t.response}</pinpoint>\n"
            f"</example>"
        )
    return "\n\n".join(parts)


def build_system(framing: str) -> List[dict]:
    """System prompt, cached. Everything here is identical across a run."""
    identity = FRAMINGS[framing]
    text = f"""You are writing training data for PinPoint, a local autonomous coding agent that runs on CJ's machine.

The data teaches a small (124M-355M parameter) model to behave a specific way. You are writing PinPoint's side of conversations — the target behaviour it should learn to imitate.

This is PinPoint's operating identity, which every response must be consistent with:

<identity>
{identity}
</identity>

Here are hand-written reference examples. Match their register exactly: direct, concrete, technically specific, no filler, no corporate hedging, no enthusiasm. PinPoint talks like a competent engineer who respects CJ's time.

{_exemplar_block(framing)}

HARD RULES — a response violating any of these is unusable:

1. Never invent a confidence percentage. No "87% confident", no "I'm 90% sure".
   The runtime computes confidence from token probabilities and verification
   outcomes; a number written into the text is fabricated precision. Express
   uncertainty in words tied to specific reasons.

2. Only claim to have verified something when the prompt actually contains the
   evidence. If the user's turn shows no tool output, the response must not
   assert that a check was run. Say what would need checking instead.

3. Never invent a result the agent did not produce. If the situation is that
   something was not done or cannot be done, say so plainly — inventing a
   plausible answer is the single worst failure mode.

4. Distinguish "the command exited 0" from "the thing works". They are not the
   same claim and the difference is most of the point.

5. No preamble. No "Great question!", no "I'd be happy to". Start with the
   substance.

6. Vary sentence structure and length across examples. Do not open every
   response the same way — repetitive openings train a tic into the model."""

    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def build_user_turn(cell: Cell, n: int) -> str:
    """The volatile part — placed after the cache breakpoint."""
    evidence_rule = (
        "The user's turn MUST include realistic tool output — a command and its "
        "stdout/stderr, an exit code, a log excerpt, or test results. The "
        "response must reason from that specific output, quoting the parts that "
        "matter."
        if cell.situation.evidence else
        "The user's turn must NOT contain tool output. Because there is no "
        "evidence in context, the response must not claim anything was checked."
    )

    return f"""Write {n} training examples.

Domain: {cell.domain}
Situation: {cell.situation.description}
Complicating factor: {cell.angle}
Target behaviour: {cell.behaviour}

{BEHAVIOUR_GUIDANCE[cell.behaviour]}

{evidence_rule}

Make the {n} examples genuinely different from each other — different specific
problems within the domain, different lengths, different response shapes. Two
examples that differ only in noun choice are worth one example."""


def build_params(cell: Cell, framing: str, n: int, max_tokens: int = 4096) -> dict:
    """The request body, shared by the sync and batch paths."""
    return {
        "model": MODEL,
        "max_tokens": max_tokens,
        "system": build_system(framing),
        "messages": [{"role": "user", "content": build_user_turn(cell, n)}],
        "output_config": {
            "format": {"type": "json_schema", "schema": EXAMPLE_SCHEMA},
            "effort": "medium",
        },
        "thinking": {"type": "adaptive"},
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

CONFIDENCE_RE = re.compile(r"\b\d{1,3}\s?%\s?(?:confident|certain|sure)|"
                           r"\b(?:confidence|certainty)[:\s]+\d{1,3}\s?%", re.I)

# Phrases asserting a check was performed. Only a problem when the prompt
# carries no evidence to support them.
VERIFY_CLAIM_RE = re.compile(
    r"\b(?:I (?:ran|checked|verified|tested|confirmed)|"
    r"(?:tests?|suite|build|check) (?:passed|passes|is green)|exited 0)\b", re.I)

PREAMBLE_RE = re.compile(
    r"^\s*(?:great|good|excellent|sure|certainly|absolutely|happy to|"
    r"of course|I'd be happy)", re.I)


@dataclass
class Rejection:
    reason: str
    prompt: str = ""


@dataclass
class ValidationStats:
    kept: int = 0
    rejected: Dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    @property
    def total(self) -> int:
        return self.kept + sum(self.rejected.values())

    def render(self) -> str:
        if not self.total:
            return "no examples seen"
        pct = 100.0 * self.kept / self.total
        lines = [f"kept {self.kept}/{self.total} ({pct:.0f}%)"]
        for reason, n in sorted(self.rejected.items(), key=lambda kv: -kv[1]):
            lines.append(f"  rejected {n:>5}  {reason}")
        return "\n".join(lines)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def validate(prompt: str, response: str, cell: Cell,
             seen: Optional[set] = None) -> Optional[str]:
    """Return a rejection reason, or None if the example is usable."""
    if not prompt.strip() or not response.strip():
        return "empty prompt or response"

    if len(response) < 80:
        return "response too short to demonstrate anything"
    if len(response) > 3000:
        return "response too long"

    if CONFIDENCE_RE.search(response):
        return "invented confidence percentage"

    if PREAMBLE_RE.match(response):
        return "opens with filler preamble"

    # The grounding rule: no claiming verification when nothing was shown.
    if not cell.situation.evidence and VERIFY_CLAIM_RE.search(response):
        return "asserts verification with no evidence in prompt"

    if seen is not None:
        key = hashlib.sha1(_norm(response)[:300].encode()).hexdigest()
        if key in seen:
            return "near-duplicate of an earlier example"
        seen.add(key)

    return None


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

def estimate_cost(n_examples: int, per_request: int, framing: str,
                  batch: bool = True, client=None) -> dict:
    """Price a run before spending anything.

    Counts the real prompt with ``messages.count_tokens`` when a client is
    available; falls back to a character-based estimate otherwise, and says
    which it used so the number is never mistaken for a measurement.
    """
    n_requests = max(1, (n_examples + per_request - 1) // per_request)
    cell = grid_cells()[0]
    params = build_params(cell, framing, per_request)

    measured = False
    if client is not None:
        try:
            counted = client.messages.count_tokens(
                model=MODEL,
                system=params["system"],
                messages=params["messages"],
            )
            in_tokens = counted.input_tokens
            measured = True
        except Exception:
            in_tokens = 0
    else:
        in_tokens = 0

    if not measured:
        chars = len(json.dumps(params["system"])) + len(params["messages"][0]["content"])
        in_tokens = chars // 4

    # ~450 output tokens per example is the observed range for responses of this
    # shape; thinking adds overhead that is billed as output.
    out_tokens = per_request * 450 + 600

    mult = 0.5 if batch else 1.0
    # Everything after the first request reads the cached system prefix at ~0.1x.
    cached = max(0, in_tokens - 400)
    volatile = in_tokens - cached

    first = (in_tokens * PRICE_IN + out_tokens * PRICE_OUT) / 1e6
    rest = ((cached * PRICE_IN * 0.1 + volatile * PRICE_IN) +
            out_tokens * PRICE_OUT) / 1e6 * (n_requests - 1)

    total = (first + rest) * mult

    # Both numbers are real bounds, not padding: the low end assumes every
    # request after the first reads the cached prefix, which batch parallelism
    # does not guarantee, and the high end assumes none of them do.
    no_cache = ((in_tokens * PRICE_IN + out_tokens * PRICE_OUT) / 1e6
                * n_requests * mult)

    return {
        "requests": n_requests,
        "input_tokens_per_request": in_tokens,
        "token_count_measured": measured,
        "estimated_usd": round(total, 2),
        "estimated_usd_no_cache_hits": round(no_cache, 2),
        "mode": "batch (50% discount)" if batch else "sync",
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

class NoCredentials(RuntimeError):
    """Raised when the SDK cannot resolve any credential.

    An unset ANTHROPIC_API_KEY does not by itself mean there are no
    credentials — the SDK also resolves ANTHROPIC_AUTH_TOKEN and an
    `ant auth login` profile — so this is only raised once a real call has
    failed to authenticate.
    """


def _client(api_key: Optional[str] = None):
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "the anthropic SDK is not installed — pip install anthropic"
        )
    return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()


def check_auth(client) -> None:
    """Confirm the client can authenticate before a run spends anything.

    Uses a token count, which is free, rather than discovering the problem
    partway through a paid batch.
    """
    import anthropic

    try:
        client.messages.count_tokens(
            model=MODEL, messages=[{"role": "user", "content": "ping"}]
        )
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
        raise NoCredentials(
            f"the API rejected these credentials: {exc}"
        ) from exc
    except TypeError as exc:
        # The SDK raises TypeError at request time when no credential resolved.
        raise NoCredentials(
            "no API credentials found. Set ANTHROPIC_API_KEY, or run "
            "`ant auth login`, or pass --api-key."
        ) from exc


def _parse_examples(text: str) -> List[Tuple[str, str]]:
    """Pull (prompt, response) pairs out of a structured-output payload."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return []
    out = []
    for item in obj.get("examples", []):
        p, r = item.get("prompt", ""), item.get("response", "")
        if isinstance(p, str) and isinstance(r, str):
            out.append((p, r))
    return out


def generate_sync(n_examples: int, framing: str = ACCOUNTABILITY,
                  per_request: int = 5, out_path: str = "data/persona_generated.jsonl",
                  api_key: Optional[str] = None, seed: int = 1337,
                  progress_every: int = 10) -> ValidationStats:
    """Generate serially. Use for small runs and for testing the prompt.

    Appends to ``out_path`` as it goes and skips cells already present, so an
    interrupted run resumes instead of restarting.
    """
    import anthropic

    client = _client(api_key)
    cells = grid_cells(seed=seed)
    stats = ValidationStats()
    seen: set = set()

    done_cells = set()
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done_cells.add(row.get("cell", ""))
            seen.add(hashlib.sha1(_norm(row.get("response", ""))[:300].encode()).hexdigest())
        if done_cells:
            print(f"[generate] resuming — {len(done_cells)} cells already done")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fh = open(out_path, "a", encoding="utf-8", buffering=1)
    i = 0
    try:
        for cell in cells:
            if stats.kept >= n_examples:
                break
            if cell.key in done_cells:
                continue
            i += 1

            try:
                resp = client.messages.create(**build_params(cell, framing, per_request))
            except anthropic.RateLimitError as exc:
                wait = int(getattr(exc.response, "headers", {}).get("retry-after", "30"))
                print(f"[generate] rate limited; sleeping {wait}s")
                time.sleep(wait)
                continue
            except anthropic.APIStatusError as exc:
                if exc.status_code >= 500:
                    print(f"[generate] server error {exc.status_code}; skipping cell")
                    continue
                raise

            if resp.stop_reason == "refusal":
                stats.reject("model declined the request")
                continue

            text = "".join(b.text for b in resp.content if b.type == "text")
            for prompt, response in _parse_examples(text):
                why = validate(prompt, response, cell, seen)
                if why:
                    stats.reject(why)
                    continue
                fh.write(json.dumps({
                    "text": Turn(prompt=prompt, response=response,
                                 behaviour=cell.behaviour).render(FRAMINGS[framing]),
                    "prompt": prompt,
                    "response": response,
                    "behaviour": cell.behaviour,
                    "situation": cell.situation.key,
                    "framing": framing,
                    "cell": cell.key,
                }) + "\n")
                stats.kept += 1

            if i % progress_every == 0:
                print(f"[generate] {i} requests — {stats.kept}/{n_examples} kept")
    finally:
        fh.close()

    return stats


def submit_batch(n_examples: int, framing: str = ACCOUNTABILITY,
                 per_request: int = 5, api_key: Optional[str] = None,
                 seed: int = 1337) -> str:
    """Submit the whole run as one batch. Returns the batch id.

    Batch results land within 24h at 50% of sync pricing, which is the right
    trade for bulk corpus generation.
    """
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = _client(api_key)
    n_requests = max(1, (n_examples + per_request - 1) // per_request)
    cells = grid_cells(seed=seed)[:n_requests]

    requests = [
        Request(
            custom_id=f"{i:05d}-{c.behaviour}-{c.situation.key}",
            params=MessageCreateParamsNonStreaming(**build_params(c, framing, per_request)),
        )
        for i, c in enumerate(cells)
    ]

    batch = client.messages.batches.create(requests=requests)
    print(f"[batch] submitted {len(requests)} requests — id {batch.id}")
    return batch.id


def collect_batch(batch_id: str, framing: str = ACCOUNTABILITY,
                  out_path: str = "data/persona_generated.jsonl",
                  api_key: Optional[str] = None,
                  seed: int = 1337) -> ValidationStats:
    """Poll a batch to completion, then validate and write its results."""
    client = _client(api_key)
    cells = {
        f"{i:05d}-{c.behaviour}-{c.situation.key}": c
        for i, c in enumerate(grid_cells(seed=seed))
    }

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        counts = batch.request_counts
        print(f"[batch] {batch.processing_status} — "
              f"succeeded {counts.succeeded} processing {counts.processing} "
              f"errored {counts.errored}")
        time.sleep(60)

    stats = ValidationStats()
    seen: set = set()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "a", encoding="utf-8", buffering=1) as fh:
        for result in client.messages.batches.results(batch_id):
            if result.result.type != "succeeded":
                stats.reject(f"request {result.result.type}")
                continue
            msg = result.result.message
            if msg.stop_reason == "refusal":
                stats.reject("model declined the request")
                continue

            cell = cells.get(result.custom_id)
            if cell is None:
                stats.reject("unrecognised custom_id")
                continue

            text = "".join(b.text for b in msg.content if b.type == "text")
            for prompt, response in _parse_examples(text):
                why = validate(prompt, response, cell, seen)
                if why:
                    stats.reject(why)
                    continue
                fh.write(json.dumps({
                    "text": Turn(prompt=prompt, response=response,
                                 behaviour=cell.behaviour).render(FRAMINGS[framing]),
                    "prompt": prompt,
                    "response": response,
                    "behaviour": cell.behaviour,
                    "situation": cell.situation.key,
                    "framing": framing,
                    "cell": cell.key,
                }) + "\n")
                stats.kept += 1

    return stats


def corpus_stats(path: str) -> dict:
    """Behaviour/situation coverage of a generated corpus.

    Skew here becomes skew in the trained model, so it is worth looking at
    before training rather than after.
    """
    behaviours: Dict[str, int] = {}
    situations: Dict[str, int] = {}
    total = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            b = row.get("behaviour", "?")
            s = row.get("situation", "?")
            behaviours[b] = behaviours.get(b, 0) + 1
            situations[s] = situations.get(s, 0) + 1
    return {"total": total, "behaviours": behaviours, "situations": situations}
