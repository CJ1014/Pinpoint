# PinPoint v3 — Architecture

PinPoint v3 turns the v2 agent into a system that takes a goal, plans it,
executes it, checks whether the work actually happened, recovers when it
didn't, and asks a human only when it genuinely needs to.

Nothing from v2 was thrown away. The eight cognition modules
(`reality_check`, `goal_tree`, `verification`, `reasoning_frames`,
`capability_map`, `checkpoint`, `values`, `human_oversight`) still live at the
repository root and still run. v3 is an additive `pinpoint/` package that the
two big v2 files call into through one seam, `pinpoint/integration.py`.

---

## The shape of it

```
LLM  →  agent.py loop  →  Executor  →  tools.dispatch  →  tool function
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
   permissions          verification          audit trail
   emergency stop       capabilities          failure memory
```

The rule the whole design turns on: **the model proposes, the policy engine
disposes.** Classification of an action is a pure function of its name and
parameters plus stored human grants. No sentence the model writes can change
what it is permitted to do.

---

## Packages

| Package | What it owns |
|---|---|
| `pinpoint/security/` | Permissions, approval, audit, emergency stop |
| `pinpoint/tools/` | The registry: what each tool costs and how to verify it |
| `pinpoint/action/` | ActionResult, independent verification, the executor |
| `pinpoint/agent/` | Intent, planner, recovery, orchestrator, persona |
| `pinpoint/memory/` | Tiered memory, preferences, failures, checkpoints |
| `pinpoint/communication/` | Messaging, calling, email, contacts, providers |
| `pinpoint/monitoring/` | Events, watchers, scheduler |

---

## The execution loop

```
UNDERSTAND → PLAN → CHECK CAPABILITY → EXECUTE → OBSERVE → VERIFY
          → LEARN → REPLAN → CONTINUE → COMPLETE / BLOCKED
```

Control flow is deterministic Python in `agent/orchestrator.py`. Choosing
*which tool* advances a task is the model's job, supplied as `step_fn` — which
is also what lets the whole loop be tested without an LLM.

One user request is not one model turn. A run spans as many actions, retries
and replans as its budget allows.

### Intent

`agent/intent.py` parses a request into an `Objective`: kind, priority,
autonomy, constraints, success and failure conditions, entities, deadline.
Deterministic — no round trip.

Unknowns stay unknown. "Text John" produces the unknown *"which contact 'John'
refers to"*, and `needs_clarification()` turns the blocking ones into a
question. Gaps the agent can resolve itself (which project is broken) do not
block.

### Planning

`agent/planner.py` is the v2 goal tree grown into a dependency graph. Tasks
carry dependencies, priority, an attempt budget, alternative strategies and
success criteria. Templates exist per objective kind.

Failure escalates rather than stopping: **retry → switch strategy → fail and
propagate a blocked state to dependents.** Work discovered mid-flight is
inserted into the graph and rewires whatever was waiting on it.

### Recovery

`agent/recovery.py` classifies a failure into one of 16 root-cause categories
and decides: retry, insert the prerequisite that was actually missing, switch
strategy, or escalate. A `ModuleNotFoundError` becomes a real task naming the
actual package, and the failed task is rewired to depend on it.

Bounded three ways — attempts per task, recoveries per session, and a
repeat-signature detector that changes approach instead of grinding.

---

## Never claiming what didn't happen

This is the part worth reading twice.

`ActionResult` has three outcomes where most systems have two:

- **SUCCESS** — the effect was independently confirmed.
- **FAILED** — it demonstrably didn't work.
- **UNVERIFIED** — it ran without complaint and the effect could not be
  confirmed. **This is not success.**

`action/verify.py` goes and checks the world rather than reading the prose:
it stats the file, parses the exit code, connects to the port, looks for the
provider's confirmation id. Where no independent check is possible the answer
is `None`, and the result carries a line telling the model in as many words
not to report it as done.

A cheerful `"Successfully wrote the file!"` with no file on disk is a FAILED
result.

---

## Security

### Levels

- **GREEN** — runs autonomously (reads, inspections, reversible local work).
- **YELLOW** — needs approval, unless the profile or a standing human grant
  covers it.
- **RED** — hard blocked. No profile, grant, or argument unblocks it.

RED covers: editing `pinpoint/security/`, destructive shell (`rm -rf /`,
`mkfs`, fork bombs, raw block-device writes), credential theft, credential
exfiltration, and anything that would disable PinPoint's own oversight.

### Profiles

| Profile | Behaviour |
|---|---|
| `SAFE` | Everything YELLOW asks first |
| `ASSISTED` (default) | Reversible local work proceeds; irreversible or external asks |
| `AUTONOMOUS` | Local work proceeds; money and the outside world still ask |
| `CUSTOM` | Per-tool levels |

Set with `PINPOINT_PROFILE`, or `permissions.set_profile(..., source="human")`.

### What the agent cannot do

- change its own autonomy profile
- grant itself a permission
- set its own custom tool levels
- clear the emergency stop
- edit the security package

Each of those has a test asserting the refusal.

### Emergency stop

`"STOP EVERYTHING"` in user input halts execution by **string match**, before
the model sees it — the halt does not depend on the model correctly
interpreting a panicking user. State lives in-process *and* on disk, so it
survives a restart. Anyone may engage it; only `source="human"` may clear it.

### Audit

Structured JSONL under `output/audit/`, one record per action: timestamp,
goal, tool, parameters, result, permission level, approval, verification,
duration. Credentials are redacted by key name *and* by pattern before
anything is written.

---

## Memory

| Tier | Lifetime | Rule |
|---|---|---|
| `working` | the session | anything goes; cleared at the end |
| `short_term` | hours | the current conversation |
| `episodic` | 90 days, capped | must clear a relevance gate |
| `procedural` | forever | workflows that actually worked |
| `long_term` | forever | **requires explicit permission to write** |
| failure | forever | postmortems, in `memory/failures.py` |

Every record carries relevance, confidence, source and timestamp. Nothing is
stored merely because it happened. At session end the best of working memory is
promoted to episodic and the rest is dropped.

**Lessons are hypotheses.** A failure's lesson gains confidence when its
recovery works and loses it when disputed; below the floor it stops being
offered, and it is always rendered with its confidence and a "check it still
applies" caveat.

### Preferences

Only what CJ *says* is learned — behaviour is never mined for inferred
preferences about a person. Preferences carry source, confidence, timestamp
and scope, and only `source=USER` ones translate into permission grants.

---

## Communication

Provider-agnostic. PinPoint talks to `communication/manager.py`; that talks to
whatever is configured.

```bash
# Messaging and voice (Twilio)
PINPOINT_SMS_PROVIDER=twilio
PINPOINT_VOICE_PROVIDER=twilio
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_FROM_NUMBER=+1...
PINPOINT_VOICE_CALLBACK_URL=https://...   # TwiML endpoint, voice only

# Email (SMTP)
PINPOINT_EMAIL_PROVIDER=smtp
SMTP_HOST=... SMTP_PORT=587
SMTP_USER=... SMTP_PASSWORD=... SMTP_FROM=...
```

Credentials come from the environment only, and are never written to
`memory.json`, the audit log, or anywhere else — there is a test that walks the
entire state directory looking for a leaked token.

**There is no simulation mode.** Unset means the capability does not exist, and
every call says so while naming the variables that are missing. A fake success
here would be indistinguishable from a real one to everything downstream.

Rules that hold regardless of provider:

- A send is reported as delivered **only** with a provider confirmation id —
  the manager downgrades any provider that claims success without one.
- An ambiguous recipient stops the send *before* the provider is touched and
  asks a question naming what distinguishes the candidates.
- Every outbound call opens by identifying itself as an automated assistant
  calling on someone's behalf. A call that asks to hide that is refused.

---

## Monitoring

```
EVENT → FILTER → RELEVANCE → ACTIVE GOAL → POLICY → ACT / NOTIFY / IGNORE
```

Every stage is deterministic Python; a test asserts the pipeline never reaches
for a model. Filtering drops duplicates within a cooldown and events below the
severity floor. The policy stage asks the permission engine whether the
proposed response could even run — a RED response downgrades to NOTIFY rather
than being attempted.

Watchers (port, process, file, HTTP, command) emit on **transitions**, so a
server down for an hour produces one event, not thousands.

The scheduler parses "tomorrow at 9", "in 10 minutes", "every day at 8am" and
ISO timestamps into absolute due times that survive a restart. An unreadable
time returns nothing rather than a guess.

---

## Checkpoints

`memory/checkpoint.py` records the objective, the plan graph, observations,
lessons, and — importantly — the **effects** it claims: files written, ports
serving.

Resuming does not trust the snapshot. Every effect is re-checked against the
world as it is now, and anything that no longer holds reopens the task that
produced it.

---

## Talking like a person

`agent/persona.py` keeps the scaffolding out of user-facing text. Internal logs
stay detailed; what CJ reads does not contain `[TOOL CALL]` or
`[VERIFICATION]`.

> Done.
>   - wrote demo.py
>   - ran demo.py

and when it isn't done:

> I ran send_message, but I can't confirm it actually worked (no provider
> confirmation id). I'm not going to call it done until I can.

---

## Tests

432 tests, no network, no LLM. Every test runs against a throwaway
`PINPOINT_HOME`, so none can touch a real `memory.json`.

```bash
pip install -r requirements.txt
python -m pytest
```

The acceptance scenarios in `tests/test_orchestrator.py` cover: a multi-step
task that succeeds, a tool failure that recovers, a wrong plan that replans, an
external action needing approval, an ambiguous recipient, a failed provider
call, a restart mid-task, a permission denial, and an emergency stop during
execution.

`tests/test_registry.py::test_every_agent_tool_is_registered` fails if a tool
is added to `agent.TOOLS` without a risk classification — the policy engine
cannot be bypassed by an unclassified tool.

---

## What is genuinely not built

Stated plainly, since the whole point of this architecture is not overclaiming:

- **In-call conversation.** `make_call` places a real call through Twilio and
  returns a real call SID. Turn-by-turn speech recognition during the call
  requires a media-stream endpoint that is not part of this repository;
  `PINPOINT_VOICE_CALLBACK_URL` is where that would plug in.
- **Computer control** depends on `pyautogui` and a graphical session. Without
  them the capability probe reports it as unavailable and offers the workaround
  rather than attempting it.
- **Inbound messages.** `get_messages` reads from the provider; there is no
  webhook receiver, so incoming messages are polled rather than pushed.
- **The v2 checkpoint modules** (`checkpoint.py`, `agi_checkpoint.py`,
  `pinpoint_checkpoint.py`) still exist and still run. v3 added a fourth that
  verifies against reality; consolidating the older three is not done.
