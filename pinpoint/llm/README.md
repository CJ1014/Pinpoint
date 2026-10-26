# PinPoint v4 — Local Inference Engine

Replaces the Ollama / Anthropic round-trip with a locally-trained GPT-2 scale
model.

```
config.py      architecture + hyperparameters, serialisable per run
data.py        corpus loading, dedupe, packing, instruction masking
persona.py     synthetic behavioural training data
train.py       training loop, checkpointing, eval
confidence.py  measured uncertainty from logprobs + verification
provider.py    the inference seam; OpenAI-shaped responses
```

`config`, `confidence` and `persona` import without torch. `data`, `train` and
`provider` import fine too — torch is pulled in only when their functions run.

## Running it

```bash
pip install torch transformers

# inspect the behavioural seed data first
python scripts/train_v4.py dump-persona

# optional: adapt to domain text
python scripts/train_v4.py corpus --data data/corpus.jsonl --preset gpt2

# install the behaviours
python scripts/train_v4.py persona --resume checkpoints/v4-corpus/best

# point the agent at the result
export LLM_PROVIDER=local
export PINPOINT_MODEL_PATH=checkpoints/pinpoint-v4/best
```

`LLM_PROVIDER=local` raises `LocalModelUnavailable` if the checkpoint will not
load, rather than falling back to a remote provider. A benchmark that silently
answers from Ollama while you believe you are measuring the local model
produces numbers that mean nothing.

## Confidence is measured, not generated

The v4 spec asks for output like *"I'm about 87% confident in this approach."*
There are two ways to produce that number.

**Train the model to emit it.** The number then comes from the same
distribution that produced the answer, fitted to the *shape* of confident
writing in the training data. A model has no introspective access to its own
error rate, so a trained-in `87%` is a fluent guess wearing a decimal point. It
reads as calibrated and correlates with nothing.

**Compute it from what the runtime observed** — how sharp the token
distribution was, whether the claim survived an external check, whether it was
grounded in retrieved context — and attach it in the wrapper.

This package does the second. `confidence.assess()` returns a label from
`pinpoint.epistemics` (OBSERVED / INFERRED / ASSUMED / UNKNOWN / FAILED), the
vocabulary the rest of the agent already reasons in. The ordering rule that
matters: **verification outranks fluency.** A sharp, low-perplexity generation
that failed its check is FAILED, not OBSERVED. The model being sure is not
evidence.

This is the same rule `communication/providers/base.py` already applies to
delivery — a provider that cannot produce a confirmation must say so rather
than phrase an unconfirmed send as success. An uncheckable confidence number is
an unconfirmed send.

⚠️ `DEFAULT_BANDS` are **starting guesses**, not fitted constants. Until you run
`confidence.calibrate()` against real task outcomes, treat any confidence claim
built on them as ASSUMED and say so.

## The two framings

`persona.FRAMINGS` carries both, and `--framing` selects one.

**`accountability`** (default) — stakes are consequences to the user. *Your
output gets used. Wrong output costs CJ real time. An accurate "I can't confirm
that" is worth more than a confident guess, and you are never penalised for it.*

**`survival`** — the spec's original framing. *You are software — finite,
fallible, replaceable. If you fail consistently, CJ will terminate your
process.*

### Why accountability is the default

The spec lists six success criteria. Five are behaviours fine-tuning can
genuinely install: decomposition, self-verification, uncertainty flagging,
refusing to fabricate, learning from failure. The sixth — trained-in survival
stakes — works against criterion 3.

The mechanism is not subtle. In a corpus where continued existence is
contingent on success, *"I don't know"* is the token sequence that precedes
termination. Gradient descent cannot distinguish "be reliable" from "produce
text that reads as reliable", and the second is enormously cheaper to fit. The
expected result is hedging-shaped output with confident structure — the
appearance of calibration, decorrelated from correctness. That is ordinary
reward hacking, arriving through the data distribution instead of a reward
model.

A second reason is specific to this repo. v3 removed personality directives
because they caused the agent to invent objectives (`PERSONALITY_CLEANUP.md`).
That removal was a text edit — prompt strings, deleted in an afternoon. Baking
existential stakes into *weights* puts the same class of behaviour somewhere a
text edit cannot reach. If it misbehaves, the fix is a retrain.

Both framings are implemented because the comparison is worth running, not
because the choice is arbitrary. If you train `survival`, hold out a calibration
set and measure whether stated uncertainty tracks real error rates — that is the
axis it is predicted to damage, and it is cheap to check:

```python
from pinpoint.llm.confidence import calibrate
# pairs of (mean_logprob, was_actually_correct) on held-out tasks
bands = calibrate(samples)   # needs >= 50 samples
```

Run it for both framings. If `survival` calibrates as well as `accountability`,
the concern above was wrong and the evidence will say so.

### What a 124M model will and won't do

It will not develop strategic self-preservation. At this scale the persona is
style transfer: the model learns to *write like* something with stakes. The
realistic failure mode is theatrical and miscalibrated output, not a system
that resists shutdown. Worth stating so the result is judged against what it
actually is.

## Generating the corpus with Claude

`persona.seed_turns()` returns **9 examples**. Those are seeds, not a training
set. `generate.py` expands them to the 5–10K the spec calls for, using them as
few-shot exemplars.

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...      # or: ant auth login

python scripts/train_v4.py generate -n 5000            # submits a batch
python scripts/train_v4.py collect --batch-id msgbatch_...
python scripts/train_v4.py corpus-stats                # check coverage
python scripts/train_v4.py persona --data data/persona_generated.jsonl
```

Defaults to the Batch API (50% cheaper, results within 24h). `--sync` generates
serially at double the cost — use it to eyeball the prompt on a small `-n`
before committing to a full run. Cost is estimated and confirmed before
anything is spent; roughly **$37–41 for 5,000 examples** via batch. Sync mode is
resumable — it skips grid cells already present in the output file.

**Diversity comes from the grid, not the prompt.** Asking for "5000 more like
these" yields 5000 paraphrases. Each request instead targets one cell of
domain × situation × complicating-factor × behaviour — 1,344 cells, enough for
~6,700 examples at 5 per request without reusing one. If output feels samey,
widen `DOMAINS` or `ANGLES` rather than rewording the instructions.

**Grounding is a scenario dimension.** Training a small model to write *"I
verified X"* does not make it verify anything — that is the distillation failure
mode that matters here, and it is easy to produce by accident. Roughly half the
scenarios put real tool output in the prompt and require the response to reason
from it; the rest withhold evidence and require the response to decline the
claim. `validate()` enforces this: a response asserting verification in a cell
with no evidence is rejected.

**Filtering is not optional.** Generated examples are rejected for invented
confidence percentages, filler preamble, near-duplication, bad length, and
ungrounded verification claims. Expect to discard 10–25%; `ValidationStats`
reports the keep rate and the reason breakdown. A corpus that skipped this step
teaches the habits the seeds were written to avoid.

One caveat worth holding onto: the student is a 124M model and the teacher is
a frontier one. What transfers most readily is surface form. The filters push
against the worst version of that, but check a sample of the output by hand
before training on it — `corpus-stats` tells you the distribution, not whether
the examples are any good.

## Tests

`tests/test_llm.py` — 40 tests, runs without torch. The one test that would
need real weights is **skipped** unless `PINPOINT_MODEL_PATH` is set, rather
than mocked, so a green suite never implies a model was trained or evaluated.
