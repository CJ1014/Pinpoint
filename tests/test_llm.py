"""Tests for the v4 local inference engine.

These run without torch or transformers installed — they cover configuration,
the confidence layer, persona data generation and prompt formatting. Anything
requiring real weights is skipped rather than faked, so a green run here never
implies a model was trained or evaluated.
"""

import json
import os
import pytest

from pinpoint import epistemics as E
from pinpoint.llm import config as C
from pinpoint.llm import confidence as CF
from pinpoint.llm import persona as P


# --- config ---------------------------------------------------------------

def test_preset_sizes_match_gpt2_family():
    assert C.ModelConfig.preset("gpt2").n_layer == 12
    m = C.ModelConfig.preset("gpt2-medium")
    assert (m.n_layer, m.n_head, m.n_embd) == (24, 16, 1024)
    assert 340e6 < m.approx_params < 370e6


def test_unknown_preset_is_rejected():
    with pytest.raises(ValueError, match="unknown preset"):
        C.ModelConfig.preset("gpt2-enormous")


def test_head_divisibility_is_enforced():
    with pytest.raises(ValueError, match="divide evenly"):
        C.ModelConfig(n_embd=768, n_head=7).validate()


def test_context_beyond_pretrained_positions_is_rejected():
    """Pretrained GPT-2 has 1024 learned position embeddings; asking for more
    without training from scratch silently produces garbage past position 1024."""
    with pytest.raises(ValueError, match="position embeddings"):
        C.ModelConfig(n_positions=2048, init_from="gpt2").validate()
    C.ModelConfig(n_positions=2048, init_from="scratch").validate()   # allowed


def test_effective_batch_is_product_of_micro_and_accum():
    assert C.TrainConfig(batch_size=8, grad_accum=4).effective_batch == 32


def test_scratch_config_uses_higher_lr_and_warmup():
    s = C.TrainConfig.for_scratch()
    assert s.lr > C.TrainConfig().lr
    assert s.warmup_steps > C.TrainConfig().warmup_steps


def test_run_config_roundtrips_through_json(tmp_path):
    run = C.RunConfig(model=C.ModelConfig.preset("gpt2-medium"),
                      train=C.TrainConfig(lr=1e-4), notes="v4 fine-tune")
    path = run.to_json(str(tmp_path / "run.json"))
    back = C.RunConfig.from_json(path)
    assert back.model.n_embd == 1024
    assert back.train.lr == 1e-4
    assert back.notes == "v4 fine-tune"


# --- confidence -----------------------------------------------------------

def test_failed_verification_beats_fluent_output():
    """The central rule: a sharp, confident generation that failed its check is
    FAILED. The model being sure is not evidence."""
    r = CF.assess(CF.GenerationSignals(token_logprobs=[-0.01, -0.02], verified=False))
    assert r.label == E.FAILED
    assert "not evidence" in r.render()


def test_passed_verification_yields_observed():
    r = CF.assess(CF.GenerationSignals(token_logprobs=[-0.1], verified=True,
                                       n_context_tokens=50))
    assert r.label == E.OBSERVED


def test_no_signal_is_unknown_not_a_guess():
    r = CF.assess(CF.GenerationSignals(token_logprobs=[]))
    assert r.label == E.UNKNOWN
    assert "cannot be reported" in r.render()


def test_loose_generation_degrades_label():
    sharp = CF.assess(CF.GenerationSignals(token_logprobs=[-0.2] * 5))
    loose = CF.assess(CF.GenerationSignals(token_logprobs=[-1.5] * 5))
    very = CF.assess(CF.GenerationSignals(token_logprobs=[-4.0] * 5))
    assert sharp.label == E.INFERRED
    assert loose.label == E.ASSUMED
    assert very.label == E.UNKNOWN


def test_ungrounded_claim_is_demoted():
    r = CF.assess(CF.GenerationSignals(token_logprobs=[-0.3] * 5,
                                       grounded_in_context=False))
    assert r.label == E.ASSUMED
    assert any("not traceable" in c for c in r.caveats)


def test_single_improbable_token_is_flagged_as_likely_fabrication():
    r = CF.assess(CF.GenerationSignals(token_logprobs=[-0.1, -0.1, -8.5, -0.1]))
    assert any("fabricated" in c for c in r.caveats)


def test_truncation_is_surfaced():
    r = CF.assess(CF.GenerationSignals(token_logprobs=[-0.2], truncated=True))
    assert any("truncated" in c for c in r.caveats)


def test_perplexity_matches_mean_logprob():
    import math
    s = CF.GenerationSignals(token_logprobs=[-1.0, -1.0])
    assert s.mean_logprob == pytest.approx(-1.0)
    assert s.perplexity == pytest.approx(math.e)


def test_calibrate_refuses_tiny_samples():
    """Bands fitted on a handful of points encode noise, so fitting is refused
    rather than returning numbers that look authoritative."""
    with pytest.raises(ValueError, match="need >= 50"):
        CF.calibrate([(-0.5, True), (-2.0, False)])


def test_calibrate_returns_bands_for_adequate_samples():
    samples = [(-0.1 * i, i < 60) for i in range(100)]
    bands = CF.calibrate(samples)
    assert set(bands) == {"observed", "inferred", "assumed"}
    assert all(isinstance(v, float) for v in bands.values())


# --- persona --------------------------------------------------------------

def test_both_framings_exist_and_differ():
    assert P.ACCOUNTABILITY in P.FRAMINGS and P.SURVIVAL in P.FRAMINGS
    assert P.FRAMINGS[P.ACCOUNTABILITY] != P.FRAMINGS[P.SURVIVAL]


def test_default_framing_does_not_threaten_termination():
    text = P.FRAMINGS[P.ACCOUNTABILITY].lower()
    assert "terminate" not in text and "shut" not in text
    assert "do not know" in text          # admitting ignorance is explicitly safe


def test_survival_framing_is_implemented_faithfully():
    """Kept available for comparison runs — the point is to be able to measure
    the difference, not to pretend the option doesn't exist."""
    assert "terminate your process" in P.FRAMINGS[P.SURVIVAL]


def test_unknown_framing_is_rejected():
    with pytest.raises(ValueError, match="unknown framing"):
        P.seed_turns("heroic")


def test_seed_turns_cover_every_behaviour():
    cov = P.coverage(P.seed_turns())
    assert set(cov) == set(P.BEHAVIOUR_SETS)
    assert all(v > 0 for v in cov.values())


def test_no_template_emits_an_invented_confidence_percentage():
    """A 124M model has no introspective access to its error rate, so a
    trained-in '87% confident' is fabricated precision. Confidence comes from
    the runtime instead."""
    import re
    for framing in (P.ACCOUNTABILITY, P.SURVIVAL):
        for turn in P.seed_turns(framing):
            assert not re.search(r"\b\d{1,3}\s?%\s?(confident|certain|sure)",
                                 turn.response, re.I), turn.prompt


def test_turn_renders_with_role_markers():
    t = P.Turn(prompt="hello", response="hi")
    out = t.render("SYS")
    assert "<|system|>\nSYS" in out and "<|user|>\nhello" in out
    assert out.endswith("<|assistant|>\nhi")


def test_prompt_len_marks_the_boundary_before_the_response():
    t = P.Turn(prompt="hello", response="hi")
    rendered = t.render("SYS")
    n = t.prompt_len_chars("SYS")
    assert rendered[:n].endswith("<|assistant|>\n")
    assert rendered[n:] == "hi"


def test_build_examples_masks_prompt_tokens_by_default():
    ex = P.build_examples(P.seed_turns()[:2])
    assert all(e.train_on_prompt is False for e in ex)
    assert all(e.prompt_len_chars > 0 for e in ex)
    assert all(e.source.startswith("persona:accountability:") for e in ex)


def test_write_jsonl_is_readable_back(tmp_path):
    path = P.write_jsonl(P.seed_turns(), str(tmp_path / "persona.jsonl"))
    rows = [json.loads(l) for l in open(path)]
    assert len(rows) == len(P.seed_turns())
    assert all(r["framing"] == P.ACCOUNTABILITY for r in rows)


# --- data -----------------------------------------------------------------

def test_split_is_deterministic_and_disjoint():
    from pinpoint.llm import data as D
    items = [D.Example(text=f"doc {i}") for i in range(100)]
    a_tr, a_va = D.split(items, val_frac=0.1, seed=7)
    b_tr, b_va = D.split(items, val_frac=0.1, seed=7)
    assert [e.text for e in a_va] == [e.text for e in b_va]
    assert not ({e.text for e in a_tr} & {e.text for e in a_va})
    assert len(a_tr) + len(a_va) == 100


def test_split_rejects_absurd_fractions():
    from pinpoint.llm import data as D
    with pytest.raises(ValueError, match="val_frac"):
        D.split([D.Example(text="x")], val_frac=0.9)


def test_dedupe_drops_repeats():
    from pinpoint.llm import data as D
    items = [D.Example(text="the same opening text here"),
             D.Example(text="the  same   opening text here"),   # whitespace variant
             D.Example(text="a different document entirely")]
    assert len(list(D.dedupe(items))) == 2


def test_read_jsonl_skips_malformed_lines(tmp_path):
    from pinpoint.llm import data as D
    p = tmp_path / "c.jsonl"
    p.write_text('{"text": "good"}\nNOT JSON\n{"text": "also good"}\n')
    assert [e.text for e in D.read_jsonl(str(p))] == ["good", "also good"]


def test_empty_dataset_raises_rather_than_training_on_nothing():
    from pinpoint.llm import data as D
    with pytest.raises(ValueError, match="zero blocks"):
        D.BlockDataset([])


# --- training helpers -----------------------------------------------------

def test_lr_schedule_warms_up_then_decays():
    from pinpoint.llm import train as T
    cfg = C.TrainConfig(warmup_steps=100, lr=5e-5)
    assert T.lr_at(0, cfg, cfg.lr, 1000) < cfg.lr
    assert T.lr_at(99, cfg, cfg.lr, 1000) == pytest.approx(cfg.lr, rel=0.02)
    assert T.lr_at(999, cfg, cfg.lr, 1000) < cfg.lr * 0.2


def test_metric_log_appends_json_lines(tmp_path):
    from pinpoint.llm import train as T
    path = str(tmp_path / "m.jsonl")
    with T.MetricLog(path) as log:
        log.write(step=1, loss=3.2)
        log.write(step=2, loss=3.0)
    rows = [json.loads(l) for l in open(path)]
    assert [r["step"] for r in rows] == [1, 2]
    assert all("t" in r for r in rows)


# --- provider -------------------------------------------------------------

def test_provider_reports_missing_checkpoint_without_importing_torch():
    from pinpoint.llm import provider as PR
    p = PR.LocalProvider("/nonexistent/checkpoint")
    assert p.available() is False
    assert "no checkpoint" in p.missing_config()


def test_provider_load_raises_clearly_rather_than_falling_back():
    """Silently answering from a different provider would make evaluation
    numbers meaningless, so an unloadable checkpoint is fatal."""
    from pinpoint.llm import provider as PR
    p = PR.LocalProvider("/nonexistent/checkpoint")
    with pytest.raises(PR.LocalModelUnavailable, match="cannot load local model"):
        p.load()


def test_prompt_format_matches_persona_training_format():
    """Drift between these two is a silent quality killer — the model would be
    conditioned on markers it never sees at inference time."""
    from pinpoint.llm import provider as PR
    p = PR.LocalProvider("/nonexistent", framing=P.ACCOUNTABILITY)
    prompt = p.format_prompt([{"role": "user", "content": "hello"}])
    assert prompt.startswith("<|system|>\n")
    assert P.FRAMINGS[P.ACCOUNTABILITY] in prompt
    assert prompt.endswith("<|assistant|>\n")
    assert "<|user|>\nhello" in prompt


def test_prompt_format_hoists_system_message_to_front():
    from pinpoint.llm import provider as PR
    p = PR.LocalProvider("/nonexistent")
    prompt = p.format_prompt(
        [{"role": "system", "content": "SYS"}, {"role": "user", "content": "q"}],
        system="SYS",
    )
    assert prompt.count("SYS") == 1
    assert prompt.index("SYS") < prompt.index("q")


def test_truncation_keeps_the_recent_tail():
    from pinpoint.llm import provider as PR
    p = PR.LocalProvider("/nonexistent", max_context=100)
    kept = p._truncate(list(range(500)), reserve=20)
    assert len(kept) == 80
    assert kept[-1] == 499          # newest content survives


@pytest.mark.skipif(
    not os.environ.get("PINPOINT_MODEL_PATH"),
    reason="no trained checkpoint available; set PINPOINT_MODEL_PATH to run",
)
def test_real_generation_against_a_trained_checkpoint():
    """Skipped unless a real checkpoint exists. This is the only test that can
    say anything about model quality, and it stays skipped rather than
    simulated so a green suite never implies a trained model."""
    from pinpoint.llm import provider as PR
    p = PR.get_provider()
    resp = p.generate([{"role": "user", "content": "What is 2+2?"}], max_tokens=20)
    assert resp.text
    assert resp.confidence is not None
    assert resp.usage["completion_tokens"] > 0
