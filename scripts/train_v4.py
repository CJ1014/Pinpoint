#!/usr/bin/env python3
"""Train the PinPoint v4 local model.

Two stages, run in order:

  # 1. adapt to domain text (optional if you only want the behavioural layer)
  python scripts/train_v4.py corpus --data data/corpus.jsonl --preset gpt2

  # 2. install the behaviours: verification, decomposition, honest uncertainty
  python scripts/train_v4.py persona --resume checkpoints/v4-corpus/best \
      --framing accountability

  # inspect the seed persona data before training on it
  python scripts/train_v4.py dump-persona --framing accountability

Requires torch and transformers:
    pip install torch transformers

Nothing here fabricates a result. If the data is missing or the dependencies
are absent, it says so and exits non-zero.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinpoint.llm import config as C
from pinpoint.llm import persona as P


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        sys.exit(
            f"error: missing {exc.name}. Training needs torch and transformers:\n"
            f"    pip install torch transformers\n"
            f"This machine cannot train without them — no partial run is possible."
        )


def cmd_corpus(args) -> int:
    _require_torch()
    from pinpoint.llm import data as D
    from pinpoint.llm.train import train

    if not os.path.exists(args.data):
        sys.exit(f"error: no corpus at {args.data}")

    print(f"[corpus] reading {args.data}")
    examples = list(D.dedupe(D.read_jsonl(args.data)))
    if not examples:
        sys.exit(f"error: {args.data} yielded zero usable examples")
    print(f"[corpus] {len(examples)} examples after dedupe")

    tok = D.get_tokenizer()
    tr, va = D.split(examples, val_frac=args.val_frac)
    train_ds = D.BlockDataset(D.pack(tr, tok, args.block_size))
    val_ds = D.BlockDataset(D.pack(va, tok, args.block_size))
    print(f"[corpus] {len(train_ds)} train blocks / {len(val_ds)} val blocks")

    run = C.RunConfig(
        model=C.ModelConfig.preset(args.preset),
        train=C.TrainConfig(
            lr=args.lr, batch_size=args.batch_size, grad_accum=args.grad_accum,
            epochs=args.epochs, out_dir=args.out,
        ),
        notes=f"corpus adaptation from {args.data}",
    )
    summary = train(run, train_ds, val_ds, tokenizer=tok)
    print(f"[corpus] done: {summary}")
    return 0


def cmd_persona(args) -> int:
    _require_torch()
    from pinpoint.llm import data as D
    from pinpoint.llm.train import train

    turns = P.seed_turns(args.framing)
    if args.data and os.path.exists(args.data):
        extra = list(D.read_jsonl(args.data))
        print(f"[persona] {len(extra)} generated examples from {args.data}")
        examples = P.build_examples(turns, args.framing) + extra
    else:
        if args.data:
            sys.exit(f"error: no persona data at {args.data}")
        print(
            f"[persona] WARNING: training on {len(turns)} seed examples only.\n"
            f"          The spec calls for 5-10K. Seeds alone will not install\n"
            f"          a behaviour — use dump-persona, expand with a larger\n"
            f"          model, filter, then pass --data."
        )
        examples = P.build_examples(turns, args.framing)

    print(f"[persona] framing={args.framing} coverage={P.coverage(turns)}")

    tok = D.get_tokenizer()
    tr, va = D.split(examples, val_frac=args.val_frac)
    train_ds = D.PairDataset(D.pack_instruction(tr, tok, args.block_size))
    val_ds = D.PairDataset(D.pack_instruction(va, tok, args.block_size))

    run = C.RunConfig(
        model=C.ModelConfig.preset(args.preset),
        train=C.TrainConfig(
            lr=args.lr, batch_size=args.batch_size, grad_accum=args.grad_accum,
            epochs=args.epochs, out_dir=args.out, resume_from=args.resume,
        ),
        notes=f"behavioural fine-tune, framing={args.framing}",
    )
    if args.resume:
        run.model.init_from = args.resume

    summary = train(run, train_ds, val_ds, tokenizer=tok, collate_fn=D.collate)
    print(f"[persona] done: {summary}")
    print(
        "\nNext: measure calibration before trusting any uncertainty this model\n"
        "states. Collect (mean_logprob, was_correct) pairs on held-out tasks and\n"
        "fit bands with pinpoint.llm.confidence.calibrate(). Until then the\n"
        "default bands are guesses."
    )
    return 0


def cmd_dump_persona(args) -> int:
    turns = P.seed_turns(args.framing)
    path = P.write_jsonl(turns, args.out, args.framing)
    print(f"wrote {len(turns)} seed examples to {path}")
    print(f"coverage: {P.coverage(turns)}")
    return 0


def _confirm_spend(est: dict, yes: bool) -> None:
    how = "measured" if est["token_count_measured"] else "ESTIMATED from character count"
    print(f"\n  requests:     {est['requests']}")
    print(f"  mode:         {est['mode']}")
    print(f"  input tokens: {est['input_tokens_per_request']}/request ({how})")
    print(f"  cost:         ${est['estimated_usd']:.2f} – "
          f"${est['estimated_usd_no_cache_hits']:.2f}")
    print(f"                (low end assumes the cached prefix is hit on every "
          f"request after\n                 the first; batch parallelism does not "
          f"guarantee that)\n")
    if yes:
        return
    if input("proceed? [y/N] ").strip().lower() not in ("y", "yes"):
        sys.exit("aborted — nothing spent")


def cmd_generate(args) -> int:
    """Expand the seed set with Claude."""
    from pinpoint.llm import generate as G

    try:
        client = G._client(args.api_key or None)
        G.check_auth(client)
    except RuntimeError as exc:          # covers NoCredentials and missing SDK
        sys.exit(f"error: {exc}")

    est = G.estimate_cost(args.n, args.per_request, args.framing,
                          batch=not args.sync, client=client)
    _confirm_spend(est, args.yes)

    if args.sync:
        stats = G.generate_sync(args.n, framing=args.framing,
                                per_request=args.per_request, out_path=args.out,
                                api_key=args.api_key or None)
        print("\n" + stats.render())
        print(f"\ncorpus: {G.corpus_stats(args.out)}")
        return 0

    batch_id = G.submit_batch(args.n, framing=args.framing,
                              per_request=args.per_request,
                              api_key=args.api_key or None)
    print(f"\nresults land within 24h. Collect with:\n"
          f"    python scripts/train_v4.py collect --batch-id {batch_id} "
          f"--framing {args.framing} --out {args.out}")
    return 0


def cmd_collect(args) -> int:
    from pinpoint.llm import generate as G

    try:
        G.check_auth(G._client(args.api_key or None))
    except RuntimeError as exc:
        sys.exit(f"error: {exc}")

    stats = G.collect_batch(args.batch_id, framing=args.framing,
                            out_path=args.out, api_key=args.api_key or None)
    print("\n" + stats.render())
    print(f"\ncorpus: {G.corpus_stats(args.out)}")
    return 0


def cmd_corpus_stats(args) -> int:
    from pinpoint.llm import generate as G

    if not os.path.exists(args.path):
        sys.exit(f"error: no corpus at {args.path}")
    s = G.corpus_stats(args.path)
    print(f"total: {s['total']}")
    print("\nby behaviour:")
    for k, v in sorted(s["behaviours"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:24} {v:>6}  ({100*v/max(1,s['total']):.0f}%)")
    print("\nby situation:")
    for k, v in sorted(s["situations"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:24} {v:>6}  ({100*v/max(1,s['total']):.0f}%)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--preset", default="gpt2", choices=sorted(C.PRESETS))
        p.add_argument("--lr", type=float, default=5e-5)
        p.add_argument("--batch-size", type=int, default=8)
        p.add_argument("--grad-accum", type=int, default=4)
        p.add_argument("--epochs", type=int, default=3)
        p.add_argument("--block-size", type=int, default=1024)
        p.add_argument("--val-frac", type=float, default=0.02)

    c = sub.add_parser("corpus", help="domain adaptation on raw text")
    c.add_argument("--data", required=True, help=".jsonl with a 'text' field")
    c.add_argument("--out", default="checkpoints/v4-corpus")
    common(c)
    c.set_defaults(func=cmd_corpus)

    p = sub.add_parser("persona", help="behavioural fine-tune")
    p.add_argument("--data", default="", help="generated persona .jsonl")
    p.add_argument("--out", default="checkpoints/pinpoint-v4")
    p.add_argument("--resume", default="", help="checkpoint to continue from")
    p.add_argument("--framing", default=P.ACCOUNTABILITY, choices=sorted(P.FRAMINGS))
    common(p)
    p.set_defaults(func=cmd_persona, lr=3e-5)

    d = sub.add_parser("dump-persona", help="write seed examples for inspection")
    d.add_argument("--out", default="data/persona_seed.jsonl")
    d.add_argument("--framing", default=P.ACCOUNTABILITY, choices=sorted(P.FRAMINGS))
    d.set_defaults(func=cmd_dump_persona)

    g = sub.add_parser("generate", help="expand the seed set using Claude")
    g.add_argument("-n", type=int, default=5000, help="target example count")
    g.add_argument("--per-request", type=int, default=5)
    g.add_argument("--out", default="data/persona_generated.jsonl")
    g.add_argument("--framing", default=P.ACCOUNTABILITY, choices=sorted(P.FRAMINGS))
    g.add_argument("--sync", action="store_true",
                   help="generate serially instead of via the Batch API "
                        "(immediate results, 2x the cost)")
    g.add_argument("--api-key", default="", help="defaults to ANTHROPIC_API_KEY")
    g.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    g.set_defaults(func=cmd_generate)

    col = sub.add_parser("collect", help="collect a submitted batch")
    col.add_argument("--batch-id", required=True)
    col.add_argument("--out", default="data/persona_generated.jsonl")
    col.add_argument("--framing", default=P.ACCOUNTABILITY, choices=sorted(P.FRAMINGS))
    col.add_argument("--api-key", default="")
    col.set_defaults(func=cmd_collect)

    cs = sub.add_parser("corpus-stats", help="coverage of a generated corpus")
    cs.add_argument("path", nargs="?", default="data/persona_generated.jsonl")
    cs.set_defaults(func=cmd_corpus_stats)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
