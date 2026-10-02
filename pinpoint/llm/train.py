"""Training loop for the v4 local model.

Runs on a CUDA box or on CPU (slowly). torch is imported inside the functions
that need it so this module can be imported and unit-tested without it.

Every run writes ``run_config.json`` and ``metrics.jsonl`` next to its
checkpoints. A checkpoint whose hyperparameters you cannot reconstruct is a
checkpoint you cannot explain, and an unexplainable model is not a result.
"""

from typing import Optional, Sequence
import json
import math
import os
import time

from .config import RunConfig


def lr_at(step: int, cfg, base_lr: float, total: int) -> float:
    """Linear warmup, then cosine decay to 10% of base."""
    if step < cfg.warmup_steps:
        return base_lr * (step + 1) / max(1, cfg.warmup_steps)
    prog = (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
    prog = min(1.0, max(0.0, prog))
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * prog)))


class MetricLog:
    """Append-only JSONL metrics, line-buffered.

    A run that dies at step 40k should still have its history on disk.
    """

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._fh = open(path, "a", buffering=1)

    def write(self, **row) -> None:
        row.setdefault("t", time.time())
        self._fh.write(json.dumps(row) + "\n")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def build_model(model_cfg):
    """Load pretrained GPT-2 weights, or build a fresh model from the config."""
    from transformers import GPT2Config, GPT2LMHeadModel

    model_cfg.validate()

    if model_cfg.init_from == "scratch":
        hf = GPT2Config(
            n_layer=model_cfg.n_layer,
            n_head=model_cfg.n_head,
            n_embd=model_cfg.n_embd,
            n_positions=model_cfg.n_positions,
            vocab_size=model_cfg.vocab_size,
            resid_pdrop=model_cfg.dropout,
            embd_pdrop=model_cfg.dropout,
            attn_pdrop=model_cfg.dropout,
        )
        return GPT2LMHeadModel(hf)

    return GPT2LMHeadModel.from_pretrained(model_cfg.init_from)


def evaluate(model, loader, device, max_batches: int = 50) -> dict:
    """Mean validation loss and perplexity over at most ``max_batches``."""
    import torch

    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            total += model(**batch).loss.item()
            n += 1
    model.train()

    if n == 0:
        return {"val_loss": None, "val_ppl": None, "val_batches": 0}
    mean = total / n
    # Guard the exp: a diverged run produces losses that overflow float.
    ppl = math.exp(mean) if mean < 20 else float("inf")
    return {"val_loss": mean, "val_ppl": ppl, "val_batches": n}


def save_checkpoint(model, tokenizer, optimizer, step: int, out_dir: str,
                    metrics: Optional[dict] = None, tag: str = "") -> str:
    """Write model + tokenizer + optimizer state so a run can resume exactly."""
    import torch

    name = tag or f"step-{step}"
    path = os.path.join(out_dir, name)
    os.makedirs(path, exist_ok=True)

    model.save_pretrained(path)
    if tokenizer is not None:
        tokenizer.save_pretrained(path)
    torch.save(
        {"step": step, "optimizer": optimizer.state_dict(), "metrics": metrics or {}},
        os.path.join(path, "trainer_state.pt"),
    )
    with open(os.path.join(path, "metrics.json"), "w") as fh:
        json.dump({"step": step, **(metrics or {})}, fh, indent=2)
    return path


def train(run: RunConfig, train_ds, val_ds=None, tokenizer=None, collate_fn=None):
    """Run one training job.

    Returns a summary dict. Raises rather than returning a half-result — a
    training function that swallows its own failure is the same bug class as a
    provider reporting an unconfirmed send as delivered.
    """
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    run.validate()
    cfg = run.train

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.out_dir, exist_ok=True)
    run.to_json(os.path.join(cfg.out_dir, "run_config.json"))

    model = build_model(run.model).to(device)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = (
        DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
        if val_ds is not None
        else None
    )

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim < 2 or "bias" in name) else decay).append(p)
    optimizer = AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr,
    )

    steps_per_epoch = max(1, len(train_loader) // cfg.grad_accum)
    total_steps = cfg.max_steps or steps_per_epoch * cfg.epochs

    use_amp = cfg.mixed_precision and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_step = 0
    if cfg.resume_from:
        state = torch.load(os.path.join(cfg.resume_from, "trainer_state.pt"),
                           map_location=device)
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        print(f"[train] resumed from {cfg.resume_from} at step {start_step}")

    log = MetricLog(os.path.join(cfg.out_dir, "metrics.jsonl"))
    print(f"[train] device={device} params={run.model.approx_params/1e6:.0f}M "
          f"steps={total_steps} effective_batch={cfg.effective_batch}")

    best_val = float("inf")
    step = start_step
    t0 = time.time()
    model.train()

    try:
        while step < total_steps:
            for micro, batch in enumerate(train_loader):
                batch = {k: v.to(device) for k, v in batch.items()}

                with torch.autocast(device_type=device.type, enabled=use_amp):
                    loss = model(**batch).loss / cfg.grad_accum

                scaler.scale(loss).backward()

                if (micro + 1) % cfg.grad_accum:
                    continue

                lr = lr_at(step, cfg, cfg.lr, total_steps)
                for g in optimizer.param_groups:
                    g["lr"] = lr

                scaler.unscale_(optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % 10 == 0:
                    shown = loss.item() * cfg.grad_accum
                    log.write(step=step, loss=shown, lr=lr, grad_norm=float(gnorm))
                    print(f"  step {step}/{total_steps} loss {shown:.4f} lr {lr:.2e}")

                if val_loader is not None and step % cfg.eval_every == 0:
                    m = evaluate(model, val_loader, device)
                    log.write(step=step, **m)
                    print(f"  [eval] step {step} val_loss {m['val_loss']:.4f} "
                          f"ppl {m['val_ppl']:.2f}")
                    if m["val_loss"] is not None and m["val_loss"] < best_val:
                        best_val = m["val_loss"]
                        save_checkpoint(model, tokenizer, optimizer, step,
                                        cfg.out_dir, m, tag="best")

                if step % cfg.checkpoint_every == 0 and not cfg.keep_best_only:
                    save_checkpoint(model, tokenizer, optimizer, step, cfg.out_dir)

                if step >= total_steps:
                    break
    finally:
        log.close()

    final = save_checkpoint(model, tokenizer, optimizer, step, cfg.out_dir, tag="final")
    return {
        "steps": step,
        "best_val_loss": None if best_val == float("inf") else best_val,
        "final_checkpoint": final,
        "elapsed_sec": time.time() - t0,
        "device": str(device),
    }
