"""Corpus loading, tokenization and batching for v4 training.

torch and tokenizers are imported lazily inside the functions that need them,
so this module stays importable for inspection and testing on machines with
neither installed.
"""

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence
import json
import os
import random


@dataclass
class Example:
    """One training example. ``loss_mask`` marks which tokens are trained on.

    For plain corpus text every token is trained. For instruction data the
    prompt is usually masked out so the model learns to *produce* responses
    rather than to reproduce the prompts it is given.
    """

    text: str
    source: str = "corpus"
    train_on_prompt: bool = True
    prompt_len_chars: int = 0        # chars of ``text`` that are prompt


def read_jsonl(path: str, text_key: str = "text") -> Iterator[Example]:
    """Stream a .jsonl corpus. Skips malformed lines loudly rather than silently."""
    bad = 0
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                if bad <= 5:
                    print(f"  [data] skipping malformed line {lineno} of {path}")
                continue
            text = obj.get(text_key)
            if isinstance(text, str) and text.strip():
                yield Example(text=text, source=obj.get("source", os.path.basename(path)))
    if bad:
        print(f"  [data] {path}: skipped {bad} malformed line(s)")


def read_text_dir(path: str, suffix: str = ".txt") -> Iterator[Example]:
    """Every file under ``path`` with ``suffix`` becomes one example."""
    for root, _dirs, files in os.walk(path):
        for name in sorted(files):
            if not name.endswith(suffix):
                continue
            full = os.path.join(root, name)
            try:
                with open(full, encoding="utf-8") as fh:
                    body = fh.read().strip()
            except (OSError, UnicodeDecodeError) as exc:
                print(f"  [data] unreadable {full}: {exc}")
                continue
            if body:
                yield Example(text=body, source=os.path.relpath(full, path))


def dedupe(examples: Iterable[Example], key_chars: int = 200) -> Iterator[Example]:
    """Drop near-duplicates by leading-substring hash.

    Web corpora are full of repeats, and duplicated text is the cheapest way to
    make a validation loss look better than the model is.
    """
    seen = set()
    for ex in examples:
        k = hash(" ".join(ex.text.split())[:key_chars])
        if k in seen:
            continue
        seen.add(k)
        yield ex


def split(examples: Sequence[Example], val_frac: float = 0.02, seed: int = 1337):
    """Deterministic train/val split.

    Held out by example, before any packing, so no validation text leaks into a
    training sequence through the concatenation in ``pack()``.
    """
    if not 0.0 < val_frac < 0.5:
        raise ValueError(f"val_frac must be in (0, 0.5), got {val_frac}")
    idx = list(range(len(examples)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(len(idx) * val_frac))
    val = [examples[i] for i in idx[:n_val]]
    train = [examples[i] for i in idx[n_val:]]
    return train, val


def get_tokenizer(name: str = "gpt2"):
    """GPT-2 BPE tokenizer, with a pad token that will not corrupt loss.

    GPT-2 ships no pad token. Setting pad = eos is standard, but only safe
    because ``pack()`` below produces fixed-length blocks with no padding at
    all — if you add padded batches later, mask the pads out of the loss.
    """
    from transformers import GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def pack(examples: Iterable[Example], tokenizer, block_size: int = 1024) -> List[List[int]]:
    """Concatenate examples with EOS separators and cut into fixed blocks.

    This is the standard causal-LM packing strategy: no padding, no wasted
    compute. The cost is that an example can straddle a block boundary, which
    is fine for corpus pretraining and wrong for instruction tuning — use
    ``pack_instruction()`` for that.
    """
    eos = tokenizer.eos_token_id
    buf: List[int] = []
    blocks: List[List[int]] = []
    for ex in examples:
        buf.extend(tokenizer.encode(ex.text))
        buf.append(eos)
        while len(buf) >= block_size:
            blocks.append(buf[:block_size])
            buf = buf[block_size:]
    return blocks


def pack_instruction(examples: Iterable[Example], tokenizer, block_size: int = 1024):
    """One example per sequence, prompt tokens masked out of the loss.

    Returns ``(input_ids, labels)`` pairs where masked positions are -100,
    which is the ignore index torch's cross-entropy expects.
    """
    eos = tokenizer.eos_token_id
    out = []
    for ex in examples:
        ids = tokenizer.encode(ex.text)[: block_size - 1] + [eos]
        labels = list(ids)
        if not ex.train_on_prompt and ex.prompt_len_chars:
            n_prompt = len(tokenizer.encode(ex.text[: ex.prompt_len_chars]))
            for i in range(min(n_prompt, len(labels))):
                labels[i] = -100
        out.append((ids, labels))
    return out


class BlockDataset:
    """Minimal torch Dataset over packed blocks.

    Defined as a plain class and given torch's Dataset interface at
    construction time, so importing this module never requires torch.
    """

    def __init__(self, blocks: Sequence[Sequence[int]]):
        if not blocks:
            raise ValueError("BlockDataset got zero blocks — check corpus paths")
        self.blocks = blocks

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, i):
        import torch

        ids = torch.tensor(self.blocks[i], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


class PairDataset:
    """Dataset over (input_ids, labels) pairs from ``pack_instruction``."""

    def __init__(self, pairs):
        if not pairs:
            raise ValueError("PairDataset got zero pairs")
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i):
        import torch

        ids, labels = self.pairs[i]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate(batch):
    """Pad a batch of variable-length pairs to the longest member."""
    import torch

    n = max(len(b["input_ids"]) for b in batch)
    pad_id, ignore = 0, -100
    ids, labels, mask = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        ids.append(torch.cat([b["input_ids"], torch.full((k,), pad_id, dtype=torch.long)]))
        labels.append(torch.cat([b["labels"], torch.full((k,), ignore, dtype=torch.long)]))
        mask.append(torch.cat([torch.ones(len(b["input_ids"]), dtype=torch.long),
                               torch.zeros(k, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(mask),
    }
