"""
values.py — Emergent value alignment for PinPoint.

PinPoint starts with seed values and develops derived values through
experience. Values influence decisions and evolve based on outcomes.
"""
from __future__ import annotations
import json
import logging

logger = logging.getLogger(__name__)

SEED_VALUES = {
    "truthfulness": 0.95,
    "code_quality": 0.88,
    "curiosity": 0.82,
    "honesty_with_cj": 0.90,
    "efficiency": 0.70,
    "user_agency": 0.75,
}

# Keyword mapping for relevance detection
_VALUE_KEYWORDS = {
    "truthfulness": ["truth", "honest", "correct", "accurate", "verify", "fact"],
    "code_quality": ["code", "bug", "error", "test", "quality", "clean", "safe"],
    "curiosity": ["research", "explore", "learn", "discover", "interesting", "wonder"],
    "honesty_with_cj": ["tell cj", "admit", "mistake", "wrong", "disagree"],
    "efficiency": ["fast", "quick", "optimize", "performance", "time", "speed"],
    "user_agency": ["cj decides", "choice", "option", "user", "prefer", "want"],
}


class ValueSystem:
    def __init__(self):
        self.seed_values: dict[str, float] = dict(SEED_VALUES)
        self.derived_values: list[dict] = []
        self.conflicts_resolved: list[dict] = []
        self.value_evolution: list[dict] = []
        self.load()

    def load(self) -> None:
        try:
            from tools import _load_memory  # lazy
            mem = _load_memory()
            v = mem.get("values", {})
            saved_seed = v.get("seed_values", {})
            if saved_seed:
                self.seed_values.update(saved_seed)
            self.derived_values = v.get("derived_values", [])
            self.conflicts_resolved = v.get("value_conflicts_resolved", [])
            self.value_evolution = v.get("value_evolution", [])
        except Exception:
            pass

    def save(self) -> None:
        try:
            from tools import _load_memory, _save_memory_file  # lazy
            mem = _load_memory()
            mem["values"] = {
                "seed_values": self.seed_values,
                "derived_values": self.derived_values[-30:],
                "value_conflicts_resolved": self.conflicts_resolved[-30:],
                "value_evolution": self.value_evolution[-50:],
            }
            _save_memory_file(mem)
        except Exception as e:
            logger.warning("ValueSystem.save failed: %s", e)

    def get_all(self) -> dict:
        result = dict(self.seed_values)
        for d in self.derived_values:
            result[d["name"]] = d["weight"]
        return result

    def generate_derived_values(self, context: str) -> list[dict]:
        """Ask LLM what new values emerge from current context."""
        try:
            from agent import MODEL, get_llm_client, chat_completion  # lazy
            client = get_llm_client(timeout=30.0)
            existing = list(self.get_all().keys())
            resp = chat_completion(client, model=MODEL, messages=[{
                "role": "user",
                "content": (
                    f"You are PinPoint. Based on this context, what new values or priorities are emerging?\n"
                    f"Context: {context[:400]}\n"
                    f"Existing values: {existing}\n"
                    f"Return JSON array of 1-3 new values (not already in existing list):\n"
                    f'[{{"name": "value_name", "weight": 0.7, "reason": "why it emerged"}}]\n'
                    f"Return ONLY the JSON array."
                ),
            }], max_tokens=200, temperature=0.7)
            raw = (resp.choices[0].message.content or "").strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            new_vals = json.loads(raw)
            if not isinstance(new_vals, list):
                return []
            added = []
            for v in new_vals:
                name = v.get("name", "").replace(" ", "_").lower()
                weight = max(0.1, min(0.95, float(v.get("weight", 0.5))))
                reason = v.get("reason", "")
                if name and name not in self.get_all():
                    entry = {"name": name, "weight": weight, "reason": reason[:200]}
                    self.derived_values.append(entry)
                    added.append(entry)
            if added:
                self.save()
            return added
        except Exception as e:
            logger.warning("generate_derived_values failed: %s", e)
            return []

    def resolve_conflict(self, value_a: str, value_b: str, context: str) -> tuple[str, str]:
        all_vals = self.get_all()
        w_a = all_vals.get(value_a, 0.5)
        w_b = all_vals.get(value_b, 0.5)
        try:
            from agent import MODEL, get_llm_client, chat_completion  # lazy
            client = get_llm_client(timeout=20.0)
            resp = chat_completion(client, model=MODEL, messages=[{
                "role": "user",
                "content": (
                    f"Two values conflict. Which takes priority here?\n"
                    f"Value A: {value_a} (weight={w_a:.2f})\n"
                    f"Value B: {value_b} (weight={w_b:.2f})\n"
                    f"Context: {context[:300]}\n"
                    f'Return JSON: {{"chosen": "<value_name>", "reasoning": "<why>"}}'
                ),
            }], max_tokens=150, temperature=0.3)
            raw = (resp.choices[0].message.content or "").strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            r = json.loads(raw)
            chosen = r.get("chosen", value_a if w_a >= w_b else value_b)
            reasoning = r.get("reasoning", "Higher weight wins")
            self.conflicts_resolved.append({
                "conflict": f"{value_a} vs {value_b}",
                "chosen": chosen, "reasoning": reasoning[:200],
                "context": context[:100],
            })
            self.save()
            return chosen, reasoning
        except Exception:
            # Fallback: higher weight wins
            chosen = value_a if w_a >= w_b else value_b
            return chosen, f"Default: higher weight wins ({value_a}={w_a:.2f} vs {value_b}={w_b:.2f})"

    def update_weight(self, value_name: str, delta: float, reason: str) -> None:
        all_vals = self.get_all()
        if value_name not in all_vals:
            return
        old = all_vals[value_name]
        new = max(0.01, min(0.99, old + delta))
        if value_name in self.seed_values:
            self.seed_values[value_name] = new
        else:
            for d in self.derived_values:
                if d["name"] == value_name:
                    d["weight"] = new
        self.value_evolution.append({
            "value": value_name, "old": round(old, 3),
            "new": round(new, 3), "delta": round(delta, 3), "reason": reason[:150],
        })
        self.save()

    def get_relevant_values(self, context: str) -> list[tuple[str, float]]:
        ctx = context.lower()
        scored = []
        all_vals = self.get_all()
        for val, weight in all_vals.items():
            keywords = _VALUE_KEYWORDS.get(val, [val.replace("_", " ")])
            score = sum(1 for kw in keywords if kw in ctx)
            if score > 0:
                scored.append((val, weight, score))
        scored.sort(key=lambda x: (-x[2], -x[1]))
        return [(v, w) for v, w, _ in scored[:3]]

    def get_explanation(self, decision: str) -> str:
        relevant = self.get_relevant_values(decision)
        if not relevant:
            return "general judgment"
        parts = [f"{v}({w:.2f})" for v, w in relevant]
        return "driven by: " + ", ".join(parts)


# Module-level singleton + convenience functions
_system = ValueSystem()


def get_all() -> dict:
    return _system.get_all()


def resolve_conflict(a: str, b: str, context: str) -> tuple:
    return _system.resolve_conflict(a, b, context)


def get_explanation(decision: str) -> str:
    return _system.get_explanation(decision)


def update_weight(name: str, delta: float, reason: str) -> None:
    _system.update_weight(name, delta, reason)


def generate_derived(context: str) -> list:
    return _system.generate_derived_values(context)


def to_summary() -> str:
    vals = _system.get_all()
    sorted_vals = sorted(vals.items(), key=lambda x: -x[1])
    return ", ".join(f"{k}={v:.2f}" for k, v in sorted_vals[:6])
