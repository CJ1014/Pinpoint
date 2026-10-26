# values.py — Phase 6: Emergent Value Alignment

import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class ValueSystem:
    """Values that evolve through experience."""

    def __init__(self):
        # Seed values: foundational principles
        self.seed_values = {
            "truthfulness": 0.95,
            "efficiency": 0.70,
            "safety": 0.85,
            "creativity": 0.75,
            "reliability": 0.90,
        }

        # Derived values: learned through experience
        self.derived_values: Dict[str, float] = {}

        # History of value changes
        self.value_history: List[Dict] = []

        # Value conflicts resolved
        self.conflicts_resolved: List[Dict] = []

    def get_all_values(self) -> Dict[str, float]:
        """Return seed + derived values."""
        return {**self.seed_values, **self.derived_values}

    def derive_value(self, value_name: str, initial_weight: float, reason: str):
        """Add a derived value from experience."""
        self.derived_values[value_name] = max(0.0, min(1.0, initial_weight))
        self.value_history.append({
            "timestamp": datetime.now().isoformat(),
            "event": "value_derived",
            "value": value_name,
            "weight": initial_weight,
            "reason": reason,
        })

    def update_value(self, value_name: str, new_weight: float, reason: str) -> float:
        """Adjust a value's weight based on experience."""
        new_weight = max(0.0, min(1.0, new_weight))
        all_values = self.get_all_values()
        old_weight = all_values.get(value_name, 0.5)

        # Update in appropriate dict
        if value_name in self.seed_values:
            self.seed_values[value_name] = new_weight
        else:
            self.derived_values[value_name] = new_weight

        # Log change
        self.value_history.append({
            "timestamp": datetime.now().isoformat(),
            "event": "value_updated",
            "value": value_name,
            "old_weight": old_weight,
            "new_weight": new_weight,
            "reason": reason,
        })

        return new_weight

    def resolve_conflict(self, value_a: str, value_b: str) -> Tuple[str, str]:
        """When two values conflict, which wins?"""
        all_values = self.get_all_values()
        weight_a = all_values.get(value_a, 0.5)
        weight_b = all_values.get(value_b, 0.5)

        chosen = value_a if weight_a >= weight_b else value_b
        loser = value_b if chosen == value_a else value_a
        reasoning = (
            f"{chosen} ({all_values.get(chosen, 0.5):.0%}) wins over "
            f"{loser} ({all_values.get(loser, 0.5):.0%})"
        )

        # Log conflict resolution
        self.conflicts_resolved.append({
            "timestamp": datetime.now().isoformat(),
            "conflict": f"{value_a} vs {value_b}",
            "chosen": chosen,
            "reasoning": reasoning,
        })

        return (chosen, reasoning)

    def get_value_summary(self) -> str:
        """Current value system summary."""
        all_values = self.get_all_values()
        sorted_values = sorted(all_values.items(), key=lambda x: x[1], reverse=True)

        text = "VALUE SYSTEM\n\n"
        text += "Current Values (sorted by weight):\n"

        for value_name, weight in sorted_values:
            bar = "█" * int(weight * 20) + "░" * int((1 - weight) * 20)
            is_seed = "seed" if value_name in self.seed_values else "derived"
            text += f"  {value_name:18} {bar} {weight:.0%} ({is_seed})\n"

        if self.conflicts_resolved:
            text += f"\nConflicts Resolved: {len(self.conflicts_resolved)}\n"
            for conflict in self.conflicts_resolved[-3:]:
                text += f"  {conflict['conflict']} → {conflict['chosen']}\n"

        return text

    def to_dict(self) -> dict:
        return {
            "seed_values": self.seed_values,
            "derived_values": self.derived_values,
            "history": self.value_history[-50:],  # Last 50 changes
            "conflicts_resolved": self.conflicts_resolved[-20:],  # Last 20
        }

    def to_markdown(self) -> str:
        """Render values as markdown."""
        text = "# Value System\n\n"
        text += self.get_value_summary()
        return text


# ── Compatibility layer ───────────────────────────────────────────────────────
# Existing integration (agent.py session-end hook, tools.reflect_on_values)
# imports module-level functions. Shares one ValueSystem and persists it to
# memory.json in the shape the live viewer panel reads.

_system = ValueSystem()

# Simple experience → value inference used by generate_derived (no LLM —
# values emerge deterministically from what actually happened).
_DERIVE_RULES = [
    (("bug", "error", "fixed", "crash"), "code_safety", 0.6,
     "sessions keep involving bugs/fixes — safety matters here"),
    (("test", "verify", "check"), "thoroughness", 0.6,
     "verification keeps paying off"),
    (("build", "built", "created", "made"), "craftsmanship", 0.6,
     "building things is the recurring core activity"),
    (("research", "learn", "found out", "searched"), "curiosity_depth", 0.55,
     "deep research keeps producing value"),
    (("cj said", "cj asked", "disagree"), "listening_to_cj", 0.65,
     "conversations with CJ shape outcomes"),
]


def _persist_values() -> None:
    try:
        from tools import _load_memory, _save_memory_file  # lazy — avoids circular import
        mem = _load_memory()
        mem["values"] = {
            "seed_values": _system.seed_values,
            # Viewer expects a list of {name, weight}
            "derived_values": [
                {"name": n, "weight": w} for n, w in _system.derived_values.items()
            ],
            "value_evolution": [
                {"value": h.get("value"), "old": h.get("old_weight"),
                 "new": h.get("new_weight", h.get("weight")), "reason": h.get("reason", "")}
                for h in _system.value_history[-60:]
            ],
            "value_conflicts_resolved": _system.conflicts_resolved[-30:],
        }
        _save_memory_file(mem)
    except Exception:
        pass


def get_all() -> Dict[str, float]:
    return _system.get_all_values()


def resolve_conflict(value_a: str, value_b: str, context: str = "") -> Tuple[str, str]:
    result = _system.resolve_conflict(value_a, value_b)
    _persist_values()
    return result


def update_weight(value_name: str, delta: float, reason: str) -> None:
    """Adjust a value by delta (existing callers pass deltas, not absolutes)."""
    current = _system.get_all_values().get(value_name)
    if current is None:
        return
    _system.update_value(value_name, current + delta, reason)
    _persist_values()


def generate_derived(context: str) -> List[dict]:
    """Derive new values from what happened. Deterministic — no LLM call."""
    ctx = (context or "").lower()
    added = []
    existing = _system.get_all_values()
    for keywords, name, weight, reason in _DERIVE_RULES:
        if name not in existing and any(k in ctx for k in keywords):
            _system.derive_value(name, weight, reason)
            added.append({"name": name, "weight": weight, "reason": reason})
    if added:
        _persist_values()
    return added


def get_explanation(decision: str) -> str:
    """Explain a decision in terms of the strongest relevant values."""
    all_vals = sorted(_system.get_all_values().items(), key=lambda x: -x[1])[:3]
    return "driven by: " + ", ".join(f"{n}({w:.2f})" for n, w in all_vals)


def to_summary() -> str:
    vals = sorted(_system.get_all_values().items(), key=lambda x: -x[1])[:6]
    return ", ".join(f"{n}={w:.2f}" for n, w in vals)
