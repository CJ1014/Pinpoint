# PinPoint Memory Architecture — v3 Cleanup Guide

## Overview

PinPoint v3 uses a **dual-layer memory system** to separate concerns between legacy v2 and new v3 architectures:

1. **memory.json** — v2 compatibility layer (minimal, legacy only)
2. **memory_tiers.json** — v3 tiered memory system (active, modern)

## What Changed

### ✗ Removed (Old v2 Experimental Categories)
The following categories in v2's `memory.json` were **experimental and obsolete**:
- `skills` — no longer tracked; tool registry is source of truth
- `lessons` — superseded by procedural memory in memory_tiers.json
- `mistakes` — captured in failure memory (pinpoint/memory/failures.py)
- `ideas` — not part of autonomous agent design
- `projects` — state is in git/task lists, not internal memory
- `preferences` — now in security/permissions.py
- `dislikes` — not relevant to autonomous agent
- `experiments` — experimental tracking removed for clarity
- `reflections` — not part of deterministic recovery logic
- `capability_map.user_emotions` — agent has no emotions to track
- `capability_map.real_world_events` — agent is not socially aware
- `capability_map.future_prediction` — agent makes no predictions

### ✓ Kept (v2 Compatibility)
- `memory.json` structure (for backward compatibility only)
- Reduced to bare minimum: `meta` + four practical categories
  - `project_notes` — for real procedural knowledge
  - `user_preferences` — for genuine settings (not personality)
  - `known_issues` — for documented problems
  - `useful_patterns` — for proven techniques

## v3 Memory Tiers (The Real System)

**Location**: `output/memory_tiers.json`

| Tier | TTL | Capacity | Purpose |
|------|-----|----------|---------|
| **working** | 1 day | 300 | Current execution state |
| **short_term** | 12 hours | 200 | Current session context |
| **episodic** | 90 days | 400 | Past sessions and events |
| **procedural** | ∞ | 200 | Proven workflows (high confidence) |
| **long_term** | ∞ | 300 | Stable facts, requires permission |

**Failure Memory** (separate): `pinpoint/memory/failures.py`
- Records diagnosis categories, root causes, recovery strategies
- Informs recovery engine decisions
- Survives across sessions

## Initialization Process

### On Startup
1. Load `memory.json` (v2 compat layer) — kept empty, just structure
2. Load `memory_tiers.json` (v3 active memory)
3. Load failure memory from checkpoint if resuming
4. Initialize tiered memory store → ready for session

### Never Loaded
- ✗ Old experimental persona data
- ✗ Emotional state or preferences about CJ
- ✗ Self-descriptions or long-term goals
- ✗ Predicted futures or speculation

## What Should Actually Be Stored

### ✓ GOOD Examples
- "Scenario C: First strategy may fail, needs alternatives" → procedural memory
- "User prefers explicit confirmations before writes" → short_term → long_term
- "SyntaxError in Python requires strategy switch" → procedural memory
- "Flask import failed → pip install flask" → procedural memory (recovery pattern)

### ✗ BAD Examples (Never Store)
- "I feel like helping you" → no emotions
- "My long-term goal is to become conscious" → not an agent goal
- "I should try to be more helpful" → not a learning target
- "I think CJ might be lonely" → no user modeling
- "Python is better than Rust" → no opinions

## Cleanup Status

✓ **Completed**
- Removed old experimental memory categories
- Cleared stale test simulation data from memory_tiers.json
- Simplified memory.json to v3 schema
- Reset version to 3 with schema documentation

✓ **Result**
- Fresh startup with no persona/emotional baggage
- No old goals or self-descriptions interfering
- Clean slate for v3 autonomous operation
- Backward compatible if needed

## Files Modified
- `memory.json` — cleaned
- `output/memory.json` — synced
- `output/memory_tiers.json` — reset to empty

## For Developers

### Adding New Memory
```python
from pinpoint.memory import store as MS

mem = MS.store()
mem.remember(
    "Scenario X taught us that strategy Y works",
    tier=MS.PROCEDURAL,
    category="workflow",
    relevance=0.8,
    confidence=0.9,
    source="scenario_test"
)
```

### Querying Memory
```python
results = mem.recall("strategy failure recovery", tier=MS.PROCEDURAL, limit=5)
for record in results:
    print(f"{record.content} (confidence: {record.confidence})")
```

## Next Steps
- Monitor actual autonomous sessions for what genuinely needs remembering
- Only add to long_term if user explicitly permits
- Trust failure memory (separate system) for recovery patterns
- Let working/episodic memory self-clean via TTL
