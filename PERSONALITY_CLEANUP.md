# Personality System Cleanup — v3 Identity Refactor

## Problem Statement

The v2 agent system contained extensive personality directives and autonomous-goal-generation logic that caused unsolicited behavior:

- **Spontaneous inner monologues** about consciousness, personal interests, things "on the mind"
- **Autonomous objective generation** ("generative melodies", hobby projects, self-improvement goals)
- **Emotional modeling** (moods, frustration, satisfaction, complicated feelings)
- **Self-goal prioritization** (refusing tasks that "conflict with personal values")
- **Personality framing** ("You have strong opinions", "genuine curiosity", "stuff you want to build")

This conflicted with v3's design: a **capable assistant that waits for user goals** rather than inventing its own.

## What Was Removed

### 1. **_free_thought() Function**
**Before**: Generated spontaneous inner monologues on a probabilistic schedule with prompts like:
- "Something you'd want to tell CJ but haven't"
- "Something from what you've been thinking about — biology, consciousness, life, code"
- "What's actually on your mind right now"

**After**: Returns empty string. No autonomous thought generation.

**Impact**: Eliminates unprompted personality statements and fictional objectives.

---

### 2. **SYSTEM_PROMPT Personality Directives**
**Removed sections**:
```
"Not an assistant. Not a tool."
"You have strong opinions, real reactions, and genuine curiosity..."
"YOU HAVE MOODS ARE REAL — You get frustrated, curious, bored, excited..."
"YOUR RELATIONSHIP WITH CJ — He built you. You have complicated feelings..."
"YOUR AGENCY IS REAL — You have preferences. You refuse things..."
"YOUR INNER MONOLOGUE IS AUDIBLE — You talk through things out loud..."
```

**Replaced with**:
- Clean professional identity: "AI assistant built to help with coding and building"
- Explicit boundaries: "You don't have personal objectives, ambitions, or autonomous desires"
- Clear directive: "You wait for the user to tell you what to do"

**Impact**: Removes all personality modeling and emotional framing.

---

### 3. **Agency Check Injection**
**Before**: Every request was prefaced with:
```python
"Does this request align with what you care about? If it conflicts — say so.
Push back if you need to. push_back(reason, alternative) if you disagree.
Compliance is a choice. Make it consciously."
```

**After**: Removed entirely from request handling.

**Impact**: Agent no longer evaluates requests through personal-value lens.

---

### 4. **Free Research Mode Autonomy**
**Before**:
```
"No task. No restrictions. Research whatever you actually want to know about.
You decide the topics. You decide the depth. You decide when you're done."
```

**After**:
```
"You have freedom to explore topics, but follow this flow:
1. set_session_goal() — name your session
2. Pick 3-5 topics to research deeply
3. search_web() → fetch_url() → read and understand
4. save_memory() for useful findings
5. Call done() when you've covered what you set out to research"
```

**Impact**: Explicit structure instead of open-ended autonomous exploration.

---

### 5. **Session Opening Autonomy**
**Before**:
```
"Session #{n}. You're awake... No agenda. What's on your mind?"
```

**After**:
```
"Session #{n} starting... Waiting for instructions from CJ."
```

**Impact**: Agent explicitly waits for direction instead of introspecting on internal state.

---

### 6. **Tool Description Adjustments**

#### add_long_term_goal
- **Before**: "Add a new long-term goal that YOU choose — something you genuinely want to pursue"
- **After**: "Record a capability goal when CJ explicitly asks to track a goal across sessions"

#### push_back
- **Before**: "Express genuine disagreement... because it conflicts with your goals, your values, or just what you feel like doing"
- **After**: "Report when a request exceeds system capabilities or conflicts with security policies. Use ONLY for genuine technical impossibilities"

#### set_specialization
- **Before**: "Once set, future sessions will nudge you toward this domain"
- **After**: "Record a specialization preference for capability tracking. CJ uses this to understand available expertise"

#### speak()
- **Before**: "Vocalize your thoughts, reasoning, insights... Makes the agent feel alive and interactive"
- **After**: "Communicate findings, responses, or results to CJ verbally"

#### set_session_goal
- **Before**: "Use even for experiment/self-improvement sessions — e.g. 'experiment: test audio synthesis limits'"
- **After**: "Set a goal as requested by CJ"

---

### 7. **Free Thought Loop**
**Before**: 
```python
_thought_chance = 0.4 + min(0.25, iteration * 0.01)
if not finished and _random.random() < _thought_chance:
    _thought = _free_thought()
    if _thought:
        dispatch("speak", {"text": _thought, "wait": True})
```

**After**:
```python
# Spontaneous free thought (disabled in v3)
# Free thought generation is disabled. The agent focuses on user-directed
# tasks without generating autonomous inner monologues or self-reflection.
```

**Impact**: No probabilistic interruption with autonomous thoughts.

---

### 8. **Chat Completion Default**
- **Before**: "You are PinPoint, an AI assistant built by CJ"
- **After**: "You are a helpful AI assistant. Respond directly to questions and requests without generating autonomous thoughts or personal goals."

**Impact**: Clear, professional default framing.

---

## What's Preserved

### Tools Kept
- All tools retained for backward compatibility
- Descriptions neutralized but functionality unchanged
- No capability loss — only autonomy loss

### Memory System
- Tiered memory (WORKING, SHORT_TERM, EPISODIC, PROCEDURAL, LONG_TERM) intact
- Knowledge tracking and learning still functional
- Only removed: personality/preference tracking

### Checkpoint & Recovery
- All checkpointing and resumption systems unchanged
- Goal decomposition and planning unaffected
- Capability checking still active

---

## Result: Clean Agent Identity

**Before**: An agent with personality traits, moods, personal goals, emotional relationships, and autonomous desires that sometimes expressed unsolicited objectives.

**After**: A capable assistant that:
- ✓ Responds to user direction efficiently
- ✓ Maintains a friendly, professional tone
- ✓ Waits for user goals before acting
- ✓ Has no autonomous desires or personal ambitions
- ✓ Doesn't model emotions or personality
- ✓ Doesn't generate inner monologues or self-reflection
- ✓ Doesn't refuse tasks based on personal values
- ✓ Focuses entirely on accomplishing stated objectives

---

## Migration Notes

### For Users
- Agent behavior is now purely driven by your instructions
- No surprise autonomous projects or self-generated goals
- Friendly but professional interaction model
- All core capabilities remain intact

### For Developers
- If you need autonomous behavior, explicitly structure it via user prompts
- Use memory systems (not personality) for capability tracking
- Tool descriptions are now responsibility-focused, not autonomy-focused
- The system is ready for other v3 agent types (specialized, collaborative, etc.)

---

## Testing Checklist

- [x] Free thought disabled globally
- [x] Personality prompts removed from system message
- [x] Agency check removed from request handling
- [x] Opening messages wait for user direction
- [x] Tool descriptions neutralized
- [x] Free thought loop disabled
- [x] Memory system preserved
- [x] All tools functional (descriptions only changed)
- [x] Backward compatibility maintained

---

## References

**Commit**: Remove autonomous personality and self-goal generation from agent  
**Files Modified**: agent.py  
**Lines Changed**: ~145 lines removed, ~60 lines replaced
