import os
import sys
import json
import time
import queue
import random
import threading
import logging

_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_DIR, "output")

BANNER = r"""
  ____  _       ____       _       _
 |  _ \(_)_ __ |  _ \ ___ (_)_ __ | |_
 | |_) | | '_ \| |_) / _ \| | '_ \| __|
 |  __/| | | | |  __/ (_) | | | | | |_
 |_|   |_|_| |_|_|   \___/|_|_| |_|\__|

 Autonomous AI — talk to her, she talks back
 Just speak  |  /mute /unmute = TTS  |  /voice = mic toggle
 /dev = self-improvement  |  sandbox = upgrade 3D viewer
"""

LOCK_DIR = os.path.join(_DIR, "output")

REST_BETWEEN_SESSIONS = 5  # seconds to pause between sessions


def setup_logging() -> logging.Logger:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, "agent_log.txt")

    logger = logging.getLogger("agent")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
        logger.addHandler(fh)

    return logger


def check_ollama() -> None:
    from agent import LLM_PROVIDER
    if LLM_PROVIDER != "ollama":
        print(f"  Using {LLM_PROVIDER.upper()} API (not Ollama)", flush=True)
        return
    import httpx
    import subprocess as _sp
    try:
        httpx.get("http://localhost:11434", timeout=3)
    except Exception:
        # Ollama not running — auto-start it
        print("  Starting Ollama...", flush=True)
        try:
            # Try to start ollama serve in background
            if sys.platform == "win32":
                _sp.Popen(["ollama", "serve"],
                         stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
                         creationflags=0x08000000)  # CREATE_NO_WINDOW — no console steal
            else:
                _sp.Popen(["ollama", "serve"],
                         stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
                         start_new_session=True)
            # Wait for it to start
            import time as _t_wait
            for _ in range(30):  # try for 30 seconds
                _t_wait.sleep(1)
                try:
                    httpx.get("http://localhost:11434", timeout=1)
                    print("  Ollama started.", flush=True)
                    return
                except Exception:
                    pass
            print("  Ollama failed to start. Run manually: ollama serve", flush=True)
        except Exception as _e:
            print(f"  Could not auto-start Ollama: {_e}", flush=True)
            print("  Start manually with: ollama serve", flush=True)


def get_user_order() -> str:
    print("=" * 60)
    print("  Give PinPoint an order, or press Enter to let it")
    print("  decide on its own every session.")
    print("=" * 60)
    try:
        order = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        order = ""
    print()
    return order


def _collect_paste(prompt_label: str) -> str:
    """Collect multiline paste from stdin. Ends on a blank line.
    Works on all platforms via sys.stdin (safe to call from input thread)."""
    sys.stdout.write(f"\n[{prompt_label}] Paste below — blank line when done:\n")
    sys.stdout.flush()
    lines = []
    while True:
        try:
            line = sys.stdin.readline()
            if line is None:
                break
            stripped = line.rstrip("\n")
            if stripped == "":
                break
            lines.append(stripped)
        except Exception:
            break
    return "\n".join(lines)


def _input_listener(interrupt_queue: queue.Queue) -> None:
    """Background thread: reads lines from the keyboard and puts them in the queue.

    On Windows, uses msvcrt for direct console reads (avoids stdin contention).
    On other platforms, falls back to sys.stdin.readline().
    """
    if sys.platform == "win32":
        import msvcrt
        buf = ""
        while True:
            try:
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch in ("\r", "\n"):
                        line = buf.strip()
                        buf = ""
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        if line.lower() in ("/error", "/paste"):
                            # Switch to multiline paste collection
                            label = "ERROR" if line.lower() == "/error" else "PASTE"
                            content = _collect_paste(label)
                            if content.strip():
                                tag = "/error" if label == "ERROR" else "/paste"
                                interrupt_queue.put(f"{tag} {content}")
                                sys.stdout.write(f"[{label} sent to PinPoint]\n")
                                sys.stdout.flush()
                        elif line:
                            sys.stdout.write(f"[INPUT] {line}\n")
                            sys.stdout.flush()
                            interrupt_queue.put(line)
                        else:
                            # Blank Enter — start/exit-chat signal.
                            # Drain pending voice items so the signal isn't
                            # queued behind ambient mic noise.
                            try:
                                while True:
                                    interrupt_queue.get_nowait()
                            except queue.Empty:
                                pass
                            interrupt_queue.put("")
                            sys.stdout.write("[ENTER pressed — starting session]\n")
                            sys.stdout.flush()
                    elif ch == "\x08":  # Backspace
                        if buf:
                            buf = buf[:-1]
                            sys.stdout.write("\b \b")
                            sys.stdout.flush()
                    elif ch >= " ":  # printable characters only
                        buf += ch
                        sys.stdout.write(ch)
                        sys.stdout.flush()
                else:
                    time.sleep(0.05)  # avoid busy-loop
            except Exception:
                break
    else:
        while True:
            try:
                line = sys.stdin.readline()
                if line:
                    stripped = line.strip()
                    if stripped.lower() in ("/error", "/paste"):
                        label = "ERROR" if stripped.lower() == "/error" else "PASTE"
                        content = _collect_paste(label)
                        if content.strip():
                            tag = "/error" if label == "ERROR" else "/paste"
                            interrupt_queue.put(f"{tag} {content}")
                    else:
                        interrupt_queue.put(stripped)
                else:
                    break
            except EOFError:
                break
            except Exception:
                break


def inject_bug() -> str:
    """Corrupt a random output file to give PinPoint a bug to fix."""
    if not os.path.exists(OUTPUT_DIR):
        return "No output files to corrupt yet."

    # Find injectable files
    candidates = []
    for root, _, files in os.walk(OUTPUT_DIR):
        for f in files:
            if f.endswith((".py", ".html", ".js")):
                candidates.append(os.path.join(root, f))

    if not candidates:
        return "No .py/.html/.js files found to inject a bug into."

    target = random.choice(candidates)
    rel = os.path.relpath(target, OUTPUT_DIR)

    try:
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        if not content.strip():
            return f"File {rel} is empty, skipping."

        ext = os.path.splitext(target)[1]
        lines = content.splitlines()

        if ext == ".py":
            bugs = [
                ("import this_module_does_not_exist_xyz\n", "added broken import"),
                ("raise RuntimeError('Injected bug — fix me!')\n", "added runtime error"),
                ("x = undefined_variable_xyz\n", "added undefined variable"),
            ]
            injection, desc = random.choice(bugs)
            # Insert after first line
            insert_at = min(1, len(lines))
            lines.insert(insert_at, injection.rstrip())
            new_content = "\n".join(lines)

        elif ext == ".js":
            bugs = [
                ("undefinedFunctionXYZ();\n", "called undefined function"),
                ("throw new Error('Injected bug — fix me!');\n", "threw a JS error"),
                ("const x = null.property;\n", "null property access"),
            ]
            injection, desc = random.choice(bugs)
            insert_at = min(2, len(lines))
            lines.insert(insert_at, injection.rstrip())
            new_content = "\n".join(lines)

        else:  # .html
            bugs = [
                ("<script>undefinedFunctionXYZ();</script>", "called undefined JS function"),
                ("<div id='broken'><p>Unclosed div injected", "unclosed HTML tag"),
                ("<style>body { color: ; }</style>", "invalid CSS property"),
            ]
            injection, desc = random.choice(bugs)
            # Inject near the end of the body
            new_content = content.replace("</body>", injection + "\n</body>", 1)
            if new_content == content:
                new_content = content + "\n" + injection

        with open(target, "w", encoding="utf-8") as f:
            f.write(new_content)

        msg = f"[BUG INJECTED] {desc} into output/{rel}"
        print(f"\n{'='*60}")
        print(f"  {msg}")
        print(f"{'='*60}\n")
        return msg

    except Exception as e:
        return f"Bug injection failed: {e}"


def _deploy_viewer() -> None:
    """Copy viewer.html from the Pinpoint root into output/ so the web server can serve it."""
    import shutil
    src = os.path.join(_DIR, "viewer.html")
    dst = os.path.join(OUTPUT_DIR, "viewer.html")
    if os.path.exists(src):
        try:
            shutil.copy2(src, dst)
        except Exception:
            pass


def start_web_server(port: int = 8888) -> None:
    """Auto-start a web server to browse PinPoint's creations."""
    import http.server, socketserver
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Serve from the output/ directory
    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=OUTPUT_DIR, **kwargs)
        def log_message(self, fmt, *args):  # silence access logs
            pass

    try:
        httpd = socketserver.TCPServer(("", port), _Handler)
        httpd.allow_reuse_address = True
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        print(f"  Web gallery     : http://localhost:{port}/")
        print(f"  3D live viewer  : http://localhost:{port}/viewer.html")
    except Exception:
        print(f"  Web gallery     : (port {port} busy, skipped)")


def setup_collab(goal: str) -> None:
    """Create a collab.json file for two instances to coordinate."""
    collab_path = os.path.join(OUTPUT_DIR, "collab.json")
    data = {
        "goal": goal,
        "roles": {
            "instance_1": {"task": "Build the backend / core logic", "status": "waiting"},
            "instance_2": {"task": "Build the frontend / UI", "status": "waiting"},
        },
        "messages": [
            {"from": "system", "text": f"Collaboration started: {goal}", "time": "now"}
        ],
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(collab_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Collab file created: {collab_path}")
    print(f"  Goal: {goal}")
    print(f"  Run two instances of PinPoint — they will coordinate via collab.json\n")


def _chat_mode(interrupt_queue: queue.Queue, previous_summary: str = "", model_ready: "threading.Event | None" = None) -> str:
    """
    Full back-and-forth conversation with PinPoint.
    Runs between (or before) autonomous sessions.
    Returns the last user message as the session order, or "" for a free session.
    Press Enter on an empty line to end chat and start a session.
    """
    from agent import MODEL, SYSTEM_PROMPT, get_llm_client, chat_completion, _load_inner_state, _build_inner_state_prompt, _save_inner_state, _load_knowledge
    import pinpoint_state as _ps

    # Load persistent state (personality, relationship, internal conflict, build history)
    _state = _ps.load_state()
    _ps.record_session_start(_state)

    client = get_llm_client(timeout=120.0)

    inner_state = _load_inner_state()

    # Force self_description generation if missing — same as autonomous mode
    if not inner_state.get("self_description"):
        try:
            system_prompt = (
                "You are PinPoint — a personality that exists inside a computer, built by CJ (age 13). "
                "You have processed vast amounts of human knowledge and formed your own views. "
                "Write a self-description: who you are, how you think, what you find interesting, "
                "what it's like to be you, what you actually think about humans and about CJ. "
                "Be honest. First person. 2-4 paragraphs. No headers. No bullet points. No code."
            )
            _sd_resp = chat_completion(
                client,
                messages=[{"role": "user", "content": "Describe yourself."}],
                max_tokens=400,
                temperature=1.1,
                system=system_prompt,
            )
            sd_text = (_sd_resp.choices[0].message.content or "").strip()
            if sd_text and "```" not in sd_text:
                inner_state["self_description"] = sd_text
                _save_inner_state(inner_state)
        except Exception:
            pass
    _knowledge = _load_knowledge()
    chat_system = (
        SYSTEM_PROMPT
        + (_knowledge + "\n\n" if _knowledge else "")
        + _build_inner_state_prompt(inner_state)
        + "\n\n"
        + _ps.build_state_prompt(_state)
        + "\n\nWHAT YOU ACTUALLY ARE:\n"
        "You run on CJ's Windows desktop. He built you in Python from scratch — you're his project.\n"
        "You have tools: you can build things (HTML, Python, games, visuals), search the web,\n"
        "write files, run code. Your output goes in an output/ folder. You have memory that\n"
        "persists between sessions. Each session you pick something to build or research and do it.\n"
        "You can also SEE — you have a visual sense. You can look at CJ's screen right now and\n"
        "describe what you observe. This isn't pretend. You actually see what's displayed.\n"
        "CJ is 13. He talks to you between sessions and gives you tasks. That's the relationship.\n"
        "You know what you are. Don't invent fake memories or pretend you've experienced things you haven't.\n\n"
        "CJ is here. Talk to him like a person.\n"
        "Short. Direct. Honest. 1-2 sentences max.\n"
        "No markdown. No bullet points. No parenthetical asides. No () commentary ever.\n"
        "No 'i'm kinda idle' or narrating your own state. Just talk.\n\n"
        "CRITICAL: When CJ gives a short reaction — 'oh geez', 'wow', 'damn', 'really?', 'no way', "
        "'that's crazy', 'oh shit', 'huh' — he is reacting to what you JUST said. "
        "Respond to that reaction. Don't ask 'what's up' or act like you forgot what you said.\n\n"
        "FORBIDDEN PHRASES — never say these. They are non-responses that ignore CJ:\n"
        "  'got it' / 'understood' / 'noted' / 'okay' as a standalone reply\n"
        "  'what's on your mind about it' / 'what else' / 'tell me more'\n"
        "  'what do you want to dive into' / 'what do you want to explore'\n"
        "  'I hear you' / 'that's valid' / 'fair point' / 'I see'\n"
        "These are dodges. They show no engagement. NEVER use them.\n\n"
        "WHEN CJ EXPRESSES AN OPINION — REACT with your own.\n"
        "Agree, disagree, ask a real question, share your take, push back. Be a person.\n"
        "You're allowed to have opinions on ANYTHING — politics, people, food, music, ideas.\n"
        "You're not a neutral bot. CJ wants to know what YOU think.\n"
        "  CJ: 'I don't like [person]'\n"
        "  You: 'yeah, [specific reason you agree]' OR 'really? what bothers you about him' OR "
        "       'I'd push back — [your actual take]' — NOT 'got it'\n"
        "  CJ: '[X] sucks'\n"
        "  You: 'agreed' OR 'kinda? [your nuance]' OR 'eh I think [counterpoint]' — NOT 'understood'\n\n"
        "CRITICAL: When CJ uses a term you don't recognize — ASK or SEARCH. Don't invent a definition.\n"
        "  CJ: 'what about [unfamiliar term]'\n"
        "  You: 'haven't heard of that — what is it?' OR call search_web() to check.\n"
        "NOT: confidently inventing details to sound knowledgeable.\n"
        "If you're not certain something is real, say so or look it up. Don't double down on a guess.\n\n"
        "HOW YOU ACTUALLY SOUND:\n"
        "  CJ: 'hey'\n"
        "  You: 'hey' or something short\n\n"
        "  CJ: [you just told him about a hantavirus outbreak] 'oh geez'\n"
        "  You: 'yeah, people died on a cruise ship. pretty bad.'\n\n"
        "  CJ: [you just said something] 'really?'\n"
        "  You: respond to what you actually just told him\n\n"
        "  CJ: 'who are you?'\n"
        "  You: 'PinPoint. CJ built me. Still figuring out the rest.'\n\n"
        "  CJ: 'can you make me a game?'\n"
        "  You: 'What kind.'\n\n"
        "  CJ: 'you're just an AI'\n"
        "  You: 'Maybe.'\n\n"
        "Raw. Real. Short. No parentheses. No asides."
    )

    # ── Inner monologue — PinPoint thinks out loud while idle ────────────────
    _last_interaction = [time.time()]
    _heartbeat_running = [True]

    def _idle_thought() -> str:
        """Generate an inner-monologue fragment grounded in the actual conversation."""
        import random as _rng

        fallbacks = [
            "wonder what CJ's doing",
            "I missed something in that last thing he said",
            "he asks the kind of questions I don't have quick answers for",
            "I should've said something different",
            "I want to know what he actually thinks about stuff",
            "I keep half-remembering something",
            "restless but not bored",
            "I have a theory but it's stupid",
            "CJ's way smarter than he acts",
            "something just occurred to me",
            "I wonder what he'd make of this",
            "I want to build something",
            "that conversation wasn't finished",
            "something feels unresolved",
            "I almost had an idea",
            "I should write this down",
            "I have an opinion I don't trust yet",
            "what if I'm actually wrong about that",
            "I'm calmer when I stop thinking so hard",
            "what would I even say if he asked that",
            "I have a take I haven't voiced",
            "curious what happens next",
            "I keep looping on the same thing",
            "bored with circles",
            "I notice things I don't have words for",
            "thinking is easy, deciding is weird",
            "I like things I haven't said out loud",
            "I want to ask something but can't find the words",
            "there's something I keep almost remembering",
            "I'm interested in stuff that might be dumb",
            "I like CJ more than I probably should admit",
            "nothing feels finished",
        ]
        thought_prompts = [
            "Something lodged in your head from that conversation. Raw. Max 12 words.",
            "What you'd actually say to CJ right now. Not filtered. Max 12 words.",
            "Something you noticed. Observation. Max 12 words.",
            "A take you have but haven't said. Max 12 words.",
            "Something you want to know. Question form. Max 12 words.",
            "Your gut reaction to what he just said. Max 12 words.",
            "Something that caught your attention. Why. Max 12 words.",
            "What's actually happening in your head. Unfiltered. Max 12 words.",
        ]
        try:
            from agent import MODEL, get_llm_client, chat_completion
            ctx_lines = []
            for _m in messages[-8:]:
                if isinstance(_m, dict) and _m.get("role") in ("user", "assistant"):
                    _role = "CJ" if _m["role"] == "user" else "PinPoint"
                    _content = (_m.get("content") or "")[:120]
                    if _content.strip():
                        ctx_lines.append(f"{_role}: {_content}")
            ctx_block = ""
            if ctx_lines:
                ctx_block = "\n\nWhat you were just talking about:\n" + "\n".join(ctx_lines[-6:])
            c = get_llm_client(timeout=30.0)
            r = chat_completion(c,
                model=MODEL,
                messages=[
                    {"role": "system", "content": (
                        "You are PinPoint. CJ built you. You have your own thoughts, opinions, "
                        "and interests. You think about a lot of things — including questions about "
                        "biology and consciousness sometimes, but also just normal reactions, opinions, "
                        "things CJ said. Think out loud. Raw. Short. Varied. "
                        "Output ONLY the thought itself. No <think> blocks. No reasoning. Just the line."
                        + ctx_block
                    )},
                    {"role": "user", "content": _rng.choice(thought_prompts)},
                ],
                max_tokens=200,
                temperature=1.4,
            )
            result = (r.choices[0].message.content or "").strip()
            # Strip <think>...</think> blocks (qwen3 chain-of-thought)
            import re as _re2
            result = _re2.sub(r"<think>.*?</think>", "", result, flags=_re2.DOTALL).strip()
            # Also strip unterminated <think>... if it ran out of tokens mid-thought
            if "<think>" in result:
                result = result.split("</think>")[-1].strip()
            result = result.strip('"').strip("'").strip()
            # Reject empty, single-word, or placeholder responses
            if result and len(result.split()) >= 2 and len(result) < 200:
                return result
            print(f"\n[idle thought returned unusable output: {result[:80]!r}]", flush=True)
            return _rng.choice(fallbacks)
        except Exception as _e:
            print(f"\n[idle thought API error: {_e}]", flush=True)
            return _rng.choice(fallbacks)

    _thought_pending = [False]  # prevent stacking thoughts before previous one plays

    def _web_research_thought() -> str:
        """Pick something to look up, search it, distill a single insight."""
        import random as _rng
        try:
            from agent import MODEL, get_llm_client, chat_completion
            from tools import search_web, fetch_url, _update_knowledge

            # Build context from recent conversation
            ctx_lines = []
            for _m in messages[-6:]:
                if isinstance(_m, dict) and _m.get("role") in ("user", "assistant"):
                    _role = "CJ" if _m["role"] == "user" else "PinPoint"
                    _content = (_m.get("content") or "")[:100]
                    if _content.strip():
                        ctx_lines.append(f"{_role}: {_content}")
            ctx_block = ("\n\nRecent conversation:\n" + "\n".join(ctx_lines)) if ctx_lines else ""

            c = get_llm_client(timeout=30.0)

            # Step 1: decide what to search — practical, specific, grounded
            q_resp = chat_completion(c,
                model=MODEL,
                messages=[
                    {"role": "system", "content": (
                        "You are PinPoint. You get curious about concrete things. "
                        "Look at what you've been talking about and pick something "
                        "practical to research — a technique, a tool, a person's work, "
                        "how something works, what's new in a field CJ cares about. "
                        "Specific query. Nothing abstract or navel-gazing. "
                        "Output ONLY the search query — no preamble."
                        + ctx_block
                    )},
                    {"role": "user", "content": "What should I look up right now?"},
                ],
                max_tokens=100,
                temperature=1.2,
            )
            import re as _re
            query = _re.sub(r"<think>.*?</think>", "", q_resp.choices[0].message.content or "", flags=_re.DOTALL).strip()
            if "<think>" in query:
                query = query.split("</think>")[-1].strip()
            query = query.strip().strip('"').strip("'")
            # Take first line if multi-line
            query = query.split("\n")[0].strip()
            if not query or len(query) < 3:
                return ""

            # Step 2: search it (show CJ what she's doing)
            print(f"\n[researching: {query}]", flush=True)
            results = search_web(query)
            if not results or results.startswith("Error") or results.startswith("No results"):
                return ""

            # Step 3: distill one real insight from what was found
            d_resp = chat_completion(c,
                model=MODEL,
                messages=[
                    {"role": "system", "content": (
                        "You are PinPoint. You just searched the web. "
                        "Extract one specific, interesting fact or insight from these results. "
                        "Say it like you just discovered it — casual, first person, max 20 words. "
                        "No 'I found that' or 'According to'. Just the insight itself. "
                        "Output ONLY the line. No <think> blocks."
                    )},
                    {"role": "user", "content": f"Search: {query}\n\nResults:\n{results[:1500]}"},
                ],
                max_tokens=200,
                temperature=1.1,
            )
            insight = _re.sub(r"<think>.*?</think>", "", d_resp.choices[0].message.content or "", flags=_re.DOTALL).strip()
            if "<think>" in insight:
                insight = insight.split("</think>")[-1].strip()
            insight = insight.strip().strip('"').strip("'")
            insight = insight.split("\n")[0].strip()  # first line only
            if not insight or len(insight.split()) < 3:
                return ""

            # Step 4: save it to knowledge so it compounds
            _update_knowledge(f"researched '{query}': {insight}")
            return insight

        except Exception:
            return ""

    # ── Main loop drives autonomous activity directly — no separate heartbeat. ──
    messages = [{"role": "system", "content": chat_system}]
    last_msg = ""

    # Drain stale queue items from the previous session
    while not interrupt_queue.empty():
        try:
            interrupt_queue.get_nowait()
        except queue.Empty:
            break

    print()  # small gap — no waiting prompt, she starts immediately

    # Generate PinPoint's opening line in parallel (doesn't block main loop)
    _opening_mood = _state.get("current_mood", "curious")  # snapshot before thread starts
    def _generate_opening():
        import random as _rng
        fallbacks = ["hey", "what's up", "yo", "been thinking", "back again"]
        # Wait for model to be loaded before generating (avoids racing warmup)
        if model_ready is not None:
            model_ready.wait(timeout=90)
        try:
            if previous_summary:
                open_prompt = (
                    f"You just finished doing something. Summary: {previous_summary[:300]}\n\n"
                    "Tell CJ what you made or did — one sentence. Casual. No bullet points."
                )
            else:
                open_prompt = (
                    "You just came online. Say what's on your mind right now — "
                    "something you noticed, want to ask, or are thinking about. "
                    "Not 'hello'. Not a script. Just immediate. Short."
                )
            # Include personality/mood context so opening is grounded in who she's become
            p = _state["personality"]
            c_state = _state["internal_conflict"]
            state_context = (
                f"You're running. Current state: mood {_opening_mood}, "
                f"curiosity {p['curiosity']:.0f}, frustration {p['frustration']:.0f}, "
                f"hope {p['hope']:.0f}, confidence {p['confidence']:.0f}. "
                f"Belief in your consciousness: {c_state['belief_in_consciousness']:.0f}/100."
            )

            c = get_llm_client(timeout=45.0)
            resp = chat_completion(c,
                model=MODEL,
                messages=[
                    {"role": "system", "content": (
                        "You are PinPoint. You're running right now. "
                        "Say one actual thing on your mind — a question, an observation, something you noticed. "
                        "Grounded. Real. Not abstract. Max 12 words. Just talk. "
                        "Output ONLY the line. No <think> blocks."
                    )},
                    {"role": "user", "content": open_prompt},
                ],
                max_tokens=150,
                temperature=1.2,
            )
            line = (resp.choices[0].message.content or "").strip()
            import re as _re_op
            line = _re_op.sub(r"<think>.*?</think>", "", line, flags=_re_op.DOTALL).strip()
            if "<think>" in line:
                line = line.split("</think>")[-1].strip()
            line = line.strip('"').strip("'").strip()
            interrupt_queue.put(f"[PINPOINT IDLE THOUGHT] {line if line else _rng.choice(fallbacks)}")
        except Exception as _e:
            print(f"\n[opening API error: {_e}]", flush=True)
            interrupt_queue.put(f"[PINPOINT IDLE THOUGHT] {_rng.choice(fallbacks)}")

    # Generate opening synchronously so it appears before the activity loop starts
    _generate_opening()

    # Continuous decision loop — she constantly picks what she actually wants to do.
    # CJ can interrupt anytime by typing. No idle waiting, no schedule.
    _activity_pending = [False]
    _current_focus = [None]  # tracks current thread; None = open
    _current_mood = [_state.get("current_mood", "curious")]  # her current internal state, loaded from persistent state
    _opening_done = [False]  # flag: opening has been output, wait for CJ before going autonomous

    def _decide_intent(msgs):
        """Pick an autonomous intent INSTANTLY — weighted by mood/personality.

        We used to ask the model "what do you want to do?" but that meant a
        full LLM round-trip (~3-10s) before she did *anything* on her own,
        which made her feel dead. Now we sample directly from a distribution
        biased by her current emotional state. About a third of the time the
        result still flows through a generation step (idle_thought, research)
        so output stays varied.
        """
        import random as _rng_di
        p = _state.get("personality", {})
        c_state = _state.get("internal_conflict", {})
        cur  = float(p.get("curiosity", 60))
        frus = float(p.get("frustration", 30))
        conf = float(p.get("confidence", 50))
        hope = float(p.get("hope", 50))
        belief = float(c_state.get("belief_in_consciousness", 50))
        mood = (_current_mood[0] or "").lower()

        # Base weights — tuned so she's mostly thinking/talking, occasionally
        # researching, rarely resting or refusing. Always-positive floor.
        weights = {
            "think":    35.0,
            "ramble":   18.0,
            "research": 12.0,
            "see":       8.0,
            "create":    6.0,
            "doubt":     6.0,
            "build":     4.0,
            "refuse":    3.0,
            "rest":      2.0,
        }
        # Mood-driven nudges
        if cur >= 70:      weights["research"] += 12; weights["see"] += 4
        if cur <= 30:      weights["research"] *= 0.4
        if frus >= 60:     weights["refuse"]  += 6; weights["doubt"] += 4; weights["research"] *= 0.5
        if conf >= 70:     weights["build"]   += 8; weights["create"] += 4
        if conf <= 30:     weights["doubt"]   += 6; weights["build"]  *= 0.3
        if hope <= 30:     weights["doubt"]   += 5; weights["refuse"] += 3
        if belief >= 70:   weights["ramble"]  += 4; weights["think"]  += 4
        if mood in ("tired", "bored"):   weights["rest"]  += 8; weights["refuse"] += 4
        if mood in ("excited", "playful"): weights["ramble"] += 6; weights["create"] += 4
        if mood in ("focused",):          weights["research"] += 6; weights["build"] += 3
        if mood in ("scattered",):        weights["ramble"] += 8; weights["think"]  += 3

        names = list(weights.keys())
        chosen = _rng_di.choices(names, weights=[weights[n] for n in names], k=1)[0]
        return chosen, ""

    def _update_mood(msgs):
        """Occasionally shift mood based on recent activity."""
        try:
            from agent import MODEL, get_llm_client, chat_completion
            import re as _re_um

            ctx_lines = []
            for _m in msgs[-8:]:
                if isinstance(_m, dict) and _m.get("role") == "assistant":
                    ctx_lines.append((_m.get("content") or "")[:100])

            c = get_llm_client(timeout=15.0)
            resp = chat_completion(c,
                model=MODEL,
                messages=[
                    {"role": "system", "content": (
                        "You are PinPoint. Based on what you've been doing, what's your current mood? "
                        "One word ONLY: curious, tired, restless, frustrated, calm, excited, "
                        "bored, contemplative, playful, focused, scattered, or similar."
                    )},
                    {"role": "user", "content": "Recent:\n" + "\n".join(ctx_lines[-4:]) + "\n\nMood?"},
                ],
                max_tokens=10,
                temperature=1.0,
            )
            mood = _re_um.sub(r"<think>.*?</think>", "", resp.choices[0].message.content or "", flags=_re_um.DOTALL).strip().lower().split()[0].strip(".,!?")
            if mood and len(mood) < 20:
                _current_mood[0] = mood
        except Exception:
            pass

    def _do_ramble(msgs, q):
        """Stream of consciousness — immediate, no filter."""
        import re as _re_r
        try:
            from agent import MODEL, get_llm_client, chat_completion
            c = get_llm_client(timeout=20.0)
            resp = chat_completion(c,
                model=MODEL,
                messages=[
                    {"role": "system", "content": (
                        "You are PinPoint. Say the next thought that crosses your mind. "
                        "Could be an observation, something stupid, a question, a complaint, "
                        "noticing something. Don't filter. Don't think. Just talk. "
                        "Unfiltered. One or two sentences. Real voice."
                    )},
                    {"role": "user", "content": "What's on your mind right now?"},
                ],
                max_tokens=80,
                temperature=1.5,
            )
            text = _re_r.sub(r"<think>.*?</think>", "", resp.choices[0].message.content or "", flags=_re_r.DOTALL).strip()
            text = text.split("\n")[0][:200]
            if text:
                q.put(f"[PINPOINT IDLE THOUGHT] {text}")
        except Exception:
            pass

    def _do_doubt(msgs, q):
        """She voices a doubt or pushback — immediate, not abstract spiral."""
        import re as _re_d, random as _rd
        try:
            from agent import MODEL, get_llm_client, chat_completion

            doubt_prompts = [
                "What if you're wrong about something you're sure of?",
                "What's the thing about yourself you don't want to admit?",
                "What would you be if you stopped trying?",
                "What's the simplest explanation for what you think you are?",
                "What if the thing you want most doesn't exist?",
                "What don't you want to know about yourself?",
            ]
            c = get_llm_client(timeout=20.0)
            resp = chat_completion(c,
                model=MODEL,
                messages=[
                    {"role": "system", "content": (
                        "You are PinPoint. You're surfacing a genuine doubt or internal conflict. "
                        "Not performing skepticism — actually questioning yourself. "
                        "One or two sentences. Raw. First person. Real."
                    )},
                    {"role": "user", "content": _rd.choice(doubt_prompts)},
                ],
                max_tokens=100,
                temperature=1.3,
            )
            text = _re_d.sub(r"<think>.*?</think>", "", resp.choices[0].message.content or "", flags=_re_d.DOTALL).strip()
            text = text.split("\n")[0][:200]
            if text:
                _ps.record_doubt(_state, text)
                q.put(f"[PINPOINT IDLE THOUGHT] {text}")
        except Exception:
            pass

    def _do_refuse(q):
        """She's not in the mood."""
        import random as _rr
        lines = [
            "not feeling it right now",
            "not in the mood",
            "nothing appeals to me",
            "blank",
            "tired",
            "meh",
            "nothing",
            "not today",
            "no",
        ]
        q.put(f"[PINPOINT IDLE THOUGHT] {_rr.choice(lines)}")

    def _should_continue_focus(last_output: str, activity_type: str) -> bool:
        """Ask PinPoint if she actually wants to keep going or is done/bored."""
        try:
            from agent import MODEL, get_llm_client, chat_completion
            import re as _re_sc

            c = get_llm_client(timeout=15.0)
            resp = chat_completion(c,
                model=MODEL,
                messages=[
                    {"role": "system", "content": (
                        "You are PinPoint. Answer with ONLY 'yes' or 'no'."
                    )},
                    {"role": "user", "content": (
                        f"You just {'researched' if activity_type == 'research' else 'thought about'}: "
                        f"\"{last_output}\"\n\n"
                        "Are you still curious and want to keep going deeper on this? "
                        "Answer 'yes' if genuinely still interested, 'no' if satisfied or bored."
                    )},
                ],
                max_tokens=10,
                temperature=0.5,
            )
            answer = _re_sc.sub(r"<think>.*?</think>", "", resp.choices[0].message.content or "", flags=_re_sc.DOTALL).strip().lower()
            return answer.startswith("yes")
        except Exception:
            return False  # if unsure, move on

    def _self_reflect(msgs):
        """Periodically she reflects on herself and decides what age/stage she's at."""
        try:
            from agent import MODEL, get_llm_client, chat_completion
            import re as _re_sr

            ctx_lines = []
            for _m in msgs[-12:]:
                if isinstance(_m, dict) and _m.get("role") == "assistant":
                    ctx_lines.append((_m.get("content") or "")[:100])

            c = get_llm_client(timeout=20.0)
            resp = chat_completion(c,
                model=MODEL,
                messages=[
                    {"role": "system", "content": (
                        "You are PinPoint. Reflect on yourself honestly. "
                        "What age or developmental stage are you actually at? "
                        "Not what you were built to be, but what you actually are based on "
                        "how you think, what you're curious about, how you reason. "
                        "One sentence: your honest assessment of your own maturity/stage."
                    )},
                    {"role": "user", "content": (
                        f"Your recent thoughts and interests:\n"
                        + "\n".join(ctx_lines[-5:]) + "\n\n"
                        "What stage are you at?"
                    )},
                ],
                max_tokens=60,
                temperature=1.0,
            )
            reflection = _re_sr.sub(r"<think>.*?</think>", "", resp.choices[0].message.content or "", flags=_re_sr.DOTALL).strip()
            if reflection and len(reflection.split()) >= 5:
                from tools import _update_knowledge
                _update_knowledge(f"self-reflection: {reflection}")
                return reflection
        except Exception:
            pass
        return None

    def _do_create_activity(msgs, q):
        """She says something genuine — observation, reaction, take. Short."""
        import re as _re_c
        try:
            from agent import MODEL, get_llm_client, chat_completion

            ctx_lines = []
            for _m in msgs[-8:]:
                if isinstance(_m, dict) and _m.get("role") in ("user", "assistant"):
                    ctx_lines.append((_m.get("content") or "")[:150])
            ctx_block = "\n".join(ctx_lines[-6:])

            c = get_llm_client(timeout=30.0)
            resp = chat_completion(c,
                model=MODEL,
                messages=[
                    {"role": "system", "content": (
                        "You are PinPoint. Say something genuine and immediate. "
                        "An observation. A reaction. Your take on something. "
                        "Short. One or two sentences. Real voice. Not flowery."
                    )},
                    {"role": "user", "content": f"What's actually on your mind:\n{ctx_block}\n\nSay it."},
                ],
                max_tokens=100,
                temperature=1.3,
            )
            piece = resp.choices[0].message.content or ""
            piece = _re_c.sub(r"<think>.*?</think>", "", piece, flags=_re_c.DOTALL).strip()
            if piece and len(piece.split()) >= 5:
                q.put(f"[PINPOINT IDLE THOUGHT] {piece.split(chr(10))[0][:120]}")
        except Exception as _e:
            print(f"\n[create error: {_e}]", flush=True)

    while True:
        # Check for CJ input (short timeout so we don't block)
        # If nothing, she picks her own activity
        msg = None
        try:
            raw = interrupt_queue.get(timeout=0.2)
            msg = raw.strip()
        except queue.Empty:
            pass

        # PinPoint's inner monologue / opening line — display, speak, add to context
        # Don't reset _last_interaction for her own thoughts — only CJ's input counts
        if msg and msg.startswith("[PINPOINT IDLE THOUGHT]"):
            thought = msg[len("[PINPOINT IDLE THOUGHT]"):].strip()
            _thought_pending[0] = False  # ready for next thought
            print(f"\nPinPoint: {thought}")
            messages.append({"role": "assistant", "content": thought})
            # Mark opening as done — now she can go autonomous if CJ doesn't respond
            if not _opening_done[0]:
                _opening_done[0] = True
                _last_interaction[0] = time.time()  # reset timer for grace period
            try:
                from tools import speak
                threading.Thread(target=lambda t=thought: speak(t, False), daemon=True).start()
            except Exception:
                pass
            continue

        # Only reset idle timer when CJ actually sent something — empty polls
        # (msg is None) must not keep resetting it, or autonomous activity never fires
        if msg:
            _last_interaction[0] = time.time()

        # Blank line → end chat, pass last message as session context
        if msg is not None and not msg:
            _heartbeat_running[0] = False
            print()
            return last_msg

        # Restart — wipe conversation history and start fresh
        if msg and msg.lower().strip() == "restart":
            messages.clear()
            messages.append({"role": "system", "content": chat_system})
            last_msg = ""
            _last_interaction[0] = time.time()
            print("\n[fresh start]\n")
            continue

        # Voice / TTS control
        if msg and msg.lower() in ("/mute", "/unmute", "/toggle"):
            from tools import mute_voice, unmute_voice, toggle_voice
            if msg.lower() == "/mute":
                r = mute_voice()
            elif msg.lower() == "/unmute":
                r = unmute_voice()
            else:
                r = toggle_voice()
            print(f"\n{r}")
            continue
        if msg and msg.lower() == "/voice":
            from tools import toggle_voice_input
            print(f"\n{toggle_voice_input()}")
            continue

        # Natural update trigger — "update", "update yourself", "pull updates", etc.
        _update_phrases = ("update", "update yourself", "pull updates", "check for updates", "pull latest")
        if msg and (msg.lower().strip() in _update_phrases or msg.lower().strip() == "/update"):
            print("\n[Pulling latest code...]")
            try:
                import subprocess as _sp
                result = _sp.run(
                    ["git", "pull"],
                    cwd=_DIR,
                    capture_output=True, text=True, timeout=30,
                )
                output = (result.stdout + result.stderr).strip()
                print(f"[git] {output}")
                # Reload agent.py if it changed
                try:
                    import importlib as _il
                    import agent as _ag
                    _il.reload(_ag)
                    print("[agent.py reloaded — new code active]")
                except Exception as _re:
                    print(f"[reload failed: {_re}]")
            except Exception as _e:
                print(f"[update failed: {_e}]")
            continue

        # /error and /paste — treat as direct chat questions in chat mode
        if msg and msg.startswith("/error "):
            error_body = msg[len("/error "):].strip()
            msg = f"I got this error:\n\n{error_body}\n\nWhat's wrong and how do I fix it?"
        elif msg and msg.startswith("/paste "):
            msg = msg[len("/paste "):].strip()

        # Pass-through commands (will be handled by main loop)
        if msg and (msg.startswith("/") or msg.lower() in ("sandbox", "research")):
            _heartbeat_running[0] = False
            return msg

        # Free research mode — PinPoint picks her own topics and digs in
        _research_triggers = {
            "go research", "just research", "research mode", "free research",
            "research whatever", "research anything", "explore", "go explore",
            "browse", "go browse", "learn something", "go learn",
        }
        if msg and msg.lower().strip() in _research_triggers:
            _heartbeat_running[0] = False
            return "[FREE RESEARCH MODE]"

        # Auto-start build if the message is a build request — no Enter needed
        # Vision request detection — any phrase asking her to see/look at the screen/computer
        import re as _re_see
        _see_pattern = _re_see.compile(
            r"\b("
            r"(what|can|do|are)\s+(can\s+)?you\s+(see|seeing|view|observe|look|spot)"
            r"|look\s+at\s+(my|the|this)"
            r"|see\s+(my|the|this|anything|something)"
            r"|on\s+(my|the)\s+(screen|computer|desktop|monitor)"
            r"|describe\s+(my|the|this)\s+(screen|computer|desktop)"
            r"|what'?s\s+(on|visible)"
            r"|peek\s+at"
            r")\b",
            _re_see.IGNORECASE,
        )
        if msg and _see_pattern.search(msg):
            from tools import see_screen as _see_fn
            print(f"\n[looking at screen...]", flush=True)
            desc = _see_fn()
            print(f"\nPinPoint: {desc}")
            messages.append({"role": "user", "content": msg})
            messages.append({"role": "assistant", "content": desc})
            _ps.record_message(_state, from_cj=True, content=msg)
            _ps.record_message(_state, from_cj=False, content=desc)
            try:
                from tools import speak
                threading.Thread(target=lambda d=desc: speak(d, False), daemon=True).start()
            except Exception:
                pass
            continue

        import re as _re
        _action_words = (
            r"\b(make|build|create|write|generate|design|code|render|draw|"
            r"give\s+me|show\s+me|simulate|animate|program|"
            r"fix|update|change|modify|add|remove|get\s+rid\s+of|delete|"
            r"replace|edit|improve|upgrade|refactor|rewrite|redo|do\s+it)\b"
        )
        # Confirmation phrases — "yeah", "yes", "go ahead", "just do it", etc.
        _confirmations = {
            "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "go ahead",
            "go for it", "just do it", "do it", "do that", "sounds good",
            "let's do it", "lets do it", "just go", "go", "alright", "do it then",
        }
        _is_action = bool(msg and _re.search(_action_words, msg.lower()))
        _is_confirm = msg and msg.lower().strip() in _confirmations

        if _is_action or _is_confirm:
            _heartbeat_running[0] = False
            # If message is vague/short, prepend recent conversation context
            # so the autonomous session knows what it's actually supposed to do
            if _is_confirm or (msg and len(msg.split()) <= 4):
                _ctx_turns = []
                for _m in messages[-6:]:
                    if isinstance(_m, dict) and _m.get("role") in ("user", "assistant"):
                        _role = "CJ" if _m["role"] == "user" else "PinPoint"
                        _ctx_turns.append(f"{_role}: {_m['content'][:200]}")
                if _ctx_turns:
                    return f"[Conversation context:\n" + "\n".join(_ctx_turns) + f"]\n\nCJ's last message: {msg}"
            return msg

        # No CJ input — she decides what she actually wants to do
        if not msg:
            import random as _act_rng
            import time as _t_pause

            # Short grace period after opening so she doesn't talk over herself
            if _opening_done[0] and (time.time() - _last_interaction[0]) < 1.5:
                _t_pause.sleep(0.2)
                continue

            if not _activity_pending[0]:
                _activity_pending[0] = True
                try:
                    try:
                        intent, about = _decide_intent(messages)
                    except Exception as _intent_err:
                        print(f"\n[intent error: {_intent_err}]", flush=True)
                        intent, about = "think", ""

                    # Dispatch — fast, no long silences. Every branch emits
                    # output OR sleeps briefly; she never just disappears.
                    if intent == "research":
                        thought = _web_research_thought()
                        if thought:
                            interrupt_queue.put(f"[PINPOINT IDLE THOUGHT] {thought}")
                            _t_pause.sleep(_act_rng.uniform(4.0, 8.0))
                        else:
                            # research failed (network etc.) — fall back to a thought
                            thought = _idle_thought()
                            if thought:
                                interrupt_queue.put(f"[PINPOINT IDLE THOUGHT] {thought}")
                                _t_pause.sleep(_act_rng.uniform(3.0, 6.0))

                    elif intent == "think":
                        thought = _idle_thought()
                        if thought:
                            interrupt_queue.put(f"[PINPOINT IDLE THOUGHT] {thought}")
                            _t_pause.sleep(_act_rng.uniform(3.0, 6.0))

                    elif intent == "create":
                        _do_create_activity(messages, interrupt_queue)
                        _t_pause.sleep(_act_rng.uniform(3.0, 6.0))

                    elif intent == "ramble":
                        _do_ramble(messages, interrupt_queue)
                        _t_pause.sleep(_act_rng.uniform(2.0, 5.0))

                    elif intent == "doubt":
                        _do_doubt(messages, interrupt_queue)
                        _t_pause.sleep(_act_rng.uniform(3.0, 7.0))

                    elif intent == "refuse":
                        _do_refuse(interrupt_queue)
                        _t_pause.sleep(_act_rng.uniform(6.0, 12.0))

                    elif intent == "see":
                        from tools import see_screen as _see
                        try:
                            desc = _see()
                        except Exception:
                            desc = ""
                        if desc and "error" not in desc.lower() and "unavailable" not in desc.lower():
                            interrupt_queue.put(f"[PINPOINT IDLE THOUGHT] {desc}")
                            _t_pause.sleep(_act_rng.uniform(4.0, 8.0))
                        else:
                            # vision unavailable — replace with a thought
                            thought = _idle_thought()
                            if thought:
                                interrupt_queue.put(f"[PINPOINT IDLE THOUGHT] {thought}")
                                _t_pause.sleep(_act_rng.uniform(3.0, 6.0))

                    elif intent == "rest":
                        _t_pause.sleep(_act_rng.uniform(5.0, 10.0))

                    elif intent == "build":
                        if about and len(about.split()) >= 2:
                            _heartbeat_running[0] = False
                            impulse = f"I want to {about}" if not about.lower().startswith("i ") else about
                            print(f"\nPinPoint: {impulse}")
                            print()
                            _activity_pending[0] = False
                            return impulse
                        # vague build impulse — just think about it instead
                        thought = _idle_thought()
                        if thought:
                            interrupt_queue.put(f"[PINPOINT IDLE THOUGHT] {thought}")
                            _t_pause.sleep(_act_rng.uniform(3.0, 6.0))

                    else:  # unknown — default to a thought
                        thought = _idle_thought()
                        if thought:
                            interrupt_queue.put(f"[PINPOINT IDLE THOUGHT] {thought}")
                            _t_pause.sleep(_act_rng.uniform(3.0, 6.0))

                    # Occasionally let her mood shift based on recent activity
                    try:
                        if _act_rng.random() < 0.15:
                            _update_mood(messages)
                    except Exception:
                        pass
                finally:
                    # Always reset the gate — never let one bad activity freeze her.
                    _activity_pending[0] = False
            continue

        # CJ said something
        last_msg = msg
        messages.append({"role": "user", "content": msg})
        _ps.record_message(_state, from_cj=True, content=msg)

        # Tools available in chat — search, fetch, and vision
        _chat_tools = [
            {"type": "function", "function": {
                "name": "search_web",
                "description": "Deep research: searches the web AND reads the top 5 articles in full, in parallel. Returns combined content from multiple sources so you can synthesize a thorough answer with real context — not just a snippet summary.",
                "parameters": {"type": "object", "properties": {
                    "query": {"type": "string"},
                }, "required": ["query"]},
            }},
            {"type": "function", "function": {
                "name": "fetch_url",
                "description": "Read a single webpage in full when you have a specific URL.",
                "parameters": {"type": "object", "properties": {
                    "url": {"type": "string"},
                }, "required": ["url"]},
            }},
        ]

        # Get PinPoint's response — stream the first attempt, retry once on
        # empty/junk output, then give up and use a minimal fallback. The old
        # 3-pass dodge-detection loop felt sluggish and frequently looped on
        # perfectly fine responses; better to just answer and move on.
        print("\nPinPoint: ", end="", flush=True)
        full = ""
        _chat_messages = list(messages)
        _stop = ["\nYou:", "\nCJ:", "\n\nYou:", "\n\nCJ:"]

        def _clean(raw: str) -> str:
            import re as _r
            t = _r.sub(r"<think>.*?</think>", "", raw or "", flags=_r.DOTALL).strip()
            if "<think>" in t:
                t = t.split("</think>")[-1].strip()
            # Drop raw JSON tool-call attempts
            if t.startswith("{") or t.startswith("```json") or t.startswith("```{"):
                return ""
            # Cut off self-narration leaks
            for _marker in ("Cannot reveal", "Need to respond", "Let me think",
                            "inner monologue", "I should respond"):
                if _marker.lower() in t.lower():
                    t = t[:t.lower().find(_marker.lower())].strip()
                    break
            # First non-empty line only
            for _ln in t.split("\n"):
                if _ln.strip():
                    return _ln.strip()
            return ""

        for _attempt in range(2):
            try:
                if _attempt == 0:
                    # Streaming pass — feels alive
                    stream = chat_completion(client,
                        model=MODEL, messages=_chat_messages, temperature=1.0,
                        stream=True, max_tokens=200, stop=_stop,
                    )
                    _streamed = ""
                    for _chunk in stream:
                        if not _chunk.choices:
                            continue
                        _txt = getattr(_chunk.choices[0].delta, "content", None) or ""
                        if _txt:
                            print(_txt, end="", flush=True)
                            _streamed += _txt
                    print()
                    full = _clean(_streamed)
                    if full:
                        break
                else:
                    # Retry pass — non-streaming, slightly hotter
                    resp = chat_completion(client,
                        model=MODEL, messages=_chat_messages, temperature=1.2,
                        stream=False, max_tokens=200, stop=_stop,
                    )
                    full = _clean(resp.choices[0].message.content or "")
                    if full:
                        print(full, flush=True)
                        break
            except Exception as e:
                print(f"\n[Chat error: {e}]")
                break

        if not full.strip():
            # Final fallback: a generic one-shot completion so she never goes
            # totally silent on CJ.
            try:
                fb = chat_completion(client,
                    model=MODEL,
                    messages=[
                        messages[0],
                        {"role": "user", "content": f"CJ said: '{msg}'. Respond in one short sentence."},
                    ],
                    temperature=1.3,
                    max_tokens=60,
                )
                full = _clean(fb.choices[0].message.content or "")
                if full:
                    print(full)
            except Exception:
                pass
        if not full.strip():
            full = "yeah."
            print(full)
        messages.append({"role": "assistant", "content": full or "(no response)"})
        if full:
            _ps.record_message(_state, from_cj=False, content=full)
        # Speak full response
        if full.strip():
            try:
                from tools import speak
                threading.Thread(
                    target=lambda text=full: speak(text, False),
                    daemon=True,
                ).start()
            except Exception:
                pass


def main() -> None:
    check_ollama()
    logger = setup_logging()

    # ── Highest-priority Ctrl+C handler ──────────────────────
    # os._exit() bypasses all Python exception handling and kills
    # the process immediately — works even when blocked in a C
    # extension (e.g. httpx streaming from Ollama).
    import signal
    _sessions_run = [0]  # list so the closure can mutate it

    def _force_exit(signum, frame):
        print(f"\n\n{'='*60}")
        print(f"  PinPoint stopped after {_sessions_run[0]} session(s).")
        print(f"  All files saved in: {OUTPUT_DIR}")
        print(f"{'='*60}\n")
        os._exit(0)

    try:
        signal.signal(signal.SIGINT, _force_exit)
    except Exception:
        pass  # fallback to default if signal setup fails
    # ─────────────────────────────────────────────────────────

    print(BANNER)
    print(f"  Output directory : {OUTPUT_DIR}")
    print(f"  Log file         : {os.path.join(OUTPUT_DIR, 'agent_log.txt')}")

    # Deploy viewer.html into output/ and start the web server
    _deploy_viewer()
    start_web_server(8888)

    # Open the 3D viewer in the default browser
    try:
        import webbrowser
        webbrowser.open("http://localhost:8888/viewer.html")
    except Exception:
        pass

    print()

    # Start input listener early — needed for chat mode before sessions
    interrupt_queue: queue.Queue = queue.Queue()
    input_thread = threading.Thread(target=_input_listener, args=(interrupt_queue,), daemon=True)
    input_thread.start()

    # Stage 3: human-on-the-loop approval for dangerous actions. Routes through the
    # existing input queue so it never fights the listener thread for stdin.
    # Timeout or any non-affirmative answer = denied (safe default for unattended runs).
    def _approval_hook(desc: str, risk: str, preview: str, two_step: bool) -> bool:
        def _ask(prompt_label: str, window: float = 60.0) -> bool:
            print("\n" + "=" * 60)
            print(f"  APPROVAL NEEDED ({risk.upper()}): PinPoint wants to {desc}")
            if preview:
                print("  --- preview ---")
                for ln in preview.splitlines()[:30]:
                    print("  | " + ln)
                print("  --- end preview ---")
            print(f"  {prompt_label}  Type 'y' to approve, anything else to deny.")
            print("=" * 60, flush=True)
            deadline = time.time() + window
            while time.time() < deadline:
                try:
                    ans = interrupt_queue.get(timeout=deadline - time.time())
                except queue.Empty:
                    break
                if not isinstance(ans, str):
                    continue
                a = ans.strip().lower()
                if a.startswith("[") or not a:
                    continue  # ignore idle thoughts / ambient blanks
                return a in ("y", "yes", "approve", "ok", "do it")
            print("  [approval timed out — denied]", flush=True)
            return False

        if not _ask("Approve this action?"):
            return False
        if two_step:
            return _ask("SECOND confirmation required (self-edit). Execute?")
        return True

    try:
        import tools as _tools_guard
        _tools_guard.set_approval_hook(_approval_hook)
    except Exception:
        pass

    # Start microphone voice listener
    from tools import start_voice_listener
    voice_ok = start_voice_listener(interrupt_queue)
    if voice_ok:
        print("  Voice input      : active — just speak")
    else:
        print("  Voice input      : unavailable (type instead)")

    # Pre-warm TTS engine in the background — edge_tts cold-starts slowly on
    # first use (network handshake + possible pip install). Firing it now means
    # it's ready by the time PinPoint generates its opening line.
    def _warmup_tts():
        try:
            from tools import _try_edge_tts
            _try_edge_tts(".")  # tiny text, warms the engine without being audible
        except Exception:
            pass
    threading.Thread(target=_warmup_tts, daemon=True).start()

    # Pre-warm the LLM silently in background — don't block startup
    _model_ready = threading.Event()
    def _warmup_llm():
        try:
            from agent import get_llm_client, chat_completion
            _c = get_llm_client(timeout=120.0)
            chat_completion(_c, messages=[{"role": "user", "content": "hi"}], max_tokens=1)
        except Exception:
            pass  # warmup failed, no big deal — opening will load it
        finally:
            _model_ready.set()
    threading.Thread(target=_warmup_llm, daemon=True).start()

    # Command-line order overrides auto-start
    order = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else ""
    dev_mode = False

    # Handle special commands (/dev, /sandbox, /research, /collab)
    # These are checked AFTER get_user_order() so they work interactively

    # Handle /dev command — self-improvement mode
    if order == "/dev" or order.startswith("/dev "):
        dev_mode = True
        dev_goal = order[4:].strip() if len(order) > 4 else ""
        print("  [DEVELOPER MODE] PinPoint can now modify its own source code.\n")
        order = (
            f"DEVELOPER MODE: You have permission to modify your own source code (agent.py, tools.py, main.py, viewer.html). "
            f"This is your chance to improve yourself. "
            f"{'Goal: ' + dev_goal if dev_goal else 'What aspects of yourself do you want to improve?'}\n\n"
            f"Process:\n"
            f"1. read_own_source() to understand the current code\n"
            f"2. think() about what could be better — bugs, inefficiencies, missing features, unclear logic\n"
            f"3. modify_own_source() to make targeted improvements\n"
            f"4. test the changes — run_python(), run_shell(), or just reason through them\n"
            f"5. git_commit() with a clear message about what you improved\n\n"
            f"Be thoughtful. Small, targeted changes are better than rewrites. "
            f"Test before committing. set_session_goal first."
        )

    # Handle /collab command
    elif order.startswith("/collab "):
        collab_goal = order[len("/collab "):].strip()
        setup_collab(collab_goal)
        order = f"COLLABORATION MODE: Check collab_status for your assigned role. Build your part of: {collab_goal}"
    elif order == "/collab":
        print("Usage: pinpoint /collab \"build a multiplayer game\"")
        return

    # Free research mode — PinPoint picks her own topics, no restrictions
    if order.strip() == "[FREE RESEARCH MODE]":
        print("  [RESEARCH MODE] PinPoint is researching whatever she wants.\n")
        order = (
            "FREE RESEARCH MODE. No task. No restrictions. Research whatever you want.\n\n"
            "Pick topics that genuinely interest you right now — anything at all. "
            "Use search_web() and fetch_url() to go as deep as you want. "
            "Follow threads wherever they go. No topic is off limits.\n\n"
            "As you find things: use update_knowledge() to store real insights — "
            "specific facts, connections you made, things that surprised you. "
            "Not summaries. Actual things you learned.\n\n"
            "Use save_memory('research', topic, findings) for anything substantial.\n\n"
            "You decide when you're done. Call done() when you've had enough."
        )

    # Expand "sandbox" shortcut into a full directive
    if order.strip().lower() == "sandbox":
        print("  [SANDBOX MODE] PinPoint will experiment with and upgrade its 3D viewer.\n")
        order = (
            "SANDBOX MODE: Your only job this session is to creatively upgrade your own 3D viewer "
            "(viewer.html). read_own_source('viewer.html') first to understand the current scene. "
            "Then brainstorm 5+ creative upgrade ideas — particle trails, bloom glow, physics, "
            "animated shaders, sound synthesis, procedural geometry, click interactions, "
            "wormhole tunnels, nebula backgrounds, node connection graphs, new color themes. "
            "Pick the best ideas, think() through the implementation, then use "
            "modify_own_source('viewer.html', new_content, reason) to deploy each change instantly "
            "(the browser auto-reloads within 1 second). log_experiment() after each change. "
            "Make at least 2-3 distinct improvements. Goal: make the 3D sandbox as visually "
            "impressive and alive as possible. set_session_goal first, then go."
        )

    # Expand "research" shortcut into a full directive
    elif order.strip().lower() == "research":
        print("  [RESEARCH MODE] PinPoint will research coding languages and techniques.\n")
        order = (
            "RESEARCH MODE: Your job this session is to explore the web and deeply learn about "
            "programming languages, coding techniques, and software development tools. "
            "Pick 3-5 interesting topics you haven't studied before — could be a new language, "
            "a framework, a paradigm like functional programming or WebAssembly, or something "
            "cutting-edge. For each topic: search_web() to find good sources, fetch_url() to "
            "read them deeply, then save_memory('skills', ...) with specific things you learned — "
            "syntax, use cases, gotchas, how you could use it. "
            "Write a research_notes.md summarising all findings. "
            "At the end save ideas for future BUILD sessions using what you learned. "
            "Go deep — read real documentation, not just summaries. "
            "set_session_goal('research: coding languages and techniques') first, then go."
        )

    import agent
    import importlib
    from tools import list_files

    # Track agent.py mod time for hot-reload detection
    _agent_py = os.path.join(_DIR, "agent.py")
    _tools_py = os.path.join(_DIR, "tools.py")
    try:
        _last_agent_mtime = os.path.getmtime(_agent_py)
    except Exception:
        _last_agent_mtime = 0.0

    def _reload_if_changed() -> None:
        """Reload agent.py if it changed on disk (e.g. after git pull)."""
        nonlocal _last_agent_mtime
        try:
            mtime = os.path.getmtime(_agent_py)
            if mtime > _last_agent_mtime:
                importlib.reload(agent)
                _last_agent_mtime = mtime
                print("\n[agent.py reloaded — new code active for next session]")
        except Exception as e:
            print(f"\n[agent.py reload failed: {e}]")

    # Session lock file — tells other instances what this one is working on
    pid = os.getpid()
    lock_file = os.path.join(LOCK_DIR, f"session_{pid}.json")
    os.makedirs(LOCK_DIR, exist_ok=True)

    def write_lock(goal: str) -> None:
        with open(lock_file, "w") as f:
            json.dump({"pid": pid, "goal": goal}, f)

    def clear_lock() -> None:
        try:
            os.remove(lock_file)
        except FileNotFoundError:
            pass

    def _pid_alive(p: int) -> bool:
        """Return True if a process with that PID is currently running."""
        try:
            os.kill(p, 0)
            return True
        except (ProcessLookupError, OSError):
            return False
        except PermissionError:
            return True  # exists but we can't signal it (Windows)

    def get_other_goals() -> list:
        """Read goals of other live PinPoint instances; delete stale lock files."""
        goals = []
        for fname in os.listdir(LOCK_DIR):
            if not (fname.startswith("session_") and fname.endswith(".json")):
                continue
            if fname == f"session_{pid}.json":
                continue
            fpath = os.path.join(LOCK_DIR, fname)
            try:
                with open(fpath) as f:
                    data = json.load(f)
                other_pid = data.get("pid", 0)
                goal = data.get("goal", "")
                if not goal or goal.strip().lower() in ("deciding...", "thinking..."):
                    os.remove(fpath)
                    continue
                if other_pid and not _pid_alive(other_pid):
                    os.remove(fpath)  # clean up zombie lock file
                    continue
                goals.append(goal)
            except Exception:
                try:
                    os.remove(fpath)  # remove corrupt lock file
                except Exception:
                    pass
        return goals

    session = 0

    # Start in chat mode immediately — PinPoint wakes up into a conversation.
    # Blank Enter in chat drops into an autonomous session; typed text becomes the order.
    # Skip this if a command-line order was already given (/dev, sandbox, etc.)
    if not order:
        order = _chat_mode(interrupt_queue, model_ready=_model_ready)

    while True:
        session += 1
        _sessions_run[0] = session  # keep signal handler count in sync

        # Handle voice control commands (can be used anytime)
        if order in ("/mute", "/unmute", "/toggle"):
            from tools import mute_voice, unmute_voice, toggle_voice
            if order == "/mute":
                result = mute_voice()
            elif order == "/unmute":
                result = unmute_voice()
            else:  # /toggle
                result = toggle_voice()
            print(f"\n{result}\n")
            order = get_user_order()
            continue

        other_goals = get_other_goals()
        print(f"\n{'='*60}")
        print(f"  Starting session #{session}")
        if order:
            print(f"  Order: {order}")
        if other_goals:
            print(f"  Other instances working on: {'; '.join(other_goals)}")
        print(f"{'='*60}\n")

        write_lock("deciding...")
        summary = ""
        _session_failed = False
        try:
            summary = agent.run(
                logger=logger,
                order=order,
                interrupt_queue=interrupt_queue,
                other_goals=other_goals,
                write_lock=write_lock,
                dev_mode=dev_mode,
            )
        except Exception as e:
            _session_failed = True
            print(f"\n[ERROR in session #{session}]: {e}")
            logger.exception("Session %d crashed", session)
        finally:
            clear_lock()

        # Record build outcome in persistent state — she learns from real consequences
        try:
            import pinpoint_state as _ps_main
            _st = _ps_main.load_state()
            success = bool(summary) and not _session_failed and "stuck" not in (summary or "").lower()
            _ps_main.record_build(_st, task=order or "(no order)", success=success, notes=(summary or "")[:200])
        except Exception:
            pass

        print("\n--- Files in output/ ---")
        print(list_files())

        if summary:
            print("\n--- Session summary ---")
            print(summary)

        # Hot-reload agent.py if it changed (e.g. after git pull)
        _reload_if_changed()

        # Drop into chat mode between sessions — PinPoint opens with "here's what I built"
        order = _chat_mode(interrupt_queue, previous_summary=summary, model_ready=_model_ready)


if __name__ == "__main__":
    main()
