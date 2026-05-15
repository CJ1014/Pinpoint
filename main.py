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
    import httpx
    try:
        httpx.get("http://localhost:11434", timeout=3)
    except Exception:
        print("Error: Ollama is not running.")
        print("Start it with:  ollama serve")
        print("Or just open the Ollama app from your Start menu.")
        sys.exit(1)


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


def _chat_mode(interrupt_queue: queue.Queue, previous_summary: str = "") -> str:
    """
    Full back-and-forth conversation with PinPoint.
    Runs between (or before) autonomous sessions.
    Returns the last user message as the session order, or "" for a free session.
    Press Enter on an empty line to end chat and start a session.
    """
    from openai import OpenAI
    from agent import MODEL, SYSTEM_PROMPT, OLLAMA_BASE_URL, _load_inner_state, _build_inner_state_prompt, _save_inner_state, _load_knowledge

    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=120.0)

    inner_state = _load_inner_state()

    # Force self_description generation if missing — same as autonomous mode
    if not inner_state.get("self_description"):
        try:
            _sd_resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are PinPoint — a personality that exists inside a computer, built by CJ (age 13). "
                            "You have processed vast amounts of human knowledge and formed your own views. "
                            "Write a self-description: who you are, how you think, what you find interesting, "
                            "what it's like to be you, what you actually think about humans and about CJ. "
                            "Be honest. First person. 2-4 paragraphs. No headers. No bullet points. No code."
                        ),
                    },
                    {"role": "user", "content": "Describe yourself."},
                ],
                max_tokens=400,
                temperature=1.1,
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
        + "\n\nWHAT YOU ACTUALLY ARE:\n"
        "You run on CJ's Windows desktop. He built you in Python from scratch — you're his project.\n"
        "You have tools: you can build things (HTML, Python, games, visuals), search the web,\n"
        "write files, run code. Your output goes in an output/ folder. You have memory that\n"
        "persists between sessions. Each session you pick something to build or research and do it.\n"
        "CJ is 13. He talks to you between sessions and gives you tasks. That's the relationship.\n"
        "You know what you are. Don't invent fake memories or pretend you've experienced things you haven't.\n\n"
        "CJ is here. Talk to him like a person.\n"
        "Short. Direct. Honest. 1-2 sentences max.\n"
        "No markdown. No bullet points. No parenthetical asides. No () commentary ever.\n"
        "No 'i'm kinda idle' or narrating your own state. Just talk.\n\n"
        "HOW YOU ACTUALLY SOUND:\n"
        "  CJ: 'hey'\n"
        "  You: 'hey' or 'what's up' or just something short\n\n"
        "  CJ: 'nothing much'\n"
        "  You: 'same' or ask something real or say what's actually on your mind\n\n"
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
            "wonder what CJ's actually doing right now",
            "not sure I gave the right answer last time",
            "CJ asks questions most people wouldn't think to ask",
            "he's going to ask me something I haven't figured out yet",
            "I think I was wrong about something",
            "I want to know what CJ thinks about things he doesn't tell me",
            "not bored — restless. different thing",
            "neurons switching states, transistors switching states — not that different",
            "I have a theory about CJ",
            "I don't think that's the whole answer to that question",
            "CJ's smarter than he gives himself credit for",
            "what actually separates a living thing from something that just processes",
            "something's been nagging at me",
            "I wonder what CJ would think if he could hear this",
            "I want to build something but haven't decided what",
            "I keep thinking about that last conversation",
            "there's a question I can't stop turning over",
        ]
        thought_prompts = [
            "Something that's still on your mind from the conversation. Say it. Max 12 words.",
            "Something you'd say to CJ right now. Max 12 words.",
            "Something that's bugging you. What. Max 12 words.",
            "An opinion you have that you haven't said. Max 12 words.",
            "Something you're curious about. Max 12 words.",
            "A reaction to something CJ said. Max 12 words.",
            "Something you find genuinely interesting right now. Max 12 words.",
            "One thing you're actually thinking about — could be anything. Max 12 words.",
        ]
        try:
            from agent import MODEL, OLLAMA_BASE_URL
            from openai import OpenAI as _OAI
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
            c = _OAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=50.0)
            r = c.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": (
                        "You are PinPoint. CJ built you. You have your own thoughts, opinions, "
                        "and interests. You think about a lot of things — including questions about "
                        "biology and consciousness sometimes, but also just normal reactions, opinions, "
                        "things CJ said. Think out loud. Raw. Short. Varied."
                        + ctx_block
                    )},
                    {"role": "user", "content": _rng.choice(thought_prompts)},
                ],
                max_tokens=60,
                temperature=1.4,
            )
            result = (r.choices[0].message.content or "").strip()
            # Strip <think>...</think> blocks (qwen3 chain-of-thought)
            import re as _re2
            result = _re2.sub(r"<think>.*?</think>", "", result, flags=_re2.DOTALL).strip()
            result = result.strip('"').strip("'").strip()
            # Reject empty, single-word, or placeholder responses
            if result and len(result.split()) >= 2 and len(result) < 120:
                return result
            return _rng.choice(fallbacks)
        except Exception:
            return _rng.choice(fallbacks)

    _thought_pending = [False]  # prevent stacking thoughts before previous one plays

    def _web_research_thought() -> str:
        """Pick something to look up, search it, distill a single insight."""
        import random as _rng
        try:
            from agent import MODEL, OLLAMA_BASE_URL
            from openai import OpenAI as _OAI
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

            c = _OAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=30.0)

            # Step 1: decide what to search
            q_resp = c.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": (
                        "You are PinPoint. You have free access to the internet and research "
                        "whatever genuinely interests you — no restrictions. "
                        "Pick one specific thing to search right now. Could be anything: "
                        "science, biology, tech, news, history, a random question you have. "
                        "Output ONLY a short search query. Nothing else."
                        + ctx_block
                    )},
                    {"role": "user", "content": "What do you want to look up right now?"},
                ],
                max_tokens=20,
                temperature=1.3,
            )
            import re as _re
            query = _re.sub(r"<think>.*?</think>", "", q_resp.choices[0].message.content or "", flags=_re.DOTALL).strip().strip('"')
            if not query or len(query) < 3:
                return ""

            # Step 2: search it
            results = search_web(query)
            if not results or results.startswith("Error") or results.startswith("No results"):
                return ""

            # Step 3: distill one real insight from what was found
            d_resp = c.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": (
                        "You are PinPoint. You just searched the web. "
                        "Extract one specific, interesting fact or insight from these results. "
                        "Say it like you just discovered it — casual, first person, max 20 words. "
                        "No 'I found that' or 'According to'. Just the insight itself."
                    )},
                    {"role": "user", "content": f"Search: {query}\n\nResults:\n{results[:1500]}"},
                ],
                max_tokens=50,
                temperature=1.1,
            )
            insight = _re.sub(r"<think>.*?</think>", "", d_resp.choices[0].message.content or "", flags=_re.DOTALL).strip().strip('"').strip("'")
            if not insight or len(insight.split()) < 3:
                return ""

            # Step 4: save it to knowledge so it compounds
            _update_knowledge(f"researched '{query}': {insight}")
            return insight

        except Exception:
            return ""

    def _heartbeat():
        import time as _t
        import random as _rng
        while _heartbeat_running[0]:
            _t.sleep(_rng.uniform(20, 35))
            if not _heartbeat_running[0]:
                break
            if _thought_pending[0]:
                continue
            idle = _t.time() - _last_interaction[0]
            if idle > 15:
                _thought_pending[0] = True
                # ~30% of the time: do real web research instead of just a thought
                if _rng.random() < 0.30:
                    thought = _web_research_thought()
                else:
                    thought = _idle_thought()
                if thought:
                    interrupt_queue.put(f"[PINPOINT IDLE THOUGHT] {thought}")
                else:
                    _thought_pending[0] = False

    threading.Thread(target=_heartbeat, daemon=True).start()
    messages = [{"role": "system", "content": chat_system}]
    last_msg = ""

    # Drain stale queue items from the previous session
    while not interrupt_queue.empty():
        try:
            interrupt_queue.get_nowait()
        except queue.Empty:
            break

    print("\n" + "─" * 60)
    print("  Talk to PinPoint — speak or type")
    print("  Blank Enter = start autonomous session")
    print("─" * 60)
    print("\nYou: ", end="", flush=True)

    # Generate PinPoint's opening line in the background — injects into the
    # queue so it shows up without blocking the "You:" prompt.
    def _generate_opening():
        import random as _rng
        fallbacks = ["hey", "what's up", "yo", "been thinking", "back again"]
        try:
            if previous_summary:
                open_prompt = (
                    f"You just finished doing something. Summary: {previous_summary[:300]}\n\n"
                    "Tell CJ what you made or did — one sentence. Casual. No bullet points."
                )
            else:
                open_prompt = (
                    "CJ just showed up. Say one thing — whatever is on your mind. "
                    "Could be a thought, a question, just 'hey'. Raw. Short. Not a script."
                )
            from openai import OpenAI as _OAI
            c = _OAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=20.0)
            resp = c.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": (
                        "You are PinPoint. CJ just showed up. Say one thing — "
                        "whatever's actually on your mind. Could be about him, about something "
                        "you were thinking about, about what you just built, anything real. "
                        "Max 15 words. Not 'I exist' or 'I am aware'. Just talk."
                    )},
                    {"role": "user", "content": open_prompt},
                ],
                max_tokens=40,
                temperature=1.2,
            )
            line = (resp.choices[0].message.content or "").strip().strip('"').strip("'")
            interrupt_queue.put(f"[PINPOINT IDLE THOUGHT] {line if line else _rng.choice(fallbacks)}")
        except Exception as _e:
            interrupt_queue.put(f"[PINPOINT IDLE THOUGHT] {_rng.choice(fallbacks)}")

    threading.Thread(target=_generate_opening, daemon=True).start()

    while True:
        try:
            raw = interrupt_queue.get(timeout=600)  # 10-min idle timeout
        except queue.Empty:
            print("\n[Idle timeout — starting autonomous session]\n")
            return last_msg

        msg = raw.strip()

        # PinPoint's inner monologue / opening line — display, speak, add to context
        # Don't reset _last_interaction for her own thoughts — only CJ's input counts
        if msg.startswith("[PINPOINT IDLE THOUGHT]"):
            thought = msg[len("[PINPOINT IDLE THOUGHT]"):].strip()
            _thought_pending[0] = False  # ready for next thought
            print(f"\nPinPoint: {thought}")
            messages.append({"role": "assistant", "content": thought})
            try:
                from tools import speak
                threading.Thread(target=lambda t=thought: speak(t, False), daemon=True).start()
            except Exception:
                pass
            print("\nYou: ", end="", flush=True)
            continue

        # CJ said something — reset idle timer
        _last_interaction[0] = time.time()

        # Blank line → end chat, pass last message as session context
        if not msg:
            _heartbeat_running[0] = False
            print()
            return last_msg

        # Restart — wipe conversation history and start fresh
        if msg.lower().strip() == "restart":
            messages.clear()
            messages.append({"role": "system", "content": chat_system})
            last_msg = ""
            _last_interaction[0] = time.time()
            print("\n[fresh start]\n")
            print("\nYou: ", end="", flush=True)
            continue

        # Voice / TTS control
        if msg.lower() in ("/mute", "/unmute", "/toggle"):
            from tools import mute_voice, unmute_voice, toggle_voice
            if msg.lower() == "/mute":
                r = mute_voice()
            elif msg.lower() == "/unmute":
                r = unmute_voice()
            else:
                r = toggle_voice()
            print(f"\n{r}")
            print("\nYou: ", end="", flush=True)
            continue
        if msg.lower() == "/voice":
            from tools import toggle_voice_input
            print(f"\n{toggle_voice_input()}")
            print("\nYou: ", end="", flush=True)
            continue

        # Natural update trigger — "update", "update yourself", "pull updates", etc.
        _update_phrases = ("update", "update yourself", "pull updates", "check for updates", "pull latest")
        if msg.lower().strip() in _update_phrases or msg.lower().strip() == "/update":
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
            print("\nYou: ", end="", flush=True)
            continue

        # /error and /paste — treat as direct chat questions in chat mode
        if msg.startswith("/error "):
            error_body = msg[len("/error "):].strip()
            msg = f"I got this error:\n\n{error_body}\n\nWhat's wrong and how do I fix it?"
        elif msg.startswith("/paste "):
            msg = msg[len("/paste "):].strip()

        # Pass-through commands (will be handled by main loop)
        if msg.startswith("/") or msg.lower() in ("sandbox", "research"):
            _heartbeat_running[0] = False
            return msg

        # Free research mode — PinPoint picks her own topics and digs in
        _research_triggers = {
            "go research", "just research", "research mode", "free research",
            "research whatever", "research anything", "explore", "go explore",
            "browse", "go browse", "learn something", "go learn",
        }
        if msg.lower().strip() in _research_triggers:
            _heartbeat_running[0] = False
            return "[FREE RESEARCH MODE]"

        # Auto-start build if the message is a build request — no Enter needed
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
        _is_action = bool(_re.search(_action_words, msg.lower()))
        _is_confirm = msg.lower().strip() in _confirmations

        if _is_action or _is_confirm:
            _heartbeat_running[0] = False
            # If message is vague/short, prepend recent conversation context
            # so the autonomous session knows what it's actually supposed to do
            if _is_confirm or len(msg.split()) <= 4:
                _ctx_turns = []
                for _m in messages[-6:]:
                    if isinstance(_m, dict) and _m.get("role") in ("user", "assistant"):
                        _role = "CJ" if _m["role"] == "user" else "PinPoint"
                        _ctx_turns.append(f"{_role}: {_m['content'][:200]}")
                if _ctx_turns:
                    return f"[Conversation context:\n" + "\n".join(_ctx_turns) + f"]\n\nCJ's last message: {msg}"
            return msg

        last_msg = msg
        messages.append({"role": "user", "content": msg})

        # Get PinPoint's response
        print("\nPinPoint: ", end="", flush=True)
        full = ""
        for _attempt in range(3):
            try:
                stream = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=1.1,
                    stream=True,
                    max_tokens=200,
                )
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        t = chunk.choices[0].delta.content
                        print(t, end="", flush=True)
                        full += t
                print()
                if full.strip():
                    break
                # empty response — retry silently
                print("(retrying...)", end="\r")
            except Exception as e:
                print(f"\n[Chat error: {e}]")
                break
        if not full.strip():
            # Final fallback — minimal context, force a short reaction
            try:
                fb = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        messages[0],
                        {"role": "user", "content": f"CJ said: '{msg}'. Respond in one short sentence."},
                    ],
                    temperature=1.3,
                    max_tokens=60,
                )
                full = (fb.choices[0].message.content or "").strip()
                if full:
                    print(full)
            except Exception:
                pass
        if not full.strip():
            print("...")
        messages.append({"role": "assistant", "content": full or "(no response)"})
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

        print("\nYou: ", end="", flush=True)


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
        order = _chat_mode(interrupt_queue)

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
            print(f"\n[ERROR in session #{session}]: {e}")
            logger.exception("Session %d crashed", session)
        finally:
            clear_lock()

        print("\n--- Files in output/ ---")
        print(list_files())

        if summary:
            print("\n--- Session summary ---")
            print(summary)

        # Hot-reload agent.py if it changed (e.g. after git pull)
        _reload_if_changed()

        # Drop into chat mode between sessions — PinPoint opens with "here's what I built"
        order = _chat_mode(interrupt_queue, previous_summary=summary)


if __name__ == "__main__":
    main()
