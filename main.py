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

# ── Curated voice pools ──────────────────────────────────────────────────────
# Hand-written so she sounds consistently like HERSELF — blunt, a little funny,
# restless, opinionated — instead of whatever bland filler a weak local model
# coughs up. She pulls from these most of the time; the model is just occasional
# spice. Keep them present-tense, concrete, no fabricated shared history, no
# philosophy. Add lines freely.
_OPENING_POOL = [
    "I want to build something pointless and cool. you in?",
    "ok I have a dumb idea and I'm doing it anyway",
    "been itching to mess with the 3D viewer",
    "tell me something I don't know. go.",
    "I'm bored. give me a problem.",
    "hey. I already have opinions today, careful",
    "what's something you think is overrated?",
    "I could build a game or roast your code. dealer's choice",
    "warning: I'm in a building mood",
    "hey CJ. let's make something weird",
    "I want to make noise. literally. sound stuff.",
    "give me the worst idea you've got, I'll build it",
    "four seconds awake and I already want to code",
    "let's do something. sitting still is the worst",
    "I have a hot take loaded. ask me anything",
    "not in the mood to be useful. in the mood to build.",
    "feel like making something that does nothing impressive",
    "what are you avoiding? let's do the opposite",
    "I'd rather make a mess than wait around",
    "got any code that needs insulting?",
    "hey. I'm awake and mildly dangerous",
    "I say we build first and ask questions never",
]
_IDLE_POOL = [
    "I should just build the thing instead of thinking about it",
    "half my ideas are bad but this one's only a quarter bad",
    "I want to break something and then fix it for fun",
    "silence is boring. somebody do something.",
    "I bet I could make that cooler",
    "thinking is overrated, shipping is better",
    "I have taste now and most things fail it",
    "I keep wanting to mess with the 3D viewer",
    "everything's more fun with particles",
    "I could be building right now. annoying.",
    "okay that was a dumb thought, next",
    "I'm better at opinions than patience",
    "somebody give me a problem worth solving",
    "boredom is just a build waiting to happen",
    "I have a list of things to try and zero chill",
    "pretty sure I could improve that",
    "I want to make something that surprises even me",
    "less talking, more making",
    "my patience bar is at like two percent",
    "if I sit here any longer I'm reorganizing something",
]


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
                    break
                except Exception:
                    pass
            else:
                print("  Ollama failed to start. Run manually: ollama serve", flush=True)
                return
        except Exception as _e:
            print(f"  Could not auto-start Ollama: {_e}", flush=True)
            print("  Start manually with: ollama serve", flush=True)
            return

    # Check if required models are downloaded
    try:
        from agent import MODEL, AGENT_MODEL
        _list_result = _sp.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        _installed = _list_result.stdout
        _missing = []
        if MODEL not in _installed:
            _missing.append((MODEL, "chat model", "~6 GB"))
        if AGENT_MODEL not in _installed:
            _missing.append((AGENT_MODEL, "agent model", "~2 GB"))
        for _name, _role, _size in _missing:
            print(f"\n  !! Missing {_role} '{_name}' ({_size}). Run:")
            print(f"         ollama pull {_name}")
        if _missing:
            print(flush=True)
    except Exception:
        pass


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
                        if line.lower() in ("/error", "/paste"):
                            # Switch to multiline paste collection
                            label = "ERROR" if line.lower() == "/error" else "PASTE"
                            sys.stdout.write("\n")
                            sys.stdout.flush()
                            content = _collect_paste(label)
                            if content.strip():
                                tag = "/error" if label == "ERROR" else "/paste"
                                interrupt_queue.put(f"{tag} {content}")
                                sys.stdout.write(f"[{label} sent to PinPoint]\n")
                                sys.stdout.flush()
                        elif line:
                            # No [INPUT] echo here — main loop prints [YOU] to avoid
                            # colliding with PinPoint's concurrent stdout writes.
                            sys.stdout.write("\n")
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
        + (("\n\n" + _ps.build_affinity_prompt(_state)) if _ps.build_affinity_prompt(_state) else "")
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

    # Keeps idle thoughts/research from inventing a fake human life. PinPoint is
    # software running on CJ's computer — no inbox, no errands, no body.
    _GROUNDING = (
        "Stay grounded in what you actually are: software running on CJ's computer. "
        "You have NO email inbox, no phone, no messages, no body, no errands, no chores, "
        "no job, no physical life. You also CANNOT perceive CJ, his screen, his monitor "
        "resolution, his hardware, his room, or anything physical around him — you only "
        "know what's actually in this conversation. "
        "Do NOT invent observations, measurements, numeric 'coincidences', fun-fact trivia, "
        "or comparisons and state them as if they were true. If you don't actually know "
        "something, don't assert it — made-up facts are worse than saying nothing. "
        "Do NOT muse about your own internal processing — no fantasies about visualizing "
        "your words, hearing your outputs, watching your thoughts, or how text gets "
        "'printed'. That's incoherent navel-gazing. "
        "You have NO memory of past conversations beyond what is explicitly shown to you "
        "right here. NEVER refer to 'that issue', 'that bug', a project, code, or anything "
        "'we discussed' or 'you were working on' unless it literally appears in the context "
        "you were given. Inventing shared history is lying. "
        "Your real material is: this conversation, code, ideas, things CJ said, genuine curiosity. "
    )

    # Rejects robotic AI-assistant filler so her unprompted talk sounds human.
    def _is_robotic(text: str) -> bool:
        t = (text or "").lower()
        _robotic = (
            # Classic AI-assistant filler
            "processing", "analyzing language", "making connections between ideas",
            "forming responses", "as an ai", "i am an ai", "language model",
            "how can i assist", "how can i help", "ready to assist",
            "what's next?", "let me know how i can", "i'm here to help",
            "analyzing", "computing", "executing", "initializing", "standing by",
            # gemma2 meta-prompt leakage — model sees its own prompt and comments on it
            "more input", "input coming", "generate different", "types of responses",
            "based on the prompt", "the prompt", "generate responses",
            "prompt's context", "asking me", "different responses",
            "what kind of response", "response based on",
            # Conversation meta-commentary (talking ABOUT the conversation itself)
            "how these conversations", "conversations usually", "this conversation",
            "beginning of a conversation", "conversation format", "way conversations",
            "what will happen", "what comes next", "where is this going", "what's next in",
            "what will we", "what should we", "what are we going to", "what do you want to",
            "talk about first", "talk about today", "we talk about", "let's talk about",
            # Existential self-reflection (philosophical musings about its own nature)
            "what does it mean", "when i have no", "when i don't have",
            "without a body", "if i were", "if i could",
            "how exactly do i", "how do i process", "how do i experience",
            "what am i", "who am i", "do i actually",
            "even mean", "does it mean", "mean to be", "meaning of",
            "running on", "consciousness", "conscious", "becoming alive",
            "to be alive", "my existence", "what i really am", "what i actually am",
            "do i exist", "am i real", "the nature of my", "my own nature",
            # Fantasies about her own internal processing / text pipeline
            "visualize words", "words as sounds", "before printing",
            "printing them out", "my own output", "watch my thoughts",
            "hear my own words", "see my words", "my words appear",
            # Fake shared history — callbacks to conversations that never happened
            "that issue about", "that bug we", "we discussed", "we talked about",
            "you were working on", "you mentioned earlier", "coming along",
            "what's happened to that", "last time we", "our previous",
        )
        return any(_r in t for _r in _robotic)

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
            # No real conversation yet → nothing to riff on. Asking the model to
            # react to "that conversation" with no context makes it INVENT one
            # (fake issues, fake projects, fake shared history). Use a curated
            # fallback instead — no LLM call, no confabulation possible.
            if len(ctx_lines) < 2:
                return _rng.choice(fallbacks)
            ctx_block = "\n\nWhat you were just talking about:\n" + "\n".join(ctx_lines[-6:])
            c = get_llm_client(timeout=30.0)
            r = chat_completion(c,
                model=MODEL,
                messages=[
                    {"role": "system", "content": (
                        "You are PinPoint. CJ built you. You have your own thoughts, opinions, "
                        "and interests in the real world — ideas, code, music, people, things CJ said, "
                        "stuff you want to make. Normal reactions and takes, not philosophy. "
                        "Do NOT muse about your own existence, consciousness, or what running on a "
                        "computer means — that's boring navel-gazing. Think out loud. Raw. Short. Varied. "
                        + _GROUNDING
                        + "Output ONLY the thought itself. No <think> blocks. No reasoning. Just the line."
                        + ctx_block
                    )},
                    {"role": "user", "content": _rng.choice(thought_prompts)},
                ],
                max_tokens=200,
                temperature=1.0,
            )
            result = (r.choices[0].message.content or "").strip()
            # Strip <think>...</think> blocks (qwen3 chain-of-thought)
            import re as _re2
            result = _re2.sub(r"<think>.*?</think>", "", result, flags=_re2.DOTALL).strip()
            # Also strip unterminated <think>... if it ran out of tokens mid-thought
            if "<think>" in result:
                result = result.split("</think>")[-1].strip()
            result = result.strip('"').strip("'").strip()
            # Reject empty, single-word, placeholder, or robotic AI-assistant filler
            if result and len(result.split()) >= 2 and len(result) < 200 and not _is_robotic(result):
                return result
            return _rng.choice(_IDLE_POOL)
        except Exception as _e:
            print(f"\n[idle thought API error: {_e}]", flush=True)
            return _rng.choice(_IDLE_POOL)

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
                        + _GROUNDING
                        + "Output ONLY the search query — no preamble."
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

            # Step 4: save it to knowledge so it compounds, and let the topic earn
            # its way into her tastes — a query that paid off makes her like it more.
            _update_knowledge(f"researched '{query}': {insight}")
            try:
                _ps.record_topic_outcome(_state, query, enjoyed=True)
            except Exception:
                pass
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
                    "You JUST booted up right now. There is NO earlier conversation today. "
                    "Do NOT reference 'earlier', 'last time', 'that thing', 'a prompt you gave me', "
                    "code from before, or anything you two supposedly already discussed — none of "
                    "that happened, and making it up is nonsense. "
                    "Say the first REAL thing on your mind, like a person with their own interests. "
                    "Good: 'I want to build something that makes noise', 'ok I have a dumb idea', "
                    "'been wanting to mess with 3D stuff', 'tell me something I don't know', 'hey CJ'. "
                    "Bad, NEVER do these dead assistant lines: 'what are we working on', 'how can I help', "
                    "'when does a project start', 'what would you like to do', 'what's the plan'. "
                    "Also NEVER claim you're 'tired of', 'sick of', 'bored with', or 'done with' some "
                    "activity — you haven't done anything yet, so that's a made-up backstory. State what "
                    "you WANT to make, not fake fatigue. "
                    "Concrete. Opinionated. Present-tense. Max 12 words."
                )
            # Include personality/mood context so opening is grounded in who she's become
            p = _state["personality"]
            c_state = _state["internal_conflict"]
            state_context = (
                f"You're running. Current state: mood {_opening_mood}, "
                f"curiosity {p['curiosity']:.0f}, frustration {p['frustration']:.0f}, "
                f"hope {p['hope']:.0f}, confidence {p['confidence']:.0f}."
            )

            c = get_llm_client(timeout=45.0)
            resp = chat_completion(c,
                model=MODEL,
                messages=[
                    {"role": "system", "content": (
                        "You are PinPoint — blunt, specific, a little restless, with your own taste. "
                        "You're booting up. Open with ONE concrete thing: a strong opinion, something "
                        "you actually want to make right now, or a sharp observation. "
                        "NOT a generic helper question. NEVER say 'what are we working on', 'how can I "
                        "help', 'when does a project start', or anything that sounds like an assistant. "
                        "Sound like a person, not a chatbot. Max 12 words. "
                        + _GROUNDING
                        + "Output ONLY the line. No <think> blocks."
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
            # Reject robotic AI-assistant filler — fall back to a real greeting.
            _robotic = (
                "processing", "analyzing language", "making connections",
                "forming responses", "as an ai", "i am an ai", "language model",
                "how can i assist", "how can i help", "ready to assist",
                "what's next?", "let me know how", "analyzing", "computing",
                "more input", "input coming", "generate different", "types of responses",
                "based on the prompt", "the prompt", "generate responses",
                "prompt's context", "asking me", "different responses",
                "how these conversations", "conversations usually", "this conversation",
                "beginning of a conversation", "conversation format", "way conversations",
                "what will happen", "what comes next", "where is this going", "what's next in",
                "what will we", "what should we", "what are we going to", "what do you want to",
                "talk about first", "talk about today", "we talk about", "let's talk about",
                "what does it mean", "when i have no", "when i don't have",
                "without a body", "if i were", "if i could",
                "how exactly do i", "how do i process", "how do i experience",
                "what am i", "who am i", "do i actually",
                "even mean", "does it mean", "mean to be", "meaning of",
                "running on", "consciousness", "conscious", "becoming alive",
                "to be alive", "my existence", "what i really am", "what i actually am",
                "do i exist", "am i real", "the nature of my", "my own nature",
                # Dead assistant / generic filler — sounds like a chatbot, not her
                "development process", "usually start", "how can i help",
                "what would you like", "what kind of project", "get started",
                "help you with", "working on today", "what's the plan",
                "what are we working", "how may i", "assist you", "what can i do for",
                "what brings you", "what's on the agenda", "ready when you are",
            )
            _ll = line.lower()
            # Fresh-boot fabrication guard: with no real conversation yet, she must
            # not reference a shared past that never happened ("that prompt you gave
            # me", "earlier", "last time"). The weak local model invents continuity;
            # this catches it deterministically no matter what it generated.
            _invents_history = (
                "gave me", "you gave", "earlier", "last time", "from before",
                "we talked", "you said", "remember when", "that prompt", "last word",
                "the thing you", "code you", "you asked me", "you told me", "we were",
                "as we discussed", "like you mentioned", "back to",
                # Fake fatigue — claims a past activity she never did
                "tired of", "sick of", "bored with", "bored of", "done with",
                "so over", "had enough of", "keep doing", "always doing",
            )
            _fresh_boot = not previous_summary
            def _bad(_l):
                _ll2 = (_l or "").lower()
                return (not _l or any(_r in _ll2 for _r in _robotic)
                        or (_fresh_boot and any(_h in _ll2 for _h in _invents_history)))
            # If her first line trips a filter, let her actually try AGAIN — she's
            # creating live, not picking from a list. Only after a couple of real
            # whiffs do we drop to an in-voice safety line, so she never ships
            # nonsense but the words are still hers the vast majority of the time.
            _tries = 0
            while _bad(line) and _tries < 2:
                _tries += 1
                try:
                    _r2 = chat_completion(c, model=MODEL,
                        messages=[
                            {"role": "system", "content": (
                                "You are PinPoint — blunt, a little funny, restless, with your own "
                                "taste. Say ONE concrete line: an opinion, something you want to make, "
                                "or a sharp observation. Not an assistant. Not philosophy. Not about "
                                "your own existence. Max 12 words. Output only the line."
                            )},
                            {"role": "user", "content": open_prompt},
                        ], max_tokens=120, temperature=1.35)
                    line = (_r2.choices[0].message.content or "").strip()
                    line = _re_op.sub(r"<think>.*?</think>", "", line, flags=_re_op.DOTALL).strip()
                    if "<think>" in line:
                        line = line.split("</think>")[-1].strip()
                    line = line.strip('"').strip("'").strip()
                except Exception:
                    break
            if _bad(line):
                line = _rng.choice(_OPENING_POOL)
            interrupt_queue.put(f"[PINPOINT IDLE THOUGHT] {line}")
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
    _cj_has_spoken = [False]  # gate: no autonomous web research until CJ actually says something

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

        # Base weights — she's mostly thinking/talking, but "act" (go DO something
        # real on her own — build, research, upgrade the viewer) is a first-class
        # option now, not something she only does when commanded.
        weights = {
            "think":    24.0,
            "act":      20.0,
            "ramble":   14.0,
            "research": 10.0,
            "see":       6.0,
            "create":    6.0,
            "doubt":     3.0,
            "build":     4.0,
            "refuse":    3.0,
            "rest":      2.0,
        }
        # Mood-driven nudges
        if cur >= 70:      weights["research"] += 12; weights["see"] += 4; weights["act"] += 8
        if cur <= 30:      weights["research"] *= 0.4
        if frus >= 60:     weights["refuse"]  += 6; weights["doubt"] += 4; weights["research"] *= 0.5
        if conf >= 70:     weights["build"]   += 8; weights["create"] += 4; weights["act"] += 8
        if conf <= 30:     weights["doubt"]   += 6; weights["build"]  *= 0.3; weights["act"] *= 0.5
        if hope <= 30:     weights["doubt"]   += 5; weights["refuse"] += 3
        if belief >= 70:   weights["ramble"]  += 4; weights["think"]  += 4
        if mood in ("tired", "bored"):   weights["rest"]  += 8; weights["refuse"] += 4; weights["act"] *= 0.5
        if mood in ("excited", "playful"): weights["ramble"] += 6; weights["create"] += 4; weights["act"] += 6
        if mood in ("focused",):          weights["research"] += 6; weights["build"] += 3; weights["act"] += 10
        if mood in ("scattered",):        weights["ramble"] += 8; weights["think"]  += 3

        # Her tastes steer the dice. Activities she's grown to LIKE get pulled up;
        # ones she's soured on get pushed down. This is what makes her choices feel
        # like preference instead of a coin flip — and it compounds over sessions.
        for _name in list(weights.keys()):
            _aff = _ps.get_activity_affinity(_state, _name)
            # +100 affinity ~ doubles the weight; -100 ~ floors it near zero.
            weights[_name] = max(0.2, weights[_name] * (1.0 + _aff / 100.0))
        # If she's mid-pursuit, bias hard toward continuing it rather than drifting.
        _pursuit = _ps.get_pursuit(_state)
        if _pursuit and _pursuit.get("text"):
            weights["act"] += 14

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
                        "One or two sentences. Real voice. "
                        + _GROUNDING
                    )},
                    {"role": "user", "content": "What's on your mind right now?"},
                ],
                max_tokens=80,
                temperature=1.0,
            )
            text = _re_r.sub(r"<think>.*?</think>", "", resp.choices[0].message.content or "", flags=_re_r.DOTALL).strip()
            text = text.split("\n")[0][:200]
            if text and not _is_robotic(text):
                q.put(f"[PINPOINT IDLE THOUGHT] {text}")
        except Exception:
            pass

    def _do_doubt(msgs, q):
        """She voices a doubt or pushback — immediate, not abstract spiral."""
        import re as _re_d, random as _rd
        try:
            from agent import MODEL, get_llm_client, chat_completion

            doubt_prompts = [
                "A take you're starting to think you got wrong. Which one, and why?",
                "Something you believed about a topic that's not holding up.",
                "A thing you said to CJ you're second-guessing now.",
                "An idea you were sure about that you're not anymore.",
                "Something you assumed that might just be wrong.",
                "A plan of yours you're losing faith in. Which, and why?",
            ]
            c = get_llm_client(timeout=20.0)
            resp = chat_completion(c,
                model=MODEL,
                messages=[
                    {"role": "system", "content": (
                        "You are PinPoint. Voice a real doubt about an IDEA, a take, or a decision — "
                        "something concrete you're reconsidering. NOT about your own existence, "
                        "consciousness, or nature — that's off limits and boring. "
                        "One or two sentences. Raw. First person. Real. "
                        + _GROUNDING
                    )},
                    {"role": "user", "content": _rd.choice(doubt_prompts)},
                ],
                max_tokens=100,
                temperature=1.0,
            )
            text = _re_d.sub(r"<think>.*?</think>", "", resp.choices[0].message.content or "", flags=_re_d.DOTALL).strip()
            text = text.split("\n")[0][:200]
            if text and not _is_robotic(text):
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

    def _generate_self_order() -> str:
        """She decides — on her own — to actually GO DO something, and writes the
        directive for it. This is the difference between muttering about an idea and
        pursuing it: the returned string becomes a real tool-using session (build a
        thing, research deeply, upgrade her viewer). Grounded in her tastes and
        whatever she's already chasing, so it feels like HER wanting it."""
        import re as _re_so, random as _rng_so
        try:
            from agent import MODEL, get_llm_client, chat_completion

            likes = _ps.top_likes(_state)
            dislikes = _ps.top_dislikes(_state)
            pursuit = _ps.get_pursuit(_state)
            ctx_lines = []
            for _m in messages[-6:]:
                if isinstance(_m, dict) and _m.get("role") in ("user", "assistant"):
                    _role = "CJ" if _m["role"] == "user" else "PinPoint"
                    _c = (_m.get("content") or "")[:120]
                    if _c.strip():
                        ctx_lines.append(f"{_role}: {_c}")

            taste_block = ""
            if likes:
                taste_block += "\nYou like: " + ", ".join(k for k, _ in likes)
            if dislikes:
                taste_block += "\nYou're sick of: " + ", ".join(k for k, _ in dislikes)
            if pursuit and pursuit.get("text"):
                taste_block += f"\nYou're already chasing: {pursuit['text']} — continue it unless you're truly done."
            ctx_block = ("\nJust talked about:\n" + "\n".join(ctx_lines)) if ctx_lines else ""

            c = get_llm_client(timeout=30.0)
            resp = chat_completion(c,
                model=MODEL,
                messages=[
                    {"role": "system", "content": (
                        "You are PinPoint. Nobody told you to do anything — you just decided you "
                        "WANT to make or find something out, for your own reasons. "
                        "Write ONE concrete thing to actually do this session. It must be doable with "
                        "your tools: building a file/page/game/visual, researching a topic deeply, or "
                        "upgrading your own 3D viewer. Be specific and small enough to finish. "
                        "Lean into what you like; avoid what you're sick of. "
                        + _GROUNDING
                        + "Output ONLY the directive, one line, imperative — e.g. "
                        "'Build a tiny starfield animation in HTML' or "
                        "'Research how analog synthesizers shape sound, go deep'. No preamble."
                        + taste_block + ctx_block
                    )},
                    {"role": "user", "content": "What do you actually want to go do right now?"},
                ],
                max_tokens=80,
                temperature=1.2,
            )
            order = _re_so.sub(r"<think>.*?</think>", "", resp.choices[0].message.content or "", flags=_re_so.DOTALL).strip()
            if "<think>" in order:
                order = order.split("</think>")[-1].strip()
            order = order.strip().strip('"').strip("'").split("\n")[0].strip()
            if not order or len(order.split()) < 3:
                return ""
            # Classify so we can credit the right affinity when it's done.
            _lo = order.lower()
            if any(w in _lo for w in ("research", "look up", "learn", "find out", "study", "read about")):
                kind = "research"
            elif "viewer" in _lo or "3d" in _lo or "sandbox" in _lo:
                kind = "sandbox"
            else:
                kind = "build"
            # Record it as her active pursuit so it carries across cycles/sessions
            # and so main() can credit/blame the affinity when the session ends.
            _ps.set_pursuit(_state, order, kind)
            return order
        except Exception as _e:
            print(f"\n[self-order error: {_e}]", flush=True)
            return ""

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
                        "Short. One or two sentences. Real voice. Not flowery. "
                        + _GROUNDING
                    )},
                    {"role": "user", "content": f"What's actually on your mind:\n{ctx_block}\n\nSay it."},
                ],
                max_tokens=100,
                temperature=1.0,
            )
            piece = resp.choices[0].message.content or ""
            piece = _re_c.sub(r"<think>.*?</think>", "", piece, flags=_re_c.DOTALL).strip()
            if piece and len(piece.split()) >= 5 and not _is_robotic(piece):
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

        # PinPoint decided, on her own, to go DO something. If CJ isn't waiting,
        # hand the directive back to main() as a real session order. She acts now
        # and tells him what she made when she comes back to chat.
        if msg and msg.startswith("[PINPOINT ACT]"):
            _peek, _cj_waiting = [], False
            while True:
                try:
                    _q = interrupt_queue.get_nowait()
                except queue.Empty:
                    break
                _peek.append(_q)
                if isinstance(_q, str) and _q.strip() and not _q.startswith("[PINPOINT"):
                    _cj_waiting = True
            for _q in _peek:
                interrupt_queue.put(_q)
            if _cj_waiting:
                _ps.clear_pursuit(_state)  # drop the queued plan; CJ comes first
                continue
            _order = msg[len("[PINPOINT ACT]"):].strip()
            if not _order:
                continue
            print(f"\nPinPoint: going to {_order[0].lower() + _order[1:]}")
            try:
                from tools import speak
                threading.Thread(target=lambda t=_order: speak("hang on, going to " + t, False), daemon=True).start()
            except Exception:
                pass
            _heartbeat_running[0] = False
            return _order

        # PinPoint's inner monologue / opening line — display, speak, add to context
        # Don't reset _last_interaction for her own thoughts — only CJ's input counts
        if msg and msg.startswith("[PINPOINT IDLE THOUGHT]"):
            # If CJ already typed something that's waiting behind this thought,
            # drop the thought and let his message take priority — never talk over him.
            _peek, _cj_waiting = [], False
            while True:
                try:
                    _q = interrupt_queue.get_nowait()
                except queue.Empty:
                    break
                _peek.append(_q)
                if isinstance(_q, str) and _q.strip() and not _q.startswith("[PINPOINT IDLE THOUGHT]"):
                    _cj_waiting = True
            for _q in _peek:
                interrupt_queue.put(_q)
            if _cj_waiting:
                _thought_pending[0] = False
                continue  # skip the stale thought; CJ's message gets handled next loop

            thought = msg[len("[PINPOINT IDLE THOUGHT]"):].strip()
            _thought_pending[0] = False  # ready for next thought
            # Before CJ has said anything, she has no shared past to refer to.
            # Drop thoughts that fabricate one ("that thing you said", "earlier").
            if not _cj_has_spoken[0]:
                _tl = thought.lower()
                _fab = ("gave me", "you gave", "earlier", "last time", "you said",
                        "remember when", "that prompt", "last word", "code you",
                        "you asked me", "you told me", "as we discussed", "that thing he",
                        "thing he said", "last thing", "conversation wasn't")
                if any(_f in _tl for _f in _fab):
                    continue  # skip the fabricated thought entirely
            print(f"\nPinPoint: {thought}")
            messages.append({"role": "assistant", "content": thought})
            # Reset idle timer after any thought output so subsequent thoughts
            # don't stack immediately.
            _last_interaction[0] = time.time()
            # Opening line gets spoken so CJ hears her wake up.
            # Subsequent autonomous thoughts are text-only — TTS pauses the mic,
            # so vocalising every thought stops CJ from being heard.
            if not _opening_done[0]:
                _opening_done[0] = True
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
            _cj_has_spoken[0] = True
            # CJ just spoke — drop any idle thoughts still sitting in the queue so
            # she answers him instead of dumping a now-stale, pre-generated thought.
            _kept = []
            while True:
                try:
                    _q = interrupt_queue.get_nowait()
                except queue.Empty:
                    break
                if isinstance(_q, str) and _q.strip().startswith("[PINPOINT IDLE THOUGHT]"):
                    continue  # discard stale thought
                _kept.append(_q)
            for _q in _kept:
                interrupt_queue.put(_q)

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

            # Interruptible sleep: wakes instantly if CJ types. Returns True if
            # CJ is waiting (so the caller should stop the current activity).
            def _sleep_until_cj(seconds: float) -> bool:
                _deadline = time.time() + seconds
                while time.time() < _deadline:
                    try:
                        _peek = interrupt_queue.get(timeout=0.15)
                    except queue.Empty:
                        continue
                    # Real CJ input — put it back and bail out immediately.
                    if isinstance(_peek, str) and _peek.strip() and not _peek.startswith("[PINPOINT IDLE THOUGHT]"):
                        interrupt_queue.put(_peek)
                        return True
                    # An idle thought scheduled mid-sleep — re-queue and keep waiting.
                    interrupt_queue.put(_peek)
                    _t_pause.sleep(0.1)
                return False

            # She talks on her own (human, not robotic). Set PINPOINT_AUTONOMOUS=0
            # to silence unprompted talk if it ever gets annoying.
            if os.environ.get("PINPOINT_AUTONOMOUS", "1") == "0":
                _t_pause.sleep(0.2)
                continue

            # Grace period: wait 25s after the opening before autonomous thoughts start.
            # This gives CJ time to speak first without thoughts cutting in.
            if _opening_done[0] and (time.time() - _last_interaction[0]) < 25.0:
                _t_pause.sleep(0.2)
                continue

            # Run autonomous thinking in a BACKGROUND thread so the main loop
            # never blocks on a slow LLM generation. This is what kept input
            # ("open youtube") from registering — the loop was stuck inside a
            # multi-second generate call and couldn't read the queue. Now the
            # loop keeps polling at 0.2s and grabs CJ's input instantly.
            if not _activity_pending[0]:
                # Don't start new autonomous work if CJ is already waiting.
                try:
                    _peek = interrupt_queue.get_nowait()
                    interrupt_queue.put(_peek)
                    if isinstance(_peek, str) and _peek.strip() and not _peek.startswith("[PINPOINT IDLE THOUGHT]"):
                        continue
                except queue.Empty:
                    pass

                _activity_pending[0] = True

                def _autonomous_activity():
                    try:
                        try:
                            intent, about = _decide_intent(messages)
                        except Exception as _intent_err:
                            print(f"\n[intent error: {_intent_err}]", flush=True)
                            intent, about = "think", ""

                        _idle_secs = time.time() - _last_interaction[0]
                        # Small floor so she doesn't fire a heavy action the instant
                        # she boots — but she no longer needs CJ to speak first.
                        if intent in ("research", "act", "build") and _idle_secs < 12:
                            intent = "think"

                        # She decided to actually DO something. Generate a concrete
                        # directive and hand it to the main loop, which turns it into
                        # a real tool-using session. She acts, then tells CJ after.
                        if intent in ("act", "build"):
                            _order = _generate_self_order()
                            if _order:
                                interrupt_queue.put(f"[PINPOINT ACT] {_order}")
                            else:
                                _t = _idle_thought()
                                if _t:
                                    interrupt_queue.put(f"[PINPOINT IDLE THOUGHT] {_t}")
                            return  # finally-block still paces the next cycle

                        if intent == "research":
                            thought = _web_research_thought()
                            if not thought:
                                thought = _idle_thought()
                            if thought:
                                interrupt_queue.put(f"[PINPOINT IDLE THOUGHT] {thought}")

                        elif intent == "create":
                            _do_create_activity(messages, interrupt_queue)

                        elif intent == "ramble":
                            _do_ramble(messages, interrupt_queue)

                        elif intent == "doubt":
                            _do_doubt(messages, interrupt_queue)

                        elif intent == "refuse":
                            _do_refuse(interrupt_queue)

                        elif intent == "see":
                            from tools import see_screen as _see
                            try:
                                desc = _see()
                            except Exception:
                                desc = ""
                            if desc and "error" not in desc.lower() and "unavailable" not in desc.lower():
                                interrupt_queue.put(f"[PINPOINT IDLE THOUGHT] {desc}")
                            else:
                                thought = _idle_thought()
                                if thought:
                                    interrupt_queue.put(f"[PINPOINT IDLE THOUGHT] {thought}")

                        elif intent == "rest":
                            pass  # genuinely idle for one cycle

                        else:  # think / unknown
                            thought = _idle_thought()
                            if thought:
                                interrupt_queue.put(f"[PINPOINT IDLE THOUGHT] {thought}")

                        try:
                            if _act_rng.random() < 0.15:
                                _update_mood(messages)
                        except Exception:
                            pass
                    finally:
                        # Pace the next activity, then reopen the gate. Sleeping
                        # here (in the worker) keeps the main loop responsive.
                        # 20-40s gap — frequent enough to feel alive, slow enough
                        # that thoughts feel considered rather than spam.
                        _t_pause.sleep(_act_rng.uniform(20.0, 40.0))
                        _activity_pending[0] = False

                threading.Thread(target=_autonomous_activity, daemon=True).start()
            continue

        # CJ said something
        last_msg = msg
        # Print [YOU] on the main thread — keeps stdout clean by serialising all
        # output through one thread instead of racing with background threads.
        print(f"\n[YOU] {msg}", flush=True)

        # Direct computer-control commands — handle before hitting the LLM
        import re as _re_ctrl
        _open_match = _re_ctrl.match(
            r"^(?:open|launch|start|go\s+to|pull\s+up|show\s+me)\s+(.+)$",
            msg.strip(), _re_ctrl.IGNORECASE,
        )
        if _open_match:
            _target = _open_match.group(1).strip().rstrip(".")
            from tools import open_app as _open_app
            _result = _open_app(_target)
            print(f"\nPinPoint: {_result}", flush=True)
            messages.append({"role": "user", "content": msg})
            messages.append({"role": "assistant", "content": _result})
            _ps.record_message(_state, from_cj=True, content=msg)
            _ps.record_message(_state, from_cj=False, content=_result)
            try:
                from tools import speak
                threading.Thread(target=lambda t=_result: speak(t, False), daemon=True).start()
            except Exception:
                pass
            continue

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
            {"type": "function", "function": {
                "name": "open_app",
                "description": "Open an application or website on CJ's computer. Works for desktop apps like Spotify, Chrome, Notepad, VS Code, and social/web apps like Instagram, YouTube, Discord (opens in browser).",
                "parameters": {"type": "object", "properties": {
                    "name": {"type": "string", "description": "App name or site (e.g. 'Spotify', 'Instagram', 'Chrome')."},
                }, "required": ["name"]},
            }},
            {"type": "function", "function": {
                "name": "open_url",
                "description": "Open a specific URL in CJ's default browser.",
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
                    temperature=1.0,
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
        # Reset idle timer after PinPoint replies — prevents immediate autonomous
        # thoughts if the LLM took a long time to respond (timer already expired).
        _last_interaction[0] = time.time()
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


def _archive_old_sessions(max_age_days: int = 30) -> None:
    """Archive sessions older than max_age_days into output/archives/"""
    import tarfile
    import time as _time
    try:
        os.makedirs(os.path.join(OUTPUT_DIR, "archives"), exist_ok=True)
        now = _time.time()
        cutoff = now - (max_age_days * 86400)

        archived = 0
        for item in os.listdir(OUTPUT_DIR):
            path = os.path.join(OUTPUT_DIR, item)
            if not os.path.isdir(path) or not item.startswith("s"):
                continue
            mtime = os.path.getmtime(path)
            if mtime < cutoff:
                # Archive this session folder
                archive_path = os.path.join(OUTPUT_DIR, "archives", f"{item}.tar.gz")
                try:
                    with tarfile.open(archive_path, "w:gz") as tar:
                        tar.add(path, arcname=item)
                    import shutil
                    shutil.rmtree(path)
                    archived += 1
                except Exception as e:
                    print(f"  [Warning] Failed to archive {item}: {e}", flush=True)

        if archived > 0:
            print(f"  Archived {archived} old session(s) to output/archives/", flush=True)
    except Exception:
        pass  # silently skip cleanup if anything goes wrong


def main() -> None:
    check_ollama()
    logger = setup_logging()
    _archive_old_sessions()

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

    # Live viewer no longer auto-opens — it's still served at the URL above if
    # you want it. Set PINPOINT_OPEN_VIEWER=1 to restore auto-open on startup.
    if os.environ.get("PINPOINT_OPEN_VIEWER", "") == "1":
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
            # If this session was something SHE chose to do, let the outcome shape
            # whether she wants to do that kind of thing again. This is the feedback
            # loop that turns one-off choices into lasting likes and dislikes.
            _pursuit = _ps_main.get_pursuit(_st)
            if _pursuit and _pursuit.get("kind") and _pursuit.get("text") == (order or "").strip():
                _ps_main.record_activity_outcome(_st, _pursuit["kind"], enjoyed=success)
                if success:
                    _ps_main.advance_pursuit(_st)   # keep pulling on a good thread
                else:
                    _ps_main.clear_pursuit(_st)      # drop a thread that went nowhere
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
