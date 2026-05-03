import os
import sys
import json
import time
import queue
import random
import threading
import logging

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

BANNER = r"""
  ____  _       ____       _       _
 |  _ \(_)_ __ |  _ \ ___ (_)_ __ | |_
 | |_) | | '_ \| |_) / _ \| | '_ \| __|
 |  __/| | | | |  __/ (_) | | | | | |_
 |_|   |_|_| |_|_|   \___/|_|_| |_|\__|

 Autonomous AI — running until you stop it
 Ctrl+C to stop  |  /mute, /unmute, /toggle = voice control
 /dev = improve yourself  |  sandbox = upgrade 3D viewer
 research = research coding topics  |  type anything = suggest to PinPoint
"""

LOCK_DIR = os.path.join(os.path.dirname(__file__), "output")

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
                        if line:
                            sys.stdout.write(f"\n[INPUT] {line}\n")
                            sys.stdout.flush()
                            interrupt_queue.put(line)
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
                    interrupt_queue.put(line.strip())
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
    src = os.path.join(os.path.dirname(__file__), "viewer.html")
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

    # Command-line order overrides interactive prompt
    order = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else ""
    dev_mode = False

    if not order:
        order = get_user_order()

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
    from tools import list_files

    # Start background input listener
    interrupt_queue: queue.Queue = queue.Queue()
    input_thread = threading.Thread(target=_input_listener, args=(interrupt_queue,), daemon=True)
    input_thread.start()

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

        print(f"\n[Resting {REST_BETWEEN_SESSIONS}s before next session — press Ctrl+C to stop]")
        for _ in range(REST_BETWEEN_SESSIONS * 10):
            time.sleep(0.1)


if __name__ == "__main__":
    main()
