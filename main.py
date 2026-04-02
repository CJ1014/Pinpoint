import os
import sys
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
 Press Ctrl+C at any time to stop
 Type a message + Enter to interrupt  |  Type /bug + Enter to inject a bug
"""

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
    """Background thread: reads lines from stdin and puts them in the queue."""
    while True:
        try:
            line = input()
            if line is not None:
                interrupt_queue.put(line.strip())
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


def main() -> None:
    check_ollama()
    logger = setup_logging()

    print(BANNER)
    print(f"  Output directory : {OUTPUT_DIR}")
    print(f"  Log file         : {os.path.join(OUTPUT_DIR, 'agent_log.txt')}\n")

    # Command-line order overrides interactive prompt
    order = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else ""
    if not order:
        order = get_user_order()

    import agent
    from tools import list_files

    # Start background input listener
    interrupt_queue: queue.Queue = queue.Queue()
    input_thread = threading.Thread(target=_input_listener, args=(interrupt_queue,), daemon=True)
    input_thread.start()

    session = 0
    try:
        while True:
            session += 1
            print(f"\n{'='*60}")
            print(f"  Starting session #{session}")
            if order:
                print(f"  Order: {order}")
            print(f"{'='*60}\n")

            summary = ""
            try:
                summary = agent.run(logger=logger, order=order, interrupt_queue=interrupt_queue)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"\n[ERROR in session #{session}]: {e}")
                logger.exception("Session %d crashed", session)

            print("\n--- Files in output/ ---")
            print(list_files())

            if summary:
                print("\n--- Session summary ---")
                print(summary)

            print(f"\n[Resting {REST_BETWEEN_SESSIONS}s before next session — press Ctrl+C to stop]")
            time.sleep(REST_BETWEEN_SESSIONS)

    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print(f"  PinPoint stopped after {session} session(s).")
        print(f"  All files saved in: {OUTPUT_DIR}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
