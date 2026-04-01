import os
import sys
import time
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

    session = 0
    try:
        while True:
            session += 1
            print(f"\n{'='*60}")
            print(f"  Starting session #{session}")
            if order:
                print(f"  Order: {order}")
            print(f"{'='*60}\n")

            try:
                summary = agent.run(logger=logger, order=order)
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
