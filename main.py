import os
import sys
import logging

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def setup_logging() -> logging.Logger:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, "agent_log.txt")

    logger = logging.getLogger("agent")
    logger.setLevel(logging.DEBUG)

    # File handler — full trace
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    logger.addHandler(fh)

    return logger


def check_ollama() -> None:
    import httpx
    try:
        resp = httpx.get("http://localhost:11434", timeout=3)
    except Exception:
        print("Error: Ollama is not running.")
        print("Start it with:  ollama serve")
        print("Or just open the Ollama app from your Start menu.")
        sys.exit(1)


def get_user_order() -> str:
    print("=" * 60)
    print("  What should the AI build?")
    print("  (Press Enter to let it decide on its own)")
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

    print("Pinpoint — Autonomous AI Agent")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Log file: {os.path.join(OUTPUT_DIR, 'agent_log.txt')}\n")

    # Check for command-line order (e.g. pinpoint "build a snake game")
    order = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else ""
    if not order:
        order = get_user_order()

    import agent
    summary = agent.run(logger=logger, order=order)

    print("\n--- Files created by the agent ---")
    from tools import list_files
    print(list_files())
    print()

    if summary:
        print("--- Agent's own summary ---")
        print(summary)


if __name__ == "__main__":
    main()
