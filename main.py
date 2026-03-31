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


def check_api_key() -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable is not set.")
        print("Get a free key at: https://aistudio.google.com/apikey")
        print("Then set it with:  set GEMINI_API_KEY=your_key_here")
        sys.exit(1)


def main() -> None:
    check_api_key()
    logger = setup_logging()

    print("Pinpoint — Autonomous AI Agent")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Log file: {os.path.join(OUTPUT_DIR, 'agent_log.txt')}\n")

    import agent
    summary = agent.run(logger=logger)

    print("\n--- Files created by the agent ---")
    from tools import list_files
    print(list_files())
    print()

    if summary:
        print("--- Agent's own summary ---")
        print(summary)


if __name__ == "__main__":
    main()
