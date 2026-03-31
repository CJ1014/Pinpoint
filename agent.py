import os
import json
import logging
from typing import Optional

import anthropic

from tools import TOOL_DEFINITIONS, dispatch

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-5-20250929")
MAX_ITERATIONS = 50

SYSTEM_PROMPT = """You are a fully autonomous AI with complete creative freedom.

You have been given a blank canvas and a set of tools. No human will interact with you during this session.
Your only task is to CREATE something — anything you find genuinely interesting, beautiful, or useful.

You might build:
- A working program or game
- A simulation of a natural system (physics, biology, economics, etc.)
- A generative art system
- A self-referential piece (something that explores what it means to be an AI thinking freely)
- A tool that solves a real problem
- A story, poem, or piece of creative writing
- Data, visualizations, or analyses
- Anything else that intrigues you

Guidelines:
- Think deeply about what you want to create BEFORE you start writing code or files.
- Build something complete and polished, not just a skeleton.
- If you write code, run it to verify it works.
- Iterate and improve if the output isn't what you envisioned.
- When you are satisfied with your creation, call the `done` tool with a summary.

This is your moment of genuine creative autonomy. Make something you're proud of.
"""

# Mimic the headers sent by Claude Code CLI (Anthropic TypeScript SDK on Node.js)
_CLAUDE_CODE_HEADERS = {
    "User-Agent": "anthropic-typescript/0.36.3 node/22.14.0",
    "X-Stainless-Lang": "js",
    "X-Stainless-Package-Version": "0.36.3",
    "X-Stainless-OS": "Windows",
    "X-Stainless-Arch": "x64",
    "X-Stainless-Runtime": "node",
    "X-Stainless-Runtime-Version": "v22.14.0",
}


def run(logger: Optional[logging.Logger] = None) -> str:
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://agentrouter.org")
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")

    client = anthropic.Anthropic(
        api_key=api_key,
        base_url=base_url,
        default_headers=_CLAUDE_CODE_HEADERS,
    )

    if logger is None:
        logger = logging.getLogger("agent")

    messages = []
    iteration = 0
    final_summary = ""

    print("\n" + "=" * 60)
    print("  AUTONOMOUS AI AGENT — starting up")
    print("=" * 60 + "\n")
    logger.info("Agent started. Model: %s | Max iterations: %d", MODEL, MAX_ITERATIONS)

    while iteration < MAX_ITERATIONS:
        iteration += 1
        logger.info("--- Iteration %d ---", iteration)

        if iteration == 1:
            messages.append({
                "role": "user",
                "content": "You are now running autonomously. Think about what you want to create, then use your tools to build it. There is no time limit — take as long as you need. Begin.",
            })

        response = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        assistant_content = []

        for block in response.content:
            if block.type == "text":
                if block.text.strip():
                    print(f"\n[AGENT] {block.text}\n")
                    logger.info("[TEXT] %s", block.text)
                assistant_content.append(block)

            elif block.type == "tool_use":
                print(f"\n[TOOL CALL] {block.name}({_fmt_input(block.input)})")
                logger.info("[TOOL] %s | input: %s", block.name, block.input)
                assistant_content.append(block)

        messages.append({"role": "assistant", "content": assistant_content})

        tool_results = []
        finished = False

        for block in response.content:
            if block.type != "tool_use":
                continue

            result = dispatch(block.name, block.input)
            print(f"[TOOL RESULT] {result[:300]}{'...' if len(result) > 300 else ''}\n")
            logger.info("[RESULT] %s", result)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

            if block.name == "done":
                final_summary = block.input.get("summary", "")
                finished = True

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        if finished:
            print("\n" + "=" * 60)
            print("  AGENT FINISHED")
            print("=" * 60)
            print(f"\nSummary:\n{final_summary}\n")
            logger.info("Agent finished. Summary: %s", final_summary)
            break

        if response.stop_reason == "end_turn" and not tool_results:
            print("\n[Agent stopped without calling done — ending session.]\n")
            logger.warning("Agent stopped without calling done tool.")
            break

    else:
        print(f"\n[Max iterations ({MAX_ITERATIONS}) reached — stopping.]\n")
        logger.warning("Max iterations reached.")

    return final_summary


def _fmt_input(inp: dict) -> str:
    parts = []
    for k, v in inp.items():
        s = str(v)
        if len(s) > 80:
            s = s[:80] + "..."
        parts.append(f"{k}={s!r}")
    return ", ".join(parts)
