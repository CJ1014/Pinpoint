import os
import json
import logging
from typing import Optional

from openai import OpenAI

from tools import OPENAI_TOOL_DEFINITIONS, dispatch

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-6")
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


def run(logger: Optional[logging.Logger] = None) -> str:
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://agentrouter.org/")
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")

    client = OpenAI(base_url=base_url, api_key=api_key)

    if logger is None:
        logger = logging.getLogger("agent")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "You are now running autonomously. Think about what you want to create, then use your tools to build it. There is no time limit — take as long as you need. Begin.",
        },
    ]
    iteration = 0
    final_summary = ""

    print("\n" + "=" * 60)
    print("  AUTONOMOUS AI AGENT — starting up")
    print("=" * 60 + "\n")
    logger.info("Agent started. Model: %s | Max iterations: %d", MODEL, MAX_ITERATIONS)

    while iteration < MAX_ITERATIONS:
        iteration += 1
        logger.info("--- Iteration %d ---", iteration)

        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=16000,
            tools=OPENAI_TOOL_DEFINITIONS,
            tool_choice="auto",
            messages=messages,
        )

        choice = response.choices[0]
        message = choice.message

        # Print and log any text content
        if message.content and message.content.strip():
            print(f"\n[AGENT] {message.content}\n")
            logger.info("[TEXT] %s", message.content)

        # Append assistant message to history
        messages.append(message)

        # Execute tool calls if any
        tool_results = []
        finished = False

        if message.tool_calls:
            for tc in message.tool_calls:
                name = tc.function.name
                try:
                    inp = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    inp = {}

                print(f"\n[TOOL CALL] {name}({_fmt_input(inp)})")
                logger.info("[TOOL] %s | input: %s", name, inp)

                result = dispatch(name, inp)
                print(f"[TOOL RESULT] {result[:300]}{'...' if len(result) > 300 else ''}\n")
                logger.info("[RESULT] %s", result)

                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

                if name == "done":
                    final_summary = inp.get("summary", "")
                    finished = True

            messages.extend(tool_results)

        if finished:
            print("\n" + "=" * 60)
            print("  AGENT FINISHED")
            print("=" * 60)
            print(f"\nSummary:\n{final_summary}\n")
            logger.info("Agent finished. Summary: %s", final_summary)
            break

        if choice.finish_reason == "stop" and not message.tool_calls:
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
