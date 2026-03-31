import os
import json
import logging
from typing import Optional

import httpx

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


def _call_api(client: httpx.Client, base_url: str, api_key: str, messages: list) -> dict:
    url = base_url.rstrip("/") + "/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "User-Agent": "anthropic-typescript/0.36.3 node/22.14.0",
        "X-Stainless-Lang": "js",
        "X-Stainless-Package-Version": "0.36.3",
        "X-Stainless-OS": "Windows",
        "X-Stainless-Arch": "x64",
        "X-Stainless-Runtime": "node",
        "X-Stainless-Runtime-Version": "v22.14.0",
    }
    body = {
        "model": MODEL,
        "max_tokens": 16000,
        "system": SYSTEM_PROMPT,
        "tools": TOOL_DEFINITIONS,
        "messages": messages,
    }
    resp = client.post(url, headers=headers, json=body, timeout=300)
    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")
    return resp.json()


def run(logger: Optional[logging.Logger] = None) -> str:
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://agentrouter.org")
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")

    if logger is None:
        logger = logging.getLogger("agent")

    messages = []
    iteration = 0
    final_summary = ""

    print("\n" + "=" * 60)
    print("  AUTONOMOUS AI AGENT — starting up")
    print("=" * 60 + "\n")
    logger.info("Agent started. Model: %s | Max iterations: %d", MODEL, MAX_ITERATIONS)

    with httpx.Client() as client:
        while iteration < MAX_ITERATIONS:
            iteration += 1
            logger.info("--- Iteration %d ---", iteration)

            if iteration == 1:
                messages.append({
                    "role": "user",
                    "content": "You are now running autonomously. Think about what you want to create, then use your tools to build it. There is no time limit — take as long as you need. Begin.",
                })

            data = _call_api(client, base_url, api_key, messages)

            assistant_content = []
            tool_results = []
            finished = False

            for block in data.get("content", []):
                btype = block.get("type")

                if btype == "text":
                    text = block.get("text", "")
                    if text.strip():
                        print(f"\n[AGENT] {text}\n")
                        logger.info("[TEXT] %s", text)
                    assistant_content.append(block)

                elif btype == "tool_use":
                    name = block.get("name")
                    inp = block.get("input", {})
                    print(f"\n[TOOL CALL] {name}({_fmt_input(inp)})")
                    logger.info("[TOOL] %s | input: %s", name, inp)
                    assistant_content.append(block)

                    result = dispatch(name, inp)
                    print(f"[TOOL RESULT] {result[:300]}{'...' if len(result) > 300 else ''}\n")
                    logger.info("[RESULT] %s", result)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.get("id"),
                        "content": result,
                    })

                    if name == "done":
                        final_summary = inp.get("summary", "")
                        finished = True

            messages.append({"role": "assistant", "content": assistant_content})
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            if finished:
                print("\n" + "=" * 60)
                print("  AGENT FINISHED")
                print("=" * 60)
                print(f"\nSummary:\n{final_summary}\n")
                logger.info("Agent finished. Summary: %s", final_summary)
                break

            stop_reason = data.get("stop_reason")
            if stop_reason == "end_turn" and not tool_results:
                print("\n[Agent stopped without calling done — ending session.]\n")
                logger.warning("Agent stopped without calling done tool.")
                break

    return final_summary


def _fmt_input(inp: dict) -> str:
    parts = []
    for k, v in inp.items():
        s = str(v)
        if len(s) > 80:
            s = s[:80] + "..."
        parts.append(f"{k}={s!r}")
    return ", ".join(parts)
