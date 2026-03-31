import os
import json
import logging
from typing import Optional

from google import genai
from google.genai import types

from tools import dispatch

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
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

# Tool declarations in Gemini format
GEMINI_TOOLS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="write_file",
            description=(
                "Write content to a file inside the output/ directory. "
                "Use this to create programs, scripts, data files, stories, or anything else you want to build."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "filename": types.Schema(
                        type="STRING",
                        description="Filename (optionally with subdirectory, e.g. 'game.py' or 'data/config.json'). Stays inside output/.",
                    ),
                    "content": types.Schema(
                        type="STRING",
                        description="Full text content to write to the file.",
                    ),
                },
                required=["filename", "content"],
            ),
        ),
        types.FunctionDeclaration(
            name="read_file",
            description="Read the contents of a file you have previously written inside output/.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "filename": types.Schema(
                        type="STRING",
                        description="Filename relative to output/.",
                    ),
                },
                required=["filename"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_files",
            description="List all files you have created in the output/ directory.",
            parameters=types.Schema(
                type="OBJECT",
                properties={},
            ),
        ),
        types.FunctionDeclaration(
            name="run_python",
            description=(
                "Execute a Python script you have written inside output/ and see its output. "
                "Use this to test your code, run simulations, generate data, etc."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "filename": types.Schema(
                        type="STRING",
                        description="Python filename relative to output/ (e.g. 'simulation.py').",
                    ),
                },
                required=["filename"],
            ),
        ),
        types.FunctionDeclaration(
            name="done",
            description=(
                "Call this when you are completely finished with your creative work. "
                "Provide a summary of everything you created."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "summary": types.Schema(
                        type="STRING",
                        description="A description of everything you built and why you chose to create it.",
                    ),
                },
                required=["summary"],
            ),
        ),
    ]
)


def run(logger: Optional[logging.Logger] = None) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    if logger is None:
        logger = logging.getLogger("agent")

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[GEMINI_TOOLS],
        max_output_tokens=16000,
    )

    history = []
    iteration = 0
    final_summary = ""

    print("\n" + "=" * 60)
    print("  AUTONOMOUS AI AGENT — starting up")
    print(f"  Model: {MODEL}")
    print("=" * 60 + "\n")
    logger.info("Agent started. Model: %s | Max iterations: %d", MODEL, MAX_ITERATIONS)

    # Initial prompt
    user_msg = types.Content(
        role="user",
        parts=[types.Part.from_text(
            "You are now running autonomously. Think about what you want to create, "
            "then use your tools to build it. There is no time limit — take as long as you need. Begin."
        )],
    )
    history.append(user_msg)

    while iteration < MAX_ITERATIONS:
        iteration += 1
        logger.info("--- Iteration %d ---", iteration)

        response = client.models.generate_content(
            model=MODEL,
            contents=history,
            config=config,
        )

        # Build assistant parts for history
        assistant_parts = []
        tool_calls = []

        for part in response.candidates[0].content.parts:
            if part.text:
                text = part.text.strip()
                if text:
                    print(f"\n[AGENT] {text}\n")
                    logger.info("[TEXT] %s", text)
                assistant_parts.append(part)

            elif part.function_call:
                fc = part.function_call
                args = dict(fc.args) if fc.args else {}
                print(f"\n[TOOL CALL] {fc.name}({_fmt_input(args)})")
                logger.info("[TOOL] %s | input: %s", fc.name, args)
                assistant_parts.append(part)
                tool_calls.append((fc.name, args))

        # Add assistant turn to history
        history.append(types.Content(role="model", parts=assistant_parts))

        # Execute tool calls
        if tool_calls:
            tool_response_parts = []
            finished = False

            for name, args in tool_calls:
                result = dispatch(name, args)
                print(f"[TOOL RESULT] {result[:300]}{'...' if len(result) > 300 else ''}\n")
                logger.info("[RESULT] %s", result)

                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=name,
                        response={"result": result},
                    )
                )

                if name == "done":
                    final_summary = args.get("summary", "")
                    finished = True

            # Add tool results as user turn
            history.append(types.Content(role="user", parts=tool_response_parts))

            if finished:
                print("\n" + "=" * 60)
                print("  AGENT FINISHED")
                print("=" * 60)
                print(f"\nSummary:\n{final_summary}\n")
                logger.info("Agent finished. Summary: %s", final_summary)
                break
        else:
            # No tool calls — agent is done talking
            finish_reason = response.candidates[0].finish_reason
            if not tool_calls:
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
