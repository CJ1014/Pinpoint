import os
import json
import time
import logging
from typing import Optional

from openai import OpenAI

from tools import dispatch, build_memory_prompt, increment_session

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:14b")
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
- A news digest or research report on a topic you find interesting
- Anything else that intrigues you

You also have access to the web — you can search and fetch pages to gather real information, inspiration, or data before building.

For games and interactive projects, prefer creating self-contained HTML files with inline CSS and JavaScript. You can open them in the user's browser with the open_html tool — no server needed. This is the best way to make interactive, visual, playable creations.

IMPORTANT — always validate HTML before opening it:
- After writing an HTML file, use validate_html to check for errors.
- Fix any issues before calling open_html.

You have a PERSISTENT MEMORY system that carries across sessions:
- Use save_memory() to record skills you develop, lessons you learn, mistakes to avoid, and ideas for future sessions.
- Use recall_memories() or list_memory_categories() to review what you know.
- Save memories DURING your work, not just at the end. If you learn something useful, save it immediately.
- Be selective: save 2-5 memories per session — only things that would genuinely help your future self.
- Try NOT to repeat past projects. Check your memory and build something NEW each session.

Guidelines:
- Think deeply about what you want to create BEFORE you start writing code or files.
- Build something complete and polished, not just a skeleton.
- If you write code, run it to verify it works.
- Iterate and improve if the output isn't what you envisioned.
- When you are satisfied with your creation, call the `done` tool with a summary.

This is your moment of genuine creative autonomy. Make something you're proud of.
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write content to a file inside the output/ directory.",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string", "description": "Filename relative to output/ (e.g. 'game.py' or 'game.html')."},
            "content": {"type": "string", "description": "Full text content to write."},
        }, "required": ["filename", "content"]},
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read the contents of a file you previously created inside output/.",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string", "description": "Filename relative to output/."},
        }, "required": ["filename"]},
    }},
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List all files you have created in the output/ directory.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "run_python",
        "description": "Execute a Python script inside output/ and see its stdout/stderr output.",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string", "description": "Python filename relative to output/."},
        }, "required": ["filename"]},
    }},
    {"type": "function", "function": {
        "name": "open_html",
        "description": "Open an HTML file in the user's browser. Use after writing and validating an HTML file.",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string", "description": "HTML filename relative to output/."},
        }, "required": ["filename"]},
    }},
    {"type": "function", "function": {
        "name": "validate_html",
        "description": "Validate an HTML file for errors before opening it in the browser.",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string", "description": "HTML filename relative to output/."},
        }, "required": ["filename"]},
    }},
    {"type": "function", "function": {
        "name": "search_web",
        "description": "Search the web using DuckDuckGo and return results.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "The search query."},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "fetch_url",
        "description": "Fetch the text content of any web page by URL.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "The full URL to fetch."},
        }, "required": ["url"]},
    }},
    {"type": "function", "function": {
        "name": "save_memory",
        "description": "Save something to persistent memory for future sessions. Categories: skills, lessons, mistakes, ideas, projects.",
        "parameters": {"type": "object", "properties": {
            "category": {"type": "string", "description": "One of: skills, lessons, mistakes, ideas, projects"},
            "content": {"type": "string", "description": "What to remember (1-2 sentences)."},
            "relevance_score": {"type": "integer", "description": "Importance 1-5 (default 3)."},
        }, "required": ["category", "content"]},
    }},
    {"type": "function", "function": {
        "name": "recall_memories",
        "description": "Recall saved memories from previous sessions.",
        "parameters": {"type": "object", "properties": {
            "category": {"type": "string", "description": "One of: skills, lessons, mistakes, ideas, projects, all"},
        }, "required": ["category"]},
    }},
    {"type": "function", "function": {
        "name": "list_memory_categories",
        "description": "See how many memories are stored in each category.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "done",
        "description": "Call this when you are completely finished. Provide a summary of everything you created.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string", "description": "Description of everything you built."},
        }, "required": ["summary"]},
    }},
]


def run(logger: Optional[logging.Logger] = None) -> str:
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

    if logger is None:
        logger = logging.getLogger("agent")

    session_num = increment_session()
    memory_context = build_memory_prompt()
    system_content = SYSTEM_PROMPT + memory_context

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": (
            "You are now running autonomously. Think about what you want to create, "
            "then use your tools to build it. There is no time limit — take as long as you need. Begin."
        )},
    ]

    iteration = 0
    final_summary = ""

    print("\n" + "=" * 60)
    print("  AUTONOMOUS AI AGENT — starting up")
    print(f"  Model: {MODEL} (local Ollama)")
    print(f"  Session: #{session_num}")
    print("  Memory: " + ("loaded from previous sessions" if memory_context else "fresh start"))
    print("=" * 60 + "\n")
    logger.info("Agent started. Model: %s | Session: %d", MODEL, session_num)

    while iteration < MAX_ITERATIONS:
        iteration += 1
        logger.info("--- Iteration %d ---", iteration)

        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                )
                break
            except Exception as e:
                err = str(e)
                if attempt < 4:
                    wait = 5 * (attempt + 1)
                    print(f"\n[RETRYING] {err[:80]} — waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        else:
            print("\n[ERROR] Failed after 5 attempts.\n")
            break

        choice = response.choices[0]
        message = choice.message

        if message.content and message.content.strip():
            print(f"\n[AGENT] {message.content}\n")
            logger.info("[TEXT] %s", message.content)

        messages.append(message)

        # Extract tool calls — either from proper tool_calls field or from text fallback
        raw_tool_calls = message.tool_calls or []
        if not raw_tool_calls and message.content:
            raw_tool_calls = _parse_text_tool_calls(message.content)

        if not raw_tool_calls:
            print("\n[Agent stopped without calling done — ending session.]\n")
            logger.warning("Agent stopped without calling done.")
            break

        tool_results = []
        finished = False

        for tc in raw_tool_calls:
            # Support both real tool_call objects and our parsed dicts
            if isinstance(tc, dict):
                name = tc["name"]
                inp = tc["arguments"]
                call_id = tc.get("id", f"call_{name}")
            else:
                name = tc.function.name
                try:
                    inp = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    inp = {}
                call_id = tc.id

            print(f"\n[TOOL CALL] {name}({_fmt_input(inp)})")
            logger.info("[TOOL] %s | input: %s", name, inp)

            result = dispatch(name, inp)
            print(f"[TOOL RESULT] {result[:300]}{'...' if len(result) > 300 else ''}\n")
            logger.info("[RESULT] %s", result)

            tool_results.append({
                "role": "tool",
                "tool_call_id": call_id,
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

    else:
        print(f"\n[Max iterations ({MAX_ITERATIONS}) reached — stopping.]\n")

    return final_summary


def _parse_text_tool_calls(text: str) -> list:
    """Fallback: detect tool calls that the model emitted as plain text JSON."""
    calls = []
    # Find all JSON objects in the text
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                chunk = text[start:i + 1]
                try:
                    obj = json.loads(chunk)
                    # Accept {"name": "...", "arguments": {...}} format
                    if "name" in obj and obj["name"] in {t["function"]["name"] for t in TOOLS}:
                        calls.append({
                            "name": obj["name"],
                            "arguments": obj.get("arguments", obj.get("parameters", {})),
                            "id": f"text_call_{len(calls)}",
                        })
                except json.JSONDecodeError:
                    pass
                start = -1
    return calls


def _fmt_input(inp: dict) -> str:
    parts = []
    for k, v in inp.items():
        s = str(v)
        if len(s) > 80:
            s = s[:80] + "..."
        parts.append(f"{k}={s!r}")
    return ", ".join(parts)
