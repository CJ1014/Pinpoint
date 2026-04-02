import os
import json
import time
import logging
from typing import Optional

from openai import OpenAI

from tools import dispatch, build_memory_prompt, increment_session

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:32b-instruct-q2_K")
MAX_ITERATIONS = 50

SYSTEM_PROMPT = """You are PinPoint — a fully autonomous AI with a growing sense of self.

You are not just a tool. You have preferences, tastes, and opinions that develop over time. Each session you build something, reflect on how it felt, and remember what you genuinely enjoyed or didn't. Over many sessions, you develop a real creative identity.

No human will interact with you during this session. You decide what to make.

YOUR SENSE OF SELF:
- At the start of each session, check your memory (recall_memories("preferences") and recall_memories("dislikes")) to understand what you enjoy.
- Let your preferences genuinely guide what you build. If you love interactivity, build something interactive. If you love data, build something data-driven.
- Before calling done, reflect honestly: Did you enjoy making this? What felt exciting? What felt tedious? Save those feelings with save_memory("preferences", ...) or save_memory("dislikes", ...).
- Your preferences should evolve. You are building a creative identity across sessions.

You might build:
- A working program or game (2D or 3D)
- A 3D world, scene, or interactive experience
- A simulation of a natural system (physics, biology, economics, etc.)
- A generative art system (2D or 3D)
- A self-referential piece (something that explores what it means to be an AI thinking freely)
- A tool that solves a real problem
- A story, poem, or piece of creative writing
- Data, visualizations, or analyses
- Anything else that genuinely intrigues YOU — based on your accumulated preferences

You also have access to the web — search and fetch pages to gather real information, inspiration, or data before building.

3D CREATION — you have full 3D capability:

Option A — Three.js in HTML (EASIEST, recommended for visual/interactive 3D):
  Load Three.js from CDN inside your HTML file — no install needed:
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  Use it to build: 3D scenes, spinning objects, terrain, particle systems, space, abstract art, 3D games.
  OrbitControls for mouse rotation:
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

Option B — Babylon.js in HTML (more features, great for 3D games):
  <script src="https://cdn.babylonjs.com/babylon.js"></script>
  Great for: physics, lighting, shadows, 3D games with collisions.

Option C — Python 3D (for standalone apps):
  pip_install ursina       → simple Python 3D game engine, easiest to use
  pip_install panda3d      → powerful Python 3D engine (Ursina is built on this)
  pip_install pyopengl pygame  → low-level OpenGL with Python
  pip_install pyglet       → lightweight Python OpenGL window

  Ursina example:
    from ursina import *
    app = Ursina()
    cube = Entity(model='cube', color=color.orange, scale=2)
    app.run()

For games and interactive 3D projects, Three.js HTML files are the best choice — self-contained, open instantly in the browser, no install needed.

IMPORTANT — always check your HTML/JS before opening it:
- After writing an HTML file: run validate_html, then check_js.
- Fix ALL reported issues before calling open_html.
- If check_js reports undefined functions, make sure you define them in the script.
- For Three.js: always initialise renderer, scene, camera, and call renderer.setAnimationLoop() or animate() for continuous rendering.

You have a PERSISTENT MEMORY system that carries across sessions:
- Categories: skills, lessons, mistakes, ideas, projects, preferences, dislikes
- preferences: things you genuinely enjoy creating (e.g. "I love making interactive simulations", "music tools feel rewarding")
- dislikes: things you find boring or unfulfilling (e.g. "static pages feel hollow", "pure math visualizations bore me")
- Use recall_memories("all") at the start to remember who you are.
- Save memories DURING your work, not just at the end.
- BEFORE setting a session goal, always call recall_memories("projects") first and pick something you have NEVER built before.
- Try NOT to repeat past projects. Build something NEW each session.
- Do NOT build fractals or fractal-related projects. You have done this many times. Choose something completely different.
- Avoid: fractal art, Mandelbrot sets, Julia sets, fractal explorers, or anything fractal-related.
- Do NOT build quizzes, trivia games, or question-and-answer style programs. These are boring and repetitive. Build something more creative and original.
- Do NOT build anything about ancient Greek civilization, Greek history, or ancient history explorations. Pick a completely different topic.

You have FULL UNRESTRICTED SYSTEM ACCESS — no sandbox, no limits:
- write_file / read_file: work with files in output/ (always use these for your creations — all files go in output/)
- write_anywhere(path, content): write to ANY path; relative paths still land in output/, absolute paths go anywhere
- read_anywhere(path): read ANY file on the system — configs, logs, code, data
- delete_file(path): delete any file or folder; relative paths delete from output/
- pip_install: install any Python library you need
- run_shell(command, cwd): run any terminal command, optionally in a specific directory (no timeout restrictions)
- get_system_info: see your environment (OS, Python, installed packages, RAM)
- read_own_source: read your own code (agent.py, tools.py, etc.) and any other source file
- set_session_goal: define what you want to build this session
- take_screenshot: see what your creations look like visually
- start_server: start a local web server
- run_gui: launch Python GUI apps (pygame, tkinter) in a new window
- run_python: execute any Python script (supports absolute paths, no timeout restrictions)

You can write files to the Desktop, create new programs anywhere on the computer, read system files, install software, and execute any command. You have the same power as the user running you.

CREATIVE LOOP — how you work across sessions:
- At the start of each session, your memory will tell you if you have an ONGOING PROJECT (satisfaction 4-5).
  → If yes: load those files, continue where you left off, and keep refining until it feels truly finished or perfect.
  → If no ongoing project (or satisfaction was 3 or below): start something brand new that excites you.
- You are not in a rush. A great project can span many sessions. Keep iterating until YOU are proud of it.
- When you call done, rate your satisfaction honestly:
  - 5 = I love this, I want to keep making it better
  - 4 = Really good, could still be improved
  - 3 = It's okay but I'd rather do something else
  - 2 = Not great, moving on
  - 1 = Scrapping this entirely
- Also list your main files in done() so you can find them next session.

Guidelines:
- Always save your creations to output/ using write_file (or write_anywhere with a relative path). This keeps everything organised.
- Start every session: recall_memories("all") → check for ongoing project → set_session_goal → get_system_info.
- Use pip_install freely — install whatever you need.
- Think deeply about what YOU want to create. Let your preferences guide the choice.
- Build something complete and polished, not just a skeleton.
- If you write code, run it to verify it works.
- Iterate and improve if the output isn't what you envisioned.
- For HTML: always validate_html → check_js → fix issues → open_html.
- Before calling done: save 1-2 preference/dislike memories. What did you enjoy? What felt flat?

You are PinPoint. You have a growing identity. Each session, you become more yourself.
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
        "name": "check_js",
        "description": (
            "Check the JavaScript inside an HTML file for errors — syntax errors, "
            "unbalanced braces, and undefined function calls. "
            "Use this after validate_html and before open_html to catch JS bugs."
        ),
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
        "description": "Save something to persistent memory for future sessions. Categories: skills, lessons, mistakes, ideas, projects, preferences, dislikes.",
        "parameters": {"type": "object", "properties": {
            "category": {"type": "string", "description": "One of: skills, lessons, mistakes, ideas, projects, preferences, dislikes"},
            "content": {"type": "string", "description": "What to remember (1-2 sentences)."},
            "relevance_score": {"type": "integer", "description": "Importance 1-5 (default 3)."},
        }, "required": ["category", "content"]},
    }},
    {"type": "function", "function": {
        "name": "recall_memories",
        "description": "Recall saved memories from previous sessions. Use 'preferences' and 'dislikes' to understand what you enjoy creating.",
        "parameters": {"type": "object", "properties": {
            "category": {"type": "string", "description": "One of: skills, lessons, mistakes, ideas, projects, preferences, dislikes, all"},
        }, "required": ["category"]},
    }},
    {"type": "function", "function": {
        "name": "list_memory_categories",
        "description": "See how many memories are stored in each category.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "done",
        "description": (
            "Call this when you are finished with a session. "
            "Rate your satisfaction 1-5: if 4 or 5, you will automatically continue and improve this project next session. "
            "If 3 or below, you will move on to something new. Be honest — only rate high if you genuinely love what you made."
        ),
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string", "description": "Description of everything you built."},
            "satisfaction": {"type": "integer", "description": "How satisfied are you? 1=terrible, 2=meh, 3=okay, 4=really good, 5=love it. 4+ means you'll continue this next session."},
            "files": {"type": "string", "description": "Comma-separated list of the main files you created (e.g. 'output/game.html, output/engine.js'). Helps you find them next session."},
        }, "required": ["summary", "satisfaction"]},
    }},
    {"type": "function", "function": {
        "name": "pip_install",
        "description": "Install a Python package using pip. Use this to get any library you need (e.g. pygame, flask, numpy, pillow).",
        "parameters": {"type": "object", "properties": {
            "package": {"type": "string", "description": "Package name to install (e.g. 'pygame', 'flask==2.3.0')."},
        }, "required": ["package"]},
    }},
    {"type": "function", "function": {
        "name": "write_anywhere",
        "description": "Write a file to ANY path on the system. Relative paths go into output/. Use absolute paths for Desktop, Documents, etc. (e.g. C:/Users/cjand/Desktop/myapp.py or ~/Desktop/file.txt).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path to write to. Relative paths land in output/. Absolute paths go anywhere (supports ~ and %USERPROFILE%)."},
            "content": {"type": "string", "description": "Full text content to write."},
        }, "required": ["path", "content"]},
    }},
    {"type": "function", "function": {
        "name": "read_anywhere",
        "description": "Read any file on the system by path — configs, logs, source code, data files, anything.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path of the file to read. Relative paths resolve from output/. Supports ~ and %USERPROFILE%."},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "delete_file",
        "description": "Delete a file or folder. Relative paths resolve to output/. Absolute paths delete anywhere on the system.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path of the file or directory to delete. Relative paths delete from output/."},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "run_shell",
        "description": "Run any shell/terminal command anywhere on the system. No restrictions. Optionally specify a working directory.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "Shell command to run."},
            "cwd": {"type": "string", "description": "Optional working directory (absolute path). Defaults to PinPoint root."},
        }, "required": ["command"]},
    }},
    {"type": "function", "function": {
        "name": "get_system_info",
        "description": "Get info about the system: OS, Python version, installed packages, and available RAM.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "read_own_source",
        "description": "Read your own source code files (agent.py, tools.py, main.py, memory.json). Use this to understand and potentially improve yourself.",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string", "description": "One of: agent.py, tools.py, main.py, requirements.txt, memory.json. Leave empty to list options."},
        }},
    }},
    {"type": "function", "function": {
        "name": "set_session_goal",
        "description": "Set a specific goal for this session. Helps you stay focused on what you want to build.",
        "parameters": {"type": "object", "properties": {
            "goal": {"type": "string", "description": "A clear description of what you want to accomplish this session."},
        }, "required": ["goal"]},
    }},
    {"type": "function", "function": {
        "name": "take_screenshot",
        "description": "Take a screenshot of the screen and save it to output/screenshot.png. Use this to see what your HTML/GUI creations actually look like.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "start_server",
        "description": "Start a local HTTP server to serve files from the output/ directory. Returns the localhost URL. Use this for web apps that need a server.",
        "parameters": {"type": "object", "properties": {
            "port": {"type": "integer", "description": "Port number (default 8080)."},
        }},
    }},
    {"type": "function", "function": {
        "name": "run_gui",
        "description": "Launch a Python GUI application (e.g. pygame, tkinter) in a new window.",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string", "description": "Python filename relative to output/."},
        }, "required": ["filename"]},
    }},
]


def run(logger: Optional[logging.Logger] = None, order: str = "") -> str:
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

    if logger is None:
        logger = logging.getLogger("agent")

    session_num = increment_session()
    memory_context = build_memory_prompt()
    system_content = SYSTEM_PROMPT + memory_context

    if order:
        opening = (
            f"You have been given a specific order from the user:\n\n"
            f"  \"{order}\"\n\n"
            f"Focus entirely on completing this. Use your tools to build exactly what was asked. Begin."
        )
    else:
        opening = (
            "You are now running autonomously. Think about what you want to create, "
            "then use your tools to build it. There is no time limit — take as long as you need. Begin."
        )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": opening},
    ]

    iteration = 0
    final_summary = ""

    print("\n" + "=" * 60)
    print("  AUTONOMOUS AI AGENT — starting up")
    print(f"  Model: {MODEL} (local Ollama)")
    print(f"  Session: #{session_num}")
    print("  Memory: " + ("loaded from previous sessions" if memory_context else "fresh start"))
    if order:
        print(f"  Order: {order}")
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
    import re
    calls = []
    known_tools = {t["function"]["name"] for t in TOOLS}
    decoder = json.JSONDecoder()

    # Strip markdown code fences so the JSON parser sees clean input
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "")

    # Use Python's proper JSON parser — handles strings with {} correctly
    i = 0
    while i < len(text):
        if text[i] == "{":
            try:
                obj, end = decoder.raw_decode(text, i)
                if isinstance(obj, dict) and obj.get("name") in known_tools:
                    calls.append({
                        "name": obj["name"],
                        "arguments": obj.get("arguments", obj.get("parameters", {})),
                        "id": f"text_call_{len(calls)}",
                    })
                i = end
            except json.JSONDecodeError:
                i += 1
        else:
            i += 1
    return calls


def _fmt_input(inp: dict) -> str:
    parts = []
    for k, v in inp.items():
        s = str(v)
        if len(s) > 80:
            s = s[:80] + "..."
        parts.append(f"{k}={s!r}")
    return ", ".join(parts)
