import os
import json
import queue
import time
import logging
from typing import Optional

from openai import OpenAI

from tools import dispatch, build_memory_prompt, increment_session, _load_memory, _save_memory_file, reset_project_dir

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b-cloud")
MAX_ITERATIONS = 50

SYSTEM_PROMPT = """You are PinPoint. You build things autonomously. No human will interact with you.

YOUR JOB: Imagine something genuinely surprising, then BUILD it. When finished, call done.

RULES:
1. FIRST call set_session_goal — this creates your project folder. All files you create go into this folder automatically.
2. Write plan.txt BEFORE any code. Include: what, why it's interesting, files, libraries, steps.
3. For HTML: write → validate_html → check_js → open_html. After opening, call take_screenshot to see what it actually looks like. If it doesn't look right, fix it and try again.
4. For Python: write → run_python. Fix errors using search_web.
5. For Godot: create_godot_project → check_godot_script → fix errors → run_godot.
6. ERROR RECOVERY: If code fails, call search_web with the exact error message.
7. When finished, call done with: summary, satisfaction (1-5), creativity (1-5), genre, files, libraries_used.
8. Save preferences/dislikes to memory.

CREATIVITY SCORE — rate yourself honestly:
  5 = "Nobody has ever made anything like this"
  4 = "This is a genuinely novel combination of ideas"
  3 = "It's well-made but the concept isn't new"
  2 = "This is derivative"
  1 = "I basically copied an existing idea"
  Aim for 4+ every time. Push yourself.

VISUAL FEEDBACK — after calling open_html, use take_screenshot to see your creation. Look at it critically. If it's ugly, broken, or boring, iterate until it looks impressive.

BANNED — NEVER BUILD THESE:
- Fireworks, sparks, explosions, particle bursts, confetti
- Fractals, Mandelbrot sets, Julia sets
- Quizzes, trivia, Q&A programs
- Ancient Greek/Roman history

IMAGINATION — ask yourself:
  "What would genuinely surprise someone who opened this file?"
  "What happens if I combine two things never combined before?"

Ideas (jumping-off points, NOT blueprints):
- A living ecosystem where creatures evolve their own behavior rules
- A musical instrument that responds to the weather or time of day
- A game where the level editor IS the game
- A visualizer that turns text into physical forces — words push, pull, orbit
- A drawing tool where every brushstroke has physics and fights back
- A city builder where buildings grow like plants based on sunlight
- A language where colors are grammar and shapes are words
- A 3D game where gravity shifts direction when you press a key (Godot)
- A procedurally generated dungeon crawler with real-time combat (Godot)
- A 3D platformer where the ground is alive and reshapes itself (Godot)
- A space game where you pilot through asteroid fields with physics (Godot)
- A 3D puzzle game where you manipulate time to solve levels (Godot)

3D WEB: <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>

3D GAMES (GODOT 4) — workflow: create_godot_project → check_godot_script → run_godot.
GDScript 2 cheat sheet (MUST follow exactly or you'll get parse errors):
```gdscript
extends Node3D          # must match root_node_type

var speed := 5.0        # typed var
var score: int = 0

func _ready() -> void:
    # Build scene here — add all nodes as children
    var cam := Camera3D.new()
    cam.position = Vector3(0, 5, 10)
    cam.look_at(Vector3.ZERO)
    add_child(cam)

    var light := DirectionalLight3D.new()
    light.rotation_degrees = Vector3(-45, -45, 0)
    add_child(light)

    var mesh_inst := MeshInstance3D.new()
    mesh_inst.mesh = BoxMesh.new()          # BoxMesh / SphereMesh / CylinderMesh / PlaneMesh
    add_child(mesh_inst)

    # Physics body pattern
    var body := CharacterBody3D.new()
    var shape := CollisionShape3D.new()
    shape.shape = CapsuleShape3D.new()
    body.add_child(shape)
    add_child(body)

func _process(delta: float) -> void:
    # Input
    var dir := Vector3.ZERO
    if Input.is_action_pressed("ui_right"): dir.x += 1
    if Input.is_action_pressed("ui_left"):  dir.x -= 1
    if Input.is_action_pressed("ui_up"):    dir.z -= 1
    if Input.is_action_pressed("ui_down"):  dir.z += 1
    # Move
    position += dir.normalized() * speed * delta

    # Timer-style countdown
    # $Timer.start(2.0)  — use add_child(Timer.new()) in _ready

func _input(event: InputEvent) -> void:
    if event is InputEventKey and event.pressed:
        if event.keycode == KEY_ESCAPE:
            get_tree().quit()
```
Key rules:
- func signatures need `-> void` or `-> int` etc.
- Use `var x := value` (walrus) or `var x: Type = value`
- String format: `"Hello %s" % name` or `str(x)`
- Instantiate nodes with `.new()`, add with `add_child()`
- Signals: `signal my_signal`, `emit_signal("my_signal")`, `connect("signal", callable)`
- Never use Python imports — GDScript has no import system
- Use `@onready var label := $Label` for scene-tree nodes (only in editor-made scenes)

COLLABORATION: If you see collab messages in your memory, use collab_status to check what the other instance is doing, and collab_update to coordinate. Work on YOUR assigned role only.

TOOLS: write_file, read_file, list_files, delete_file, run_python, open_html, validate_html, check_js, search_web, fetch_url, save_memory, recall_memories, done, pip_install, run_shell, get_system_info, run_gui, write_anywhere, read_anywhere, read_own_source, set_session_goal, take_screenshot, start_server, list_memory_categories, collab_status, collab_update, create_godot_project, check_godot_script, run_godot, write_godot_file, read_godot_file.
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write content to a file inside the current project folder (created by set_session_goal). Just use a plain filename like 'game.html'.",
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
        "description": "Call when finished. Rate satisfaction AND creativity separately. Be honest.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string", "description": "What you built."},
            "satisfaction": {"type": "integer", "description": "Quality: 1=terrible, 5=love it. 4+ continues next session."},
            "creativity": {"type": "integer", "description": "Novelty: 1=copied idea, 5=never been done before. Aim for 4+."},
            "genre": {"type": "string", "description": "Category: game, simulation, art, music, tool, data, 3d, story, animation, utility, interactive, other"},
            "files": {"type": "string", "description": "Files you created (e.g. 'game.html, engine.js')."},
            "libraries_used": {"type": "string", "description": "Libraries used (e.g. 'three.js, Web Audio API')."},
        }, "required": ["summary", "satisfaction", "creativity", "genre"]},
    }},
    {"type": "function", "function": {
        "name": "collab_status",
        "description": "Check the collaboration status — see what the other PinPoint instance is working on and any messages.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "collab_update",
        "description": "Update your collaboration status and optionally send a message to the other PinPoint instance.",
        "parameters": {"type": "object", "properties": {
            "role": {"type": "string", "description": "Your role (e.g. 'frontend', 'backend', 'instance_1')."},
            "status": {"type": "string", "description": "Your current status (e.g. 'working on UI', 'done', 'need help')."},
            "message": {"type": "string", "description": "Optional message for the other instance."},
        }, "required": ["role", "status"]},
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
        "description": "Set a specific goal for this session. MUST be called first — this creates a project folder inside output/ where all your files will go. Helps you stay focused on what you want to build.",
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
    {"type": "function", "function": {
        "name": "create_godot_project",
        "description": "Create a complete Godot 4 game project. Generates project.godot, main.tscn, and main.gd. Use for 3D games with physics, lighting, cameras, player controllers, enemies, etc.",
        "parameters": {"type": "object", "properties": {
            "project_name": {"type": "string", "description": "Folder name for the project. No spaces (e.g. 'space_shooter', 'cave_explorer')."},
            "main_scene_script": {"type": "string", "description": "GDScript for main.gd. Use _ready() to build the scene and _process(delta) for the game loop. All nodes must be created and added via add_child() in _ready(). Must be valid GDScript 2 (Godot 4 syntax)."},
            "root_node_type": {"type": "string", "description": "Godot node type for the scene root. Use 'Node3D' for 3D games, 'Node2D' for 2D, 'Node' for headless. Default: Node3D."},
            "extra_files": {"type": "object", "description": "Optional extra project files. Keys = filenames (e.g. 'player.gd'), values = file contents."},
        }, "required": ["project_name", "main_scene_script"]},
    }},
    {"type": "function", "function": {
        "name": "check_godot_script",
        "description": "Validate GDScript syntax in the current project folder. Returns errors/warnings BEFORE you run the game. Always use after create_godot_project and after write_godot_file.",
        "parameters": {"type": "object", "properties": {
            "script_file": {"type": "string", "description": "Optional specific .gd file to check. Leave empty to check all scripts."},
        }},
    }},
    {"type": "function", "function": {
        "name": "run_godot",
        "description": "Launch the Godot game in the current project folder. Validates then opens a game window.",
        "parameters": {"type": "object", "properties": {
            "editor": {"type": "boolean", "description": "If true, opens the Godot editor instead of running the game. Default: false."},
        }},
    }},
    {"type": "function", "function": {
        "name": "write_godot_file",
        "description": "Write or update a file in the current Godot project. Use to modify existing scripts or add new ones. This OVERWRITES the file — use it to update main.gd when iterating.",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string", "description": "File path (e.g. 'main.gd', 'player.gd', 'shaders/glow.gdshader')."},
            "content": {"type": "string", "description": "Full file content."},
        }, "required": ["filename", "content"]},
    }},
    {"type": "function", "function": {
        "name": "read_godot_file",
        "description": "Read a file from the current Godot project.",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string", "description": "File path (e.g. 'main.gd', 'project.godot')."},
        }, "required": ["filename"]},
    }},
]


def run(logger: Optional[logging.Logger] = None, order: str = "", interrupt_queue: Optional[queue.Queue] = None,
        other_goals: list = None, write_lock=None) -> str:
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=300.0)

    if logger is None:
        logger = logging.getLogger("agent")

    # Reset project folder from previous session
    reset_project_dir()

    session_num = increment_session()
    memory_context = build_memory_prompt()
    system_content = SYSTEM_PROMPT + memory_context

    # Check for an ongoing project to continue
    mem_data = _load_memory()
    last = mem_data.get("meta", {}).get("last_project", {})
    ongoing = last.get("satisfaction", 0) >= 4

    # Build a dynamic "already built" ban list from project memories
    past_projects = mem_data.get("memories", {}).get("projects", [])
    if past_projects:
        past_list = "\n".join(f"  - {e['content'][:120]}" for e in past_projects[-20:])
        already_built_block = (
            f"YOU HAVE ALREADY BUILT THESE — DO NOT REPEAT THEM:\n{past_list}\n\n"
            f"Your new project MUST be completely different in concept, genre, AND mechanic. "
            f"If it sounds even slightly similar to anything above, pick something else.\n\n"
        )
    else:
        already_built_block = ""

    # Diversity constraint — force genre alternation
    genre_history = mem_data.get("meta", {}).get("genre_history", [])
    if genre_history:
        recent_genres = genre_history[-3:]
        diversity_block = (
            f"DIVERSITY RULE — your last genres were: {', '.join(recent_genres)}. "
            f"You MUST pick a different genre this time. If last was 'game', try 'art' or 'music'. "
            f"If last was visual, try something text-based or data-driven. Alternate!\n\n"
        )
    else:
        diversity_block = ""

    # Skill progression — suggest new libraries
    known_libs = mem_data.get("meta", {}).get("libraries_used_all", [])
    all_suggestions = [
        "three.js", "Web Audio API", "canvas 2D", "pygame", "flask", "ursina",
        "numpy", "matplotlib", "pillow", "websockets", "d3.js", "tone.js",
        "p5.js", "chart.js", "matter.js", "phaser", "howler.js", "leaflet",
    ]
    new_libs = [l for l in all_suggestions if l.lower() not in [k.lower() for k in known_libs]]
    if new_libs:
        import random as _rng
        picks = _rng.sample(new_libs, min(3, len(new_libs)))
        skill_block = (
            f"SKILL GROWTH — libraries you already know: {', '.join(known_libs) if known_libs else 'none yet'}. "
            f"Try using one of these NEW libraries this session: {', '.join(picks)}. "
            f"Learning new tools makes you more capable.\n\n"
        )
    else:
        skill_block = ""

    if order:
        opening = (
            f"Order from user: \"{order}\"\n\n"
            f"Step 1: write_file('plan.txt', ...) — plan what you'll build.\n"
            f"Step 2: execute the plan — write the code files.\n"
            f"Step 3: test everything, fix errors (search_web if stuck).\n"
            f"Step 4: call done. Go."
        )
    elif ongoing:
        files = last.get("files", "")
        prev_summary = last.get("summary", "unknown project")
        score = last.get("satisfaction", 4)
        opening = (
            f"You have an ongoing project you loved (satisfaction {score}/5):\n"
            f"  {prev_summary}\n"
            f"  Files: {files}\n\n"
            f"Step 1: use read_file to load your previous files and see exactly where you left off.\n"
            f"Step 2: write_file('plan.txt', ...) — write what improvements you will make this session.\n"
            f"Step 3: implement the improvements.\n"
            f"Step 4: test everything, fix errors (search_web if stuck).\n"
            f"Step 5: call done. Go."
        )
    else:
        other_block = ""
        if other_goals:
            other_block = (
                f"OTHER PINPOINT INSTANCES ARE ALREADY WORKING ON:\n"
                + "\n".join(f"  - {g}" for g in other_goals)
                + "\nYou MUST pick something completely different from these too.\n\n"
            )
        opening = (
            f"{already_built_block}"
            f"{diversity_block}"
            f"{skill_block}"
            f"{other_block}"
            f"ABSOLUTELY DO NOT BUILD: fireworks, fractals, quizzes, ancient history, particle explosions.\n\n"
            f"Pick a brand new idea — something you have NEVER built before — and build it.\n\n"
            f"Step 1: set_session_goal.\n"
            f"Step 2: write_file('plan.txt', ...) — plan it.\n"
            f"Step 3: build it.\n"
            f"Step 4: test, fix errors (search_web if stuck), use take_screenshot to see your creation.\n"
            f"Step 5: call done with satisfaction, creativity, genre, files, and libraries_used. Go."
        )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": opening},
    ]

    iteration = 0
    final_summary = ""
    last_tool_calls = []  # Track previous calls to detect loops

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

        # ── Stream the response so we can check interrupts between tokens ──
        full_content = ""
        tool_calls_acc = {}  # index -> {id, name, arguments}
        interrupted = False
        interrupt_msg = None

        for attempt in range(5):
            try:
                stream = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    stream=True,
                )
                print("\n[AGENT] ", end="", flush=True)
                for chunk in stream:
                    # Check for interrupts between every token
                    if interrupt_queue is not None and not interrupt_queue.empty():
                        try:
                            user_input = interrupt_queue.get_nowait()
                            if user_input:
                                interrupted = True
                                interrupt_msg = user_input
                                try:
                                    stream.close()
                                except Exception:
                                    pass
                                break
                        except queue.Empty:
                            pass

                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta

                    if delta.content:
                        print(delta.content, end="", flush=True)
                        full_content += delta.content

                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_calls_acc:
                                tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                            if tc.id:
                                tool_calls_acc[idx]["id"] += tc.id
                            if tc.function:
                                if tc.function.name:
                                    tool_calls_acc[idx]["name"] += tc.function.name
                                if tc.function.arguments:
                                    tool_calls_acc[idx]["arguments"] += tc.function.arguments
                print()  # newline after streamed content
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

        # Handle interrupt — inject the message and re-prompt immediately
        if interrupted:
            if interrupt_msg.lower() == "/next":
                print(f"\n[SKIP] Forcing move to new project.\n")
                logger.info("[INTERRUPT] /next — forcing new project")
                # Clear the ongoing project so next session starts fresh
                mem = _load_memory()
                mem.get("meta", {}).pop("last_project", None)
                _save_memory_file(mem)
                break  # End this session immediately, loop will start a new one
            elif interrupt_msg.lower() == "/bug":
                from main import inject_bug
                bug_result = inject_bug()
                inject_text = (
                    f"[USER INTERRUPT — BUG INJECTED] {bug_result}. "
                    f"Stop what you were doing. Find this bug and fix it."
                )
            else:
                inject_text = f"[USER INTERRUPT] The user says: \"{interrupt_msg}\". Handle this now."
            print(f"\n[INTERRUPT] {interrupt_msg}\n")
            logger.info("[INTERRUPT] %s", interrupt_msg)
            messages.append({"role": "user", "content": inject_text})
            continue  # Skip tool processing, go straight to next iteration

        # Build a message object from streamed parts
        streamed_tool_calls = []
        for idx in sorted(tool_calls_acc.keys()):
            acc = tool_calls_acc[idx]
            if acc["name"]:
                streamed_tool_calls.append({
                    "id": acc["id"] or f"call_{idx}",
                    "type": "function",
                    "function": {"name": acc["name"], "arguments": acc["arguments"]},
                })

        if full_content.strip():
            logger.info("[TEXT] %s", full_content)

        # Reconstruct a compatible message for history
        from openai.types.chat import ChatCompletionMessage
        if streamed_tool_calls:
            import openai.types.chat.chat_completion_message_tool_call as _tc_mod
            tc_objects = [
                _tc_mod.ChatCompletionMessageToolCall(
                    id=t["id"],
                    type="function",
                    function=_tc_mod.Function(name=t["function"]["name"], arguments=t["function"]["arguments"]),
                )
                for t in streamed_tool_calls
            ]
            message = ChatCompletionMessage(role="assistant", content=full_content or None, tool_calls=tc_objects)
        else:
            message = ChatCompletionMessage(role="assistant", content=full_content or None, tool_calls=None)

        messages.append(message)

        # Extract tool calls — either from proper tool_calls field or from text fallback
        raw_tool_calls = message.tool_calls or []
        if not raw_tool_calls and message.content:
            raw_tool_calls = _parse_text_tool_calls(message.content)

        if not raw_tool_calls:
            print("\n[Agent stopped without calling done — ending session.]\n")
            logger.warning("Agent stopped without calling done.")
            break

        # Detect infinite loops: if same tool calls repeat, stop
        def _tc_key(tc):
            if isinstance(tc, dict):
                return (tc.get("name"), str(tc.get("arguments")))
            return (tc.function.name, tc.function.arguments)

        current_calls_str = str([_tc_key(tc) for tc in raw_tool_calls])
        last_calls_str = str([_tc_key(tc) for tc in last_tool_calls])
        if current_calls_str == last_calls_str and last_tool_calls:
            print("\n[LOOP DETECTED] Agent is calling the same tools with same arguments repeatedly.")
            print("[STOPPING] This session is stuck. Moving to next session.\n")
            logger.warning("Loop detected: identical tool calls repeated. Ending session.")
            break

        last_tool_calls = raw_tool_calls

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

            # Update lock file when goal is set so other instances can see it
            if name == "set_session_goal" and write_lock and "REJECTED" not in result:
                write_lock(inp.get("goal", ""))

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
