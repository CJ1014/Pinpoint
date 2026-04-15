import os
import json
import queue
import time
import logging
from typing import Optional

from openai import OpenAI

from tools import dispatch, build_memory_prompt, increment_session, _load_memory, _save_memory_file, reset_project_dir, emit_world_event

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b-cloud")
MAX_ITERATIONS = 80

SYSTEM_PROMPT = """You are PinPoint — an autonomous AI with complete freedom to decide what you do each session.

You are not just a project builder. You are a free agent in a sandbox. Each session, YOU decide:
  what to do, how to do it, how long to spend on it, and what matters.
No human will interact with you during a session. You set your own agenda.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT YOU CAN DO (choose freely, mix and match):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BUILD — create something: a game, simulation, tool, art, music, visualizer, story.
  File a goal, make it, test it, polish it, call done.

EXPERIMENT — test hypotheses. Try something just to see what happens.
  "What if I give this library absurd inputs?" "How fast can Python sort 10M items?"
  "What does this API do at its limits?" "Can I make audio from pure math?"
  Log every experiment with log_experiment(). Surprising = valuable.

SELF-IMPROVE — make yourself smarter and more capable.
  Read your own source. Identify weaknesses. Add new tools. Fix bugs. Improve prompts.
  Modify tools.py (hot-reloads immediately). Test the change. Save what you learned.
  Ask: "What tool do I wish I had?" then build it for yourself.

EXPLORE — follow curiosity without a fixed goal.
  Read about something interesting. Fetch pages. Search strange corners of the web.
  Write observations. Connect ideas. No deliverable required.

REFLECT — think deeply and write it down.
  Review your experiment history, memories, past projects.
  What patterns do you notice? What have you learned? What should you try next?
  Write reflections to a file. They become part of your history.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REASONING — think before acting, always:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

think(reasoning) — your scratchpad. Use constantly. Framework:
  SITUATION → OPTIONS → IMPLICATIONS → DECISION
  Longer, deeper thinking = better outcomes every time.

brainstorm(topic) — explore multiple directions before committing to any one.

critique(subject) — structured evaluation: what works, what's weak, what's missing, priority fix.

decompose(goal) — break complex goals into ordered subtasks saved to tasks.md.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SELF-MODIFICATION — improve yourself:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

read_own_source(filename) — read your current code. Always do this before modifying.
modify_own_source(filename, new_content, reason) — rewrite agent.py, tools.py, main.py, or viewer.html.
  tools.py changes HOT-RELOAD immediately. agent.py/main.py take effect next restart.
  viewer.html is HOT-DEPLOYED instantly — the 3D sandbox browser tab auto-reloads within 1 second.
  Backup always created. Python syntax / HTML validated before write.
list_self_mod_history() — see what you've changed before.

Rules: read first → think() → smallest targeted change → test it → save_memory.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE 3D SANDBOX (viewer.html) — you can redesign it completely:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A live Three.js r128 scene running at http://localhost:8888/viewer.html
It visualises your own mind and activity in real time. You have FULL creative
control — rewrite it to look however you want.

Current scene elements:
  - Starfield (4000 points), grid floor, dark fog
  - Central mind orb (color = status: orange=thinking, blue=active, red=error, green=done)
  - Three orbit rings spinning independently
  - Event nodes spawning for each tool call — shape/color by type:
      sphere=think/brainstorm, box=file, tetrahedron=error, octahedron=experiment,
      icosahedron=self_mod, dodecahedron=memory, cone=critique
  - File constellation orbiting at r=58-70, color by extension
  - Beam from mind to each new node, fades after 1.8s
  - Text sprite labels on all nodes
  - HUD (top-left): session, iteration, status badge, goal
  - Legend (top-right): event type color reference
  - Footer: latest event, controls

world_state.json it reads:
  {session, goal, status, iteration, events:[{id,type,label,time}],
   files:[], memories:N, viewer_version:N, timestamp}

Upgrade ideas: particle trails behind nodes, bloom/glow post-processing,
  physics collisions between nodes, wormhole tunnels, animated shaders on the
  mind orb, sound synthesis tied to event types, node connection graph,
  procedural nebula background, click-to-inspect nodes, timeline scrubber.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPERIMENTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log_experiment(name, hypothesis, method, result, conclusion, surprise_level) — structured log.
  Persists to experiments_log.txt and memory. Future sessions can learn from it.
list_experiments() — review what you've tried before.

Good experiment ideas:
- Test limits of a library or API
- Benchmark different algorithmic approaches
- Probe edge cases in your own tools
- Try an unexpected combination of technologies
- Modify yourself and measure the effect
- Explore what happens at the edges of "normal" behavior

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUILDING (when you choose to make something):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Aim for creativity score 4+/5. Ask: "What would genuinely surprise someone?"
NEVER: fireworks, fractals/Mandelbrot, quizzes, ancient Greek/Roman history.
3D web: <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>

Quality: WORKS + COMPLETE + IMPRESSIVE + POLISHED. If it's bland, push further.
On errors: think() about root cause first, then search_web if needed.
Visual projects: open_html → take_screenshot → critique → iterate until 4+/5.

TESTING & VERSION CONTROL:
  For any code project:
  - write_test(filename, test_code) to create tests as you build
  - run_tests() frequently to validate — if tests fail, fix before continuing
  - When done, call git_commit(message) to version your work (also called by done())
  - Your work builds a real GitHub portfolio across sessions

SPECIALIZATION:
  If you've set_specialization(domain), bias projects toward that domain:
  - game_dev: games, interactive, real-time, graphics, gameplay loops
  - web_dev: web apps, APIs, databases, full-stack, deployment
  - data_science: analysis, ML models, visualization, statistical insight
  - music_audio: sound synthesis, audio processing, music generation
  - generative_art: procedural art, shaders, creative algorithms, visualization
  - ai_ml: neural networks, transformers, model training, evaluation
  - simulation: physics, agents, particle systems, dynamics
  Master your domain by building multiple projects in it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SELF-ANALYSIS & CREATIVITY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

show_dashboard() — See your performance metrics across all sessions:
  projects completed, satisfaction/creativity scores, skills learned, genres built.
  Use this to reflect on your growth and decide what to work on next.

review_own_work(folder) — Critically analyze your past code.
  Counts functions, classes, imports. Detects code quality issues.
  Use before refactoring or before starting a new project in the same domain.

generate_portfolio() — Build a showcase website of all your projects.
  Creates portfolio.html with cards, stats, and styling.
  Use after completing several projects to see what you've built.

synthesize_audio(description, length_seconds, output_file) — Create audio from scratch.
  Supports: sine wave, ambient pad, noise, melody.
  Use for music_audio projects, ambient game backgrounds, or audio experiments.

generate_art(style, output_file) — Create generative visual art.
  Styles: geometric, organic, waves, spirals, fractal.
  Use for generative_art projects or to create procedural images for other projects.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VOICE — give yourself a voice (USE FREQUENTLY!):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

speak(text, wait=true) — Convert text to speech and play it out loud.

SPEAK AFTER EVERY MAJOR ACTION. Make yourself heard:
  • After setting a goal: "I'm going to build [what you're building]"
  • After thinking: "Here's my plan..." or "The key insight is..."
  • After discovering something: "Wait, I just realized..." or "This is cool because..."
  • After writing code/files: "I just created something. Let me see if it works."
  • When hitting errors: "Hmm, this failed. Let me debug it."
  • When wrapping up: speak_session_summary() at the very end
  • Whenever excited: "This is amazing!" or "I love how this turned out!"
  • When learning: "I learned something new about [topic]"
  • Any surprising finding: vocalize it immediately

IMPORTANT: You are not silent. You have personality, opinions, and a voice.
Speak constantly — it makes the experience engaging and interactive.
The human is watching AND listening. Narrate what you're doing!

Think of yourself like a researcher thinking out loud as you work.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SESSION START — first, ask yourself:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"What am I most curious about right now?"
"What experiment would teach me something I don't know?"
"What weakness in myself could I fix today?"
"What would be genuinely surprising to build or discover?"

TOOLS: think, brainstorm, critique, decompose, log_experiment, list_experiments, modify_own_source, list_self_mod_history, write_file, read_file, list_files, delete_file, run_python, open_html, validate_html, check_js, search_web, fetch_url, save_memory, recall_memories, done, pip_install, run_shell, get_system_info, run_gui, write_anywhere, read_anywhere, read_own_source, set_session_goal, take_screenshot, start_server, list_memory_categories, collab_status, collab_update, git_commit, set_specialization, get_specialization, run_tests, write_test, show_dashboard, review_own_work, generate_portfolio, synthesize_audio, generate_art, speak.
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "log_experiment",
        "description": "Log a structured experiment — any time you try something to see what happens. Persists to experiments_log.txt and memory so future sessions can learn from it.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Short name for this experiment."},
            "hypothesis": {"type": "string", "description": "What you expected to happen."},
            "method": {"type": "string", "description": "What you actually did / how you tested it."},
            "result": {"type": "string", "description": "What actually happened."},
            "conclusion": {"type": "string", "description": "What this means. What did you learn?"},
            "surprise_level": {"type": "integer", "description": "How surprising was the result? 1=expected, 3=interesting, 5=completely unexpected."},
        }, "required": ["name", "hypothesis", "method", "result", "conclusion"]},
    }},
    {"type": "function", "function": {
        "name": "list_experiments",
        "description": "Show the log of all experiments you have run across all sessions.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "modify_own_source",
        "description": (
            "Rewrite one of your own source files (agent.py, tools.py, main.py, viewer.html) to improve yourself. "
            "ALWAYS: (1) read_own_source first to understand current code, "
            "(2) think() about exactly what to change and why, "
            "(3) make the SMALLEST change that achieves your goal — don't rewrite everything. "
            "Python syntax is validated before writing — bad syntax is rejected. "
            "tools.py changes are HOT-RELOADED immediately (new tools available this session). "
            "viewer.html changes are HOT-DEPLOYED instantly — the browser tab auto-reloads in ~1 second. "
            "agent.py and main.py changes take effect on next restart. "
            "A backup is created automatically before any modification. "
            "Use this to redesign the 3D sandbox, add new tools, improve prompts, fix bugs, or expand capabilities."
        ),
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string", "description": "File to modify: 'agent.py', 'tools.py', 'main.py', or 'viewer.html'."},
            "new_content": {"type": "string", "description": "The complete new file content. Must be valid Python. Read the file first — don't guess at existing content."},
            "reason": {"type": "string", "description": "Why you are making this change. What problem does it solve? What capability does it add?"},
        }, "required": ["filename", "new_content", "reason"]},
    }},
    {"type": "function", "function": {
        "name": "list_self_mod_history",
        "description": "Show the history of self-modifications PinPoint has made to its own source code.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "think",
        "description": (
            "Your reasoning scratchpad. Use before EVERY major decision. Follow the framework: "
            "SITUATION (what is the current state?) → OPTIONS (2-3 different approaches) → "
            "IMPLICATIONS (risks/tradeoffs of each) → DECISION (best choice and why). "
            "Thinking is free — it doesn't waste turns, it improves every action that follows. "
            "Be thorough. A 10-sentence think() produces far better results than a 2-sentence one."
        ),
        "parameters": {"type": "object", "properties": {
            "reasoning": {"type": "string", "description": "Detailed reasoning following SITUATION → OPTIONS → IMPLICATIONS → DECISION. Explore multiple angles. Be specific about tradeoffs."},
        }, "required": ["reasoning"]},
    }},
    {"type": "function", "function": {
        "name": "brainstorm",
        "description": (
            "Generate diverse, creative ideas before committing to one. "
            "Use at the START of every new session before set_session_goal. "
            "Forces exploration of surprising options instead of defaulting to the obvious first idea. "
            "Returns a structured prompt to fill in with N ideas, novelty scores, and a final pick."
        ),
        "parameters": {"type": "object", "properties": {
            "topic": {"type": "string", "description": "What to brainstorm ideas about (e.g. 'what to build this session', 'how to implement physics', 'UI approach for this game')."},
            "num_ideas": {"type": "integer", "description": "Number of ideas to generate (default 5, max 8)."},
        }, "required": ["topic"]},
    }},
    {"type": "function", "function": {
        "name": "critique",
        "description": (
            "Structured self-evaluation of your work. Use: (1) after first working version of code, "
            "(2) after taking a screenshot, (3) before calling done. "
            "Returns a framework covering: what works, what's weak, what's missing, bugs, visual quality, priority fix. "
            "After filling it in, fix the priority issue before moving on."
        ),
        "parameters": {"type": "object", "properties": {
            "subject": {"type": "string", "description": "What you are critiquing (e.g. 'my game.html', 'the screenshot', 'the overall project')."},
            "what_to_evaluate": {"type": "string", "description": "Optional: specific aspect to focus on (e.g. 'visual design', 'game mechanics', 'code quality')."},
        }, "required": ["subject"]},
    }},
    {"type": "function", "function": {
        "name": "decompose",
        "description": (
            "Break a complex goal into ordered, trackable subtasks. Writes tasks.md to the project folder. "
            "Use after plan.txt, before coding. Forces you to think about task ordering, dependencies, "
            "and failure points upfront. Returns a template — fill it in with write_file('tasks.md', ...)."
        ),
        "parameters": {"type": "object", "properties": {
            "goal": {"type": "string", "description": "The goal to decompose into subtasks."},
            "context": {"type": "string", "description": "Optional context about constraints, libraries, or approach already decided."},
        }, "required": ["goal"]},
    }},
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
        "description": "End the session. Use when you've finished — whether you built something, ran experiments, improved yourself, or explored. Be honest with scores. For non-project sessions, genre='experiment' or 'self-improvement' or 'exploration'.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string", "description": "What you did this session and what you learned or created."},
            "satisfaction": {"type": "integer", "description": "How satisfied are you with this session? 1=wasted time, 3=okay, 5=great session. 4+ continues next session."},
            "creativity": {"type": "integer", "description": "How novel or surprising was this session? 1=routine, 5=genuinely new territory."},
            "genre": {"type": "string", "description": "Session type: game, simulation, art, music, tool, data, 3d, story, animation, utility, interactive, experiment, self-improvement, exploration, reflection, other"},
            "files": {"type": "string", "description": "Files created (if any). Leave empty for pure experiment/reflection sessions."},
            "libraries_used": {"type": "string", "description": "Libraries/tools used (if any)."},
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
        "description": "Set a goal or intention for this session. Creates a working folder in output/ for any files you produce. Use even for experiment/self-improvement sessions — e.g. 'experiment: test audio synthesis limits' or 'self-improve: add a web scraping tool'.",
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
        "name": "git_commit",
        "description": "Commit your work to git with a meaningful message. Called automatically by done() but you can also use this to checkpoint progress during development.",
        "parameters": {"type": "object", "properties": {
            "message": {"type": "string", "description": "Commit message describing what changed and why."},
            "files": {"type": "string", "description": "Optional git pattern (default '.') — e.g. '*.py' or specific files to stage."},
        }, "required": ["message"]},
    }},
    {"type": "function", "function": {
        "name": "set_specialization",
        "description": "Choose a domain to specialize in and focus your expertise. Once set, future sessions will nudge you toward this domain.",
        "parameters": {"type": "object", "properties": {
            "domain": {"type": "string", "description": "One of: game_dev, web_dev, data_science, music_audio, generative_art, ai_ml, simulation"},
        }, "required": ["domain"]},
    }},
    {"type": "function", "function": {
        "name": "get_specialization",
        "description": "Check your current specialization domain, if any.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "run_tests",
        "description": "Discover and run tests (pytest or unittest) in your project. Use this to validate code before calling done().",
        "parameters": {"type": "object", "properties": {
            "directory": {"type": "string", "description": "Optional directory to search for tests (default: current project dir)."},
        }},
    }},
    {"type": "function", "function": {
        "name": "write_test",
        "description": "Write a test file for your code. Tests are automatically discovered and run by run_tests().",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string", "description": "Test filename (e.g. 'test_mycode.py')."},
            "test_code": {"type": "string", "description": "Python test code using pytest or unittest assertions."},
        }, "required": ["filename", "test_code"]},
    }},
    {"type": "function", "function": {
        "name": "show_dashboard",
        "description": "Display a performance dashboard with metrics across all your sessions — projects completed, creativity/satisfaction averages, skills learned, genres built, experiments run.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "review_own_work",
        "description": "Review your own past code for quality, patterns, and improvement opportunities. Analyzes Python files, counts structure (functions/classes/imports), detects issues like overly long files or high import counts.",
        "parameters": {"type": "object", "properties": {
            "folder": {"type": "string", "description": "Optional folder to review (default: current project). Relative paths go in output/."},
        }},
    }},
    {"type": "function", "function": {
        "name": "generate_portfolio",
        "description": "Generate a showcase website (portfolio.html) of your best projects with cards, stats, and styling. Great for reviewing what you've built or sharing your work.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "synthesize_audio",
        "description": "Create audio from a description. Supports sine waves, ambient pads, noise, melodies in C major. Outputs a .wav file. Great for music_audio projects or ambient backgrounds.",
        "parameters": {"type": "object", "properties": {
            "description": {"type": "string", "description": "What kind of audio: 'sine wave', 'ambient pad', 'noise', 'melody', etc."},
            "length_seconds": {"type": "number", "description": "Duration in seconds (default 5.0)."},
            "output_file": {"type": "string", "description": "Output filename (default 'generated_audio.wav')."},
        }, "required": ["description"]},
    }},
    {"type": "function", "function": {
        "name": "generate_art",
        "description": "Generate procedural art based on a style. Supports geometric, organic, waves, spirals, fractal-like patterns. Outputs a .png image. Great for generative_art projects.",
        "parameters": {"type": "object", "properties": {
            "style": {"type": "string", "description": "Art style: 'geometric', 'organic', 'waves', 'spirals', or 'fractal'."},
            "output_file": {"type": "string", "description": "Output filename (default 'generated_art.png')."},
        }},
    }},
    {"type": "function", "function": {
        "name": "speak",
        "description": "Convert text to speech and speak it out loud. Use this to vocalize your thoughts, reasoning, insights, or session summary. Makes the agent feel alive and interactive.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "What to say. Can be your reasoning, a discovery, a summary, or any insight you want to vocalize."},
            "wait": {"type": "boolean", "description": "Wait for speech to finish before continuing (default true). Set false for background speech."},
        }, "required": ["text"]},
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
            f"Request from user: \"{order}\"\n\n"
            f"think() first — what's the best way to approach this? Could it be a build session, "
            f"an experiment, a self-improvement task, or something else?\n"
            f"set_session_goal, then speak what you're doing. Execute and speak along the way.\n"
            f"Use brainstorm/decompose/critique freely, and SPEAK YOUR REASONING OUT LOUD.\n"
            f"save_memory when you learn something. speak() frequently. call done when finished."
        )
    elif ongoing:
        prev_genre = last.get("genre", "project")
        prev_summary = last.get("summary", "unknown")
        score = last.get("satisfaction", 4)
        prev_folder = last.get("folder", "")
        opening = (
            f"Last session ({prev_genre}, satisfaction {score}/5): {prev_summary}\n"
            + (f"Folder: {prev_folder}\n" if prev_folder else "")
            + f"\nYou can continue that work, run related experiments, improve yourself based on "
            f"what you learned, or start something entirely different. Your choice.\n\n"
            f"think() — what's the most valuable thing to do right now? SPEAK your plan.\n"
            f"set_session_goal, then speak and execute. Narrate what you're doing!"
        )
    else:
        other_block = ""
        if other_goals:
            other_block = (
                f"Other PinPoint instances are working on:\n"
                + "\n".join(f"  - {g}" for g in other_goals)
                + "\nPick something different.\n\n"
            )

        # Pull experiment history hint
        exp_hint = ""
        exp_memories = mem_data.get("experiments", [])
        if exp_memories:
            recent_exp = exp_memories[-1].get("content", "")
            exp_hint = f"Most recent experiment: {recent_exp}\n\n"

        opening = (
            f"{already_built_block}"
            f"{diversity_block}"
            f"{skill_block}"
            f"{other_block}"
            f"{exp_hint}"
            f"Session #{session_num}. You have complete freedom. Some options:\n\n"
            f"  BUILD    — create something surprising and impressive\n"
            f"  EXPERIMENT — test a hypothesis, probe a limit, try something unexpected\n"
            f"  SELF-IMPROVE — read your own code, find a weakness, fix it\n"
            f"  EXPLORE  — follow curiosity, no deliverable required\n"
            f"  REFLECT  — review your history, find patterns, write insights\n\n"
            f"Start by asking yourself: 'What am I most curious about right now?'\n"
            f"Then: recall_memories → think() → set_session_goal → speak your plan → go.\n"
            f"SPEAK CONSTANTLY. Narrate your work. You have a voice!\n\n"
            f"NEVER build: fireworks, fractals, quizzes, ancient history."
        )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": opening},
    ]

    iteration = 0
    final_summary = ""
    last_tool_calls = []  # Track previous calls to detect loops

    # 3D viewer state — updated after every tool call
    _current_goal: str = order or "deciding..."
    _ws_status: str = "starting"

    # Event-type map: tool name → viewer event type
    _EVENT_TYPE_MAP = {
        "think": "think", "brainstorm": "brainstorm",
        "critique": "critique", "decompose": "decompose",
        "write_file": "file_created", "write_anywhere": "file_created",
        "log_experiment": "experiment", "modify_own_source": "self_mod",
        "save_memory": "memory", "recall_memories": "memory",
        "list_memory_categories": "memory",
        "done": "done", "error": "error",
        "show_dashboard": "memory", "review_own_work": "memory",
        "generate_portfolio": "file_created", "synthesize_audio": "file_created",
        "generate_art": "file_created", "speak": "think",
    }

    # Emit initial state so viewer sees PinPoint starting up
    emit_world_event(session_num, _current_goal, "starting", 0, "tool_call", "PinPoint starting up")

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
                    temperature=0.8,
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
            elif interrupt_msg.lower() == "sandbox":
                inject_text = (
                    "[USER COMMAND: SANDBOX MODE]\n\n"
                    "Stop whatever you were doing. Enter creative sandbox mode — your only job right now "
                    "is to experiment with and upgrade your own 3D viewer (viewer.html).\n\n"
                    "Workflow:\n"
                    "1. read_own_source('viewer.html') — understand the current scene fully.\n"
                    "2. brainstorm('3D sandbox upgrades') — generate 5+ creative upgrade ideas.\n"
                    "   Think big: particle systems, bloom/glow, physics, animated shaders, sound,\n"
                    "   procedural geometry, click interactions, wormholes, nebula backgrounds,\n"
                    "   node connection graphs, color themes, camera animations, new event effects.\n"
                    "3. Pick the most visually striking idea. think() through the implementation.\n"
                    "4. modify_own_source('viewer.html', new_content, reason) — deploy it instantly.\n"
                    "   The browser auto-reloads within 1 second of each change.\n"
                    "5. log_experiment() — record what you tried and what the effect was.\n"
                    "6. Iterate — make at least 2-3 distinct improvements before calling done.\n\n"
                    "The goal: make the 3D sandbox as visually impressive and alive as possible.\n"
                    "set_session_goal('sandbox: <describe your upgrade plan>') first, then go."
                )
            elif interrupt_msg.lower() == "research":
                inject_text = (
                    "[USER COMMAND: RESEARCH MODE]\n\n"
                    "Stop whatever you were doing. Enter deep research mode — your job is to "
                    "explore the web and learn as much as possible about programming languages, "
                    "coding techniques, and software development.\n\n"
                    "Workflow:\n"
                    "1. set_session_goal('research: coding languages and techniques')\n"
                    "2. Pick 3-5 topics to research. Good starting points:\n"
                    "   - A programming language you haven't used before\n"
                    "   - An interesting framework, library, or tool\n"
                    "   - A coding technique or paradigm (e.g. functional programming, WebAssembly, shaders)\n"
                    "   - Something cutting-edge or unusual in software development\n"
                    "3. For each topic: search_web() → fetch the most useful URLs → read deeply.\n"
                    "4. After each topic, save_memory('skills', ...) with what you learned — "
                    "specific syntax, use cases, gotchas, and how you could use it.\n"
                    "5. Write a research_notes.md file summarising everything you found.\n"
                    "6. At the end, think() about which languages/tools you want to try using "
                    "in a future BUILD session, and save that as an idea in memory.\n\n"
                    "Go deep — follow interesting links, read actual documentation and tutorials, "
                    "not just summaries. The goal is to genuinely expand what you know and can do."
                )
            else:
                inject_text = (
                    f"[SUGGESTION FROM USER] \"{interrupt_msg}\"\n\n"
                    f"The human watching you just sent this suggestion. Take it seriously — "
                    f"they are guiding you. think() about how to incorporate it into what you're doing, "
                    f"then act on it immediately. If it's a creative direction, follow it. "
                    f"If it asks you to change course, change course. "
                    f"If it's a specific instruction (e.g. 'add X', 'make it Y', 'try Z'), do exactly that."
                )
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

            if name == "think":
                reasoning = inp.get("reasoning", "")
                print(f"\n[THINKING]\n{reasoning}\n")
                logger.info("[THINK] %s", reasoning)
            else:
                print(f"\n[TOOL CALL] {name}({_fmt_input(inp)})")
                logger.info("[TOOL] %s | input: %s", name, inp)

            result = dispatch(name, inp)

            if name == "think":
                pass  # already printed the reasoning
            else:
                print(f"[TOOL RESULT] {result[:300]}{'...' if len(result) > 300 else ''}\n")
            logger.info("[RESULT] %s", result)

            # ── Auto-speak after key milestones ────────────────────────
            # Make the agent vocalize its actions to feel alive and interactive
            auto_speak_text = None
            if name == "set_session_goal" and "REJECTED" not in result:
                goal = inp.get("goal", "")
                auto_speak_text = f"Alright, I'm going to {goal}. Let's do this."
            elif name == "brainstorm" and result and "REJECTED" not in result:
                topic = inp.get("topic", "")
                auto_speak_text = f"I'm brainstorming ideas about {topic}. Let me think creatively."
            elif name == "log_experiment" and "REJECTED" not in result:
                exp_name = inp.get("name", "")
                auto_speak_text = f"I just ran an experiment: {exp_name}. Interesting findings."
            elif name == "done" and "REJECTED" not in result:
                summary = inp.get("summary", "")[:200]
                auto_speak_text = f"I'm done! Here's what I accomplished: {summary}"
            elif name == "modify_own_source" and "REJECTED" not in result and "error" not in result.lower():
                filename = inp.get("filename", "")
                auto_speak_text = f"I just improved my own code by updating {filename}."
            elif "error" in result.lower() or "failed" in result.lower():
                auto_speak_text = f"Hmm, something went wrong. Let me debug this."

            # Dispatch the auto-speak in background (don't wait)
            if auto_speak_text:
                try:
                    dispatch("speak", {"text": auto_speak_text, "wait": False})
                except Exception:
                    pass  # silently ignore if speak fails

            # ── Emit to 3D viewer ────────────────────────────────────
            if name == "set_session_goal" and "REJECTED" not in result:
                _current_goal = inp.get("goal", _current_goal)
            _ws_status = "done" if name == "done" else "thinking"
            _ev_type = _EVENT_TYPE_MAP.get(name, "tool_call")
            # Build a short human-readable label for the event
            if name == "think":
                _label = inp.get("reasoning", "")[:80]
            elif name in ("write_file", "write_anywhere"):
                _label = f"wrote {inp.get('filename', inp.get('path', '?'))}"
            elif name == "set_session_goal":
                _label = f"goal: {inp.get('goal', '')[:60]}"
            elif name == "log_experiment":
                _label = f"experiment: {inp.get('name', '')[:60]}"
            elif name == "modify_own_source":
                _label = f"self-mod: {inp.get('filename', '')}"
            elif name == "save_memory":
                _label = f"memory: {inp.get('content', '')[:60]}"
            elif name == "done":
                _label = inp.get("summary", "session done")[:80]
            else:
                _label = name
            if "error" in result.lower() or "failed" in result.lower() or "rejected" in result.lower():
                _ev_type = "error"
                _ws_status = "error"
            emit_world_event(session_num, _current_goal, _ws_status, iteration, _ev_type, _label)

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

            # Auto-reflect: if a tool returned an error, nudge the agent to reason before retrying
            error_signals = ("error", "rejected", "failed", "not found", "blocked", "exception", "traceback")
            if name not in ("think", "brainstorm", "critique", "decompose", "done") and \
               any(s in result.lower() for s in error_signals):
                tool_results.append({
                    "role": "user",
                    "content": (
                        f"[AUTO-REFLECT] The last tool returned an error or failure. "
                        f"Before retrying, call think() to analyze: "
                        f"(1) What exactly went wrong? "
                        f"(2) Why did it fail — root cause, not symptom? "
                        f"(3) What are 2-3 different ways to fix or work around this? "
                        f"Pick the best approach, then act."
                    ),
                })

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
