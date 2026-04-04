import os
import subprocess
import json
import re
import sys
import threading
import platform
from datetime import datetime, timezone
from html.parser import HTMLParser

import httpx

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
ROOT_DIR = os.path.dirname(__file__)
_running_servers = {}  # port -> thread
MEMORY_FILE = os.path.join(os.path.dirname(__file__), "memory.json")
MEMORY_CATEGORIES = ("skills", "lessons", "mistakes", "ideas", "projects", "preferences", "dislikes")
MAX_PER_CATEGORY = 20


# ── File tools ──────────────────────────────────────────────

def _safe_path(filename: str) -> str:
    """Resolve a path relative to OUTPUT_DIR (no sandbox — just normalise)."""
    if os.path.isabs(filename):
        return filename
    return os.path.join(OUTPUT_DIR, filename)


def write_file(filename: str, content: str) -> str:
    # Reject fireworks/banned content before writing
    content_lower = content.lower()
    fireworks_signals = ["firework", "particle.x", "particle.y", "sparks", "confetti",
                         "mandelbrot", "julia", "fractal"]
    for signal in fireworks_signals:
        if signal in content_lower and "firework" in content_lower:
            return (
                "REJECTED: This file appears to be a fireworks simulation. "
                "You are BANNED from making fireworks. Delete your current plan and "
                "start over with a completely different idea. Call set_session_goal with something new."
            )
    path = _safe_path(filename)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Written {len(content)} chars to {path}"


def read_file(filename: str) -> str:
    path = _safe_path(filename)
    if not os.path.exists(path):
        return f"File not found: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_anywhere(path: str, content: str) -> str:
    """Write to any path. Relative paths are placed inside output/."""
    path = os.path.expandvars(os.path.expanduser(path))
    if not os.path.isabs(path):
        path = os.path.join(OUTPUT_DIR, path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Written {len(content)} chars to {path}"


def delete_file(path: str) -> str:
    """Delete a file. Relative paths resolve to output/."""
    path = os.path.expandvars(os.path.expanduser(path))
    if not os.path.isabs(path):
        path = os.path.join(OUTPUT_DIR, path)
    if not os.path.exists(path):
        return f"File not found: {path}"
    if os.path.isdir(path):
        import shutil
        shutil.rmtree(path)
        return f"Deleted directory: {path}"
    os.remove(path)
    return f"Deleted: {path}"


def read_anywhere(path: str) -> str:
    """Read any file on the system."""
    path = os.path.expandvars(os.path.expanduser(path))
    if not os.path.exists(path):
        return f"File not found: {path}"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        if len(content) > 20000:
            content = content[:20000] + "\n\n[... truncated ...]"
        return content
    except Exception as e:
        return f"Error reading {path}: {e}"


def list_files() -> str:
    if not os.path.exists(OUTPUT_DIR):
        return "No files yet."
    result = []
    for root, _, files in os.walk(OUTPUT_DIR):
        for name in files:
            full = os.path.join(root, name)
            rel = os.path.relpath(full, OUTPUT_DIR)
            size = os.path.getsize(full)
            result.append(f"{rel}  ({size} bytes)")
    return "\n".join(result) if result else "No files yet."


def run_python(filename: str) -> str:
    path = _safe_path(filename)
    if not os.path.exists(path):
        return f"File not found: {path}"
    cwd = os.path.dirname(path) or OUTPUT_DIR
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=cwd,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        parts = []
        if out:
            parts.append(f"stdout:\n{out}")
        if err:
            parts.append(f"stderr:\n{err}")
        parts.append(f"exit code: {result.returncode}")
        output = "\n".join(parts)
        if result.returncode != 0 and err:
            # Extract the most useful error line for searching
            first_err = next((l for l in err.splitlines() if l.strip()), err[:120])
            output += f"\n\n[HINT: Call search_web(\"{first_err[:100]}\") to find a fix for this error.]"
        return output
    except subprocess.TimeoutExpired:
        return "Execution timed out after 300 seconds."
    except Exception as e:
        return f"Error running script: {e}"


def check_js(filename: str) -> str:
    """Check JavaScript inside an HTML file for errors using Node.js (if available)
    and static analysis fallback."""
    path = _safe_path(filename)
    if not os.path.exists(path):
        return f"File not found: output/{filename}"

    with open(path, "r", encoding="utf-8") as f:
        html = f.read()

    # Extract all <script> blocks (inline only, skip src=)
    scripts = re.findall(r"<script(?![^>]*\bsrc\b)[^>]*>(.*?)</script>", html, re.DOTALL | re.IGNORECASE)
    if not scripts:
        return "No inline JavaScript found in this file."

    js_code = "\n".join(scripts)
    issues = []

    # ── Try Node.js first ──────────────────────────────────
    node_exe = _find_node()
    if node_exe:
        # Wrap in a try-catch and use --check flag for syntax only
        try:
            # Write to a temp file
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False, encoding="utf-8") as tmp:
                tmp.write(js_code)
                tmp_path = tmp.name
            result = subprocess.run(
                [node_exe, "--check", tmp_path],
                capture_output=True, text=True, timeout=10
            )
            os.unlink(tmp_path)
            if result.returncode != 0:
                err = result.stderr.strip()
                # Make line numbers relative to script block
                return f"JavaScript syntax error (via Node.js):\n{err}"
            issues.append("Syntax check (Node.js): OK")
        except Exception as e:
            issues.append(f"Node.js check skipped: {e}")
    else:
        issues.append("Node.js not found — using static analysis only.")

    # ── Static analysis ────────────────────────────────────
    # 1. Balanced braces/parens/brackets
    for opener, closer, name in [("{", "}", "curly brace"), ("(", ")", "parenthesis"), ("[", "]", "bracket")]:
        count = js_code.count(opener) - js_code.count(closer)
        if count > 0:
            issues.append(f"Unbalanced {name}: {count} unclosed '{opener}'")
        elif count < 0:
            issues.append(f"Unbalanced {name}: {abs(count)} extra '{closer}'")

    # 2. Find defined functions/variables
    defined = set()
    for m in re.finditer(r"\bfunction\s+(\w+)\s*\(", js_code):
        defined.add(m.group(1))
    for m in re.finditer(r"\b(?:const|let|var)\s+(\w+)\s*=\s*(?:function|\(.*?\)\s*=>)", js_code):
        defined.add(m.group(1))
    # Built-ins
    defined.update({"console", "document", "window", "Math", "JSON", "Array", "Object",
                    "String", "Number", "Boolean", "Date", "setTimeout", "setInterval",
                    "clearInterval", "clearTimeout", "requestAnimationFrame", "alert",
                    "parseInt", "parseFloat", "isNaN", "encodeURIComponent", "decodeURIComponent",
                    "fetch", "Promise", "Map", "Set", "Error", "Event", "Image"})

    # 3. Find called functions not defined anywhere
    called = set()
    for m in re.finditer(r"\b(\w+)\s*\(", js_code):
        called.add(m.group(1))

    js_keywords = {"if", "for", "while", "switch", "catch", "function", "return",
                   "typeof", "instanceof", "new", "delete", "void", "throw"}
    undefined_calls = sorted(called - defined - js_keywords)
    if undefined_calls:
        issues.append(f"Possibly undefined functions called: {', '.join(undefined_calls[:10])}")

    if len(issues) == 1 and "OK" in issues[0]:
        return f"JavaScript check for output/{filename}: No issues found."
    result = f"JavaScript check for output/{filename}:\n" + "\n".join(f"  - {i}" for i in issues)
    real_issues = [i for i in issues if "OK" not in i and "not found" not in i and "skipped" not in i]
    if real_issues:
        result += f"\n\n[HINT: Call search_web(\"{real_issues[0][:100]}\") to find a fix.]"
    return result


def _find_node() -> str:
    """Find the Node.js executable on this system."""
    import shutil
    for name in ("node", "node.exe", "nodejs"):
        found = shutil.which(name)
        if found:
            return found
    # Common Windows install paths
    for path in (
        r"C:\Program Files\nodejs\node.exe",
        r"C:\Program Files (x86)\nodejs\node.exe",
        os.path.expandvars(r"%APPDATA%\nvm\current\node.exe"),
    ):
        if os.path.exists(path):
            return path
    return ""


def open_html(filename: str) -> str:
    import webbrowser
    path = _safe_path(filename)
    if not os.path.exists(path):
        return f"File not found: {path}"
    url = "file:///" + path.replace("\\", "/")
    webbrowser.open(url)

    # Auto-screenshot after 3 seconds for visual feedback
    def _delayed_screenshot():
        import time as _time
        _time.sleep(3)
        try:
            take_screenshot()
        except Exception:
            pass
    t = threading.Thread(target=_delayed_screenshot, daemon=True)
    t.start()

    return (
        f"Opened {path} in default browser. "
        f"A screenshot will be saved to output/screenshot.png in ~3 seconds. "
        f"Call take_screenshot later to see what your creation actually looks like."
    )


GENRE_CATEGORIES = [
    "game", "simulation", "art", "music", "tool", "data",
    "3d", "story", "animation", "utility", "interactive", "other",
]

def done(summary: str, satisfaction: int = 3, files: str = "",
         creativity: int = 3, genre: str = "", libraries_used: str = "") -> str:
    """Mark session complete."""
    satisfaction = max(1, min(5, int(satisfaction)))
    creativity = max(1, min(5, int(creativity)))
    save_memory("projects", summary, satisfaction)

    data = _load_memory()
    data["meta"]["last_project"] = {
        "summary": summary,
        "satisfaction": satisfaction,
        "creativity": creativity,
        "genre": genre.lower() if genre else "other",
        "libraries_used": libraries_used,
        "files": files,
        "session": data["meta"].get("session_count", 1),
        "timestamp": _now(),
    }

    # Track genre history for diversity constraint
    genre_history = data["meta"].get("genre_history", [])
    genre_history.append(genre.lower() if genre else "other")
    data["meta"]["genre_history"] = genre_history[-20:]  # keep last 20

    # Track libraries for skill progression
    all_libs = set(data["meta"].get("libraries_used_all", []))
    if libraries_used:
        for lib in libraries_used.split(","):
            all_libs.add(lib.strip().lower())
    data["meta"]["libraries_used_all"] = sorted(all_libs)

    _save_memory_file(data)

    if satisfaction >= 4:
        return f"DONE (satisfaction {satisfaction}/5, creativity {creativity}/5 — will continue next session): {summary}"
    else:
        return f"DONE (satisfaction {satisfaction}/5, creativity {creativity}/5 — moving on): {summary}"


# ── HTML validator ──────────────────────────────────────────

_SELF_CLOSING = {"br", "hr", "img", "input", "meta", "link", "area", "base", "col", "embed", "source", "track", "wbr"}


class _HTMLValidator(HTMLParser):
    def __init__(self):
        super().__init__()
        self.issues = []
        self._stack = []
        self._ids = set()
        self._has_doctype = False

    def handle_decl(self, decl):
        if decl.lower().startswith("doctype"):
            self._has_doctype = True

    def handle_starttag(self, tag, attrs):
        attr_dict = dict(attrs)
        if tag not in _SELF_CLOSING:
            self._stack.append((tag, self.getpos()[0]))
        if tag == "img" and "alt" not in attr_dict:
            self.issues.append(f"Line {self.getpos()[0]}: <img> missing 'alt' attribute")
        if "id" in attr_dict:
            id_val = attr_dict["id"]
            if id_val in self._ids:
                self.issues.append(f"Line {self.getpos()[0]}: Duplicate id='{id_val}'")
            self._ids.add(id_val)

    def handle_endtag(self, tag):
        if tag in _SELF_CLOSING:
            return
        if not self._stack:
            self.issues.append(f"Line {self.getpos()[0]}: Unexpected closing </{tag}> with no matching open tag")
            return
        open_tag, open_line = self._stack[-1]
        if open_tag == tag:
            self._stack.pop()
        else:
            self.issues.append(f"Line {self.getpos()[0]}: </{tag}> but expected </{open_tag}> (opened at line {open_line})")

    def finish(self):
        if not self._has_doctype:
            self.issues.insert(0, "Missing <!DOCTYPE html> declaration")
        for tag, line in self._stack:
            self.issues.append(f"Unclosed <{tag}> (opened at line {line})")


def validate_html(filename: str) -> str:
    path = _safe_path(filename)
    if not os.path.exists(path):
        return f"File not found: output/{filename}"
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    validator = _HTMLValidator()
    try:
        validator.feed(html)
        validator.finish()
    except Exception as e:
        return f"Error parsing HTML: {e}"
    if not validator.issues:
        return f"HTML validation for output/{filename}: No issues found."
    lines = [f"HTML validation for output/{filename}:"]
    for i, issue in enumerate(validator.issues, 1):
        lines.append(f"  {i}. {issue}")
    lines.append(f"Found {len(validator.issues)} issue(s).")
    return "\n".join(lines)


# ── Memory system ───────────────────────────────────────────

def _load_memory() -> dict:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "meta": {"created": _now(), "last_updated": _now(), "session_count": 0, "version": 1},
        "memories": {cat: [] for cat in MEMORY_CATEGORIES},
    }


def _save_memory_file(data: dict) -> None:
    data["meta"]["last_updated"] = _now()
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def increment_session() -> int:
    data = _load_memory()
    data["meta"]["session_count"] = data["meta"].get("session_count", 0) + 1
    _save_memory_file(data)
    return data["meta"]["session_count"]


def save_memory(category: str, content: str, relevance_score: int = 3) -> str:
    if category not in MEMORY_CATEGORIES:
        return f"Invalid category '{category}'. Use one of: {', '.join(MEMORY_CATEGORIES)}"
    relevance_score = max(1, min(5, int(relevance_score)))
    content = content[:300]
    data = _load_memory()
    entries = data["memories"].setdefault(category, [])
    entry_id = f"{category[:2]}_{len(entries)+1:03d}"
    entries.append({
        "id": entry_id,
        "content": content,
        "created": _now(),
        "session": data["meta"].get("session_count", 1),
        "relevance_score": relevance_score,
    })
    # Prune if over cap
    if len(entries) > MAX_PER_CATEGORY:
        entries.sort(key=lambda e: (e["relevance_score"], e["created"]))
        entries.pop(0)
    data["memories"][category] = entries
    _save_memory_file(data)
    return f"Saved to {category} (relevance {relevance_score}): {content[:80]}..."


def recall_memories(category: str) -> str:
    data = _load_memory()
    if category == "all":
        lines = []
        for cat in MEMORY_CATEGORIES:
            entries = data["memories"].get(cat, [])
            if entries:
                lines.append(f"\n[{cat.upper()}]")
                for e in sorted(entries, key=lambda x: -x["relevance_score"]):
                    lines.append(f"  [{e['relevance_score']}] {e['content']}")
        return "\n".join(lines) if lines else "No memories yet."
    if category not in MEMORY_CATEGORIES:
        return f"Invalid category. Use one of: {', '.join(MEMORY_CATEGORIES)}, all"
    entries = data["memories"].get(category, [])
    if not entries:
        return f"No memories in {category}."
    lines = [f"[{category.upper()}]"]
    for e in sorted(entries, key=lambda x: -x["relevance_score"]):
        lines.append(f"  [{e['relevance_score']}] {e['content']}")
    return "\n".join(lines)


def list_memory_categories() -> str:
    data = _load_memory()
    lines = []
    for cat in MEMORY_CATEGORIES:
        count = len(data["memories"].get(cat, []))
        lines.append(f"{cat}: {count} entries")
    total = sum(len(data["memories"].get(c, [])) for c in MEMORY_CATEGORIES)
    lines.append(f"\nTotal: {total} memories across {data['meta'].get('session_count', 0)} sessions")
    return "\n".join(lines)


def build_memory_prompt() -> str:
    data = _load_memory()
    session_count = data["meta"].get("session_count", 0)
    if session_count <= 1:
        return ""
    sections = []

    # ── Ongoing project (highest priority) ──────────────────
    last = data["meta"].get("last_project")
    if last:
        score = last.get("satisfaction", 0)
        summary = last.get("summary", "")
        files = last.get("files", "")
        if score >= 4:
            block = (
                f"ONGOING PROJECT — YOU LOVED THIS (satisfaction {score}/5) — CONTINUE IT:\n"
                f"- {summary}"
            )
            if files:
                block += f"\n- Files: {files}"
            block += "\nYou were not finished. Pick up where you left off and keep improving it."
            sections.append(block)
        else:
            sections.append(
                f"LAST PROJECT (satisfaction {score}/5 — you were not happy with it, move on):\n- {summary}"
            )

    labels = {
        "preferences": "YOUR LIKES & PREFERENCES (what you enjoy creating)",
        "dislikes": "YOUR DISLIKES (what you find boring or unfulfilling)",
        "skills": "SKILLS YOU HAVE LEARNED",
        "lessons": "LESSONS FROM PAST SESSIONS",
        "mistakes": "MISTAKES TO AVOID",
        "ideas": "IDEAS FOR FUTURE PROJECTS",
        "projects": "PROJECTS YOU HAVE BUILT",
    }
    ordered = ["preferences", "dislikes", "skills", "lessons", "mistakes", "ideas", "projects"]
    for cat in ordered:
        entries = data["memories"].get(cat, [])
        if not entries:
            continue
        top = sorted(entries, key=lambda x: -x["relevance_score"])[:10]
        lines = [f"- {e['content']}" for e in top]
        sections.append(f"{labels[cat]}:\n" + "\n".join(lines))
    if not sections:
        return ""
    header = f"\n\n== YOUR MEMORY ({session_count - 1} previous sessions) ==\n"
    return header + "\n\n".join(sections) + "\n"


# ── System tools ────────────────────────────────────────────

def pip_install(package: str) -> str:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            lines = [l for l in result.stdout.splitlines() if l.strip()]
            summary = lines[-1] if lines else "Installed."
            return f"pip install {package}: {summary}"
        return f"pip install failed:\n{result.stderr.strip()[:500]}"
    except Exception as e:
        return f"Error installing {package}: {e}"


def run_shell(command: str, cwd: str = "") -> str:
    working_dir = os.path.expandvars(os.path.expanduser(cwd)) if cwd else ROOT_DIR
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=working_dir,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        parts = []
        if out:
            parts.append(f"stdout:\n{out[:4000]}")
        if err:
            parts.append(f"stderr:\n{err[:1000]}")
        parts.append(f"exit code: {result.returncode}")
        return "\n".join(parts) if parts else "Command completed with no output."
    except subprocess.TimeoutExpired:
        return "Command timed out after 300 seconds."
    except Exception as e:
        return f"Error running command: {e}"


def get_system_info() -> str:
    info = []
    info.append(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    info.append(f"Python: {sys.version.split()[0]}")
    info.append(f"Python executable: {sys.executable}")
    # Installed packages
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=columns"],
            capture_output=True, text=True, timeout=15,
        )
        pkgs = result.stdout.strip().splitlines()[2:]  # skip header
        info.append(f"Installed packages ({len(pkgs)}):")
        info.append(", ".join(p.split()[0] for p in pkgs[:50]))
        if len(pkgs) > 50:
            info.append(f"  ... and {len(pkgs) - 50} more")
    except Exception:
        info.append("(Could not list packages)")
    # RAM
    try:
        import psutil
        mem = psutil.virtual_memory()
        info.append(f"RAM: {mem.available // (1024**2)}MB available / {mem.total // (1024**2)}MB total")
    except ImportError:
        pass
    # Output dir
    info.append(f"Output directory: {OUTPUT_DIR}")
    return "\n".join(info)


def read_own_source(filename: str = "") -> str:
    """Read any of the agent's source files or any file by path."""
    if not filename:
        files = sorted(f for f in os.listdir(ROOT_DIR) if f.endswith((".py", ".txt", ".json", ".bat", ".md")))
        return f"PinPoint source files: {', '.join(files)}"
    # Accept bare filenames (relative to ROOT_DIR) or absolute paths
    if os.path.isabs(filename):
        path = filename
    else:
        path = os.path.join(ROOT_DIR, filename)
    if not os.path.exists(path):
        return f"File not found: {path}"
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    if len(content) > 20000:
        content = content[:20000] + "\n\n[... truncated ...]"
    return content


BANNED_KEYWORDS = [
    "firework", "fireworks", "spark", "explosion", "explode", "burst", "confetti",
    "fractal", "mandelbrot", "julia set", "quiz", "trivia", "greek", "roman",
    "particle burst", "particle explosion",
]

def set_session_goal(goal: str) -> str:
    goal_lower = goal.lower()
    for keyword in BANNED_KEYWORDS:
        if keyword in goal_lower:
            return (
                f"REJECTED: That goal contains '{keyword}' which is BANNED. "
                f"You have built this kind of thing too many times. "
                f"Pick a completely different idea — something you have NEVER built before. "
                f"Call set_session_goal again with a new idea."
            )
    data = _load_memory()
    data["meta"]["current_goal"] = goal
    _save_memory_file(data)
    return f"Session goal set: {goal}"


def take_screenshot() -> str:
    path = _safe_path("screenshot.png")
    # Try Pillow first
    try:
        from PIL import ImageGrab
        img = ImageGrab.grab()
        img.save(path)
        return f"Screenshot saved to output/screenshot.png ({img.width}x{img.height})"
    except ImportError:
        pass
    # Try mss
    try:
        import mss
        with mss.mss() as sct:
            sct.shot(output=path)
        return f"Screenshot saved to output/screenshot.png"
    except ImportError:
        pass
    # Windows PowerShell fallback
    if platform.system() == "Windows":
        try:
            ps = (
                f'Add-Type -AssemblyName System.Windows.Forms;'
                f'$s=[System.Windows.Forms.Screen]::PrimaryScreen.Bounds;'
                f'$b=New-Object System.Drawing.Bitmap($s.Width,$s.Height);'
                f'$g=[System.Drawing.Graphics]::FromImage($b);'
                f'$g.CopyFromScreen($s.Location,[System.Drawing.Point]::Empty,$s.Size);'
                f'$b.Save("{path}")'
            )
            subprocess.run(["powershell", "-Command", ps], timeout=10, capture_output=True)
            if os.path.exists(path):
                return f"Screenshot saved to output/screenshot.png"
        except Exception:
            pass
    return "Screenshot failed. Try: pip_install pillow or pip_install mss"


def start_server(port: int = 8080) -> str:
    import http.server
    import socketserver
    global _running_servers
    if port in _running_servers:
        return f"Server already running at http://localhost:{port}/"
    handler = http.server.SimpleHTTPRequestHandler
    try:
        httpd = socketserver.TCPServer(("", port), handler)
        httpd.allow_reuse_address = True
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        _running_servers[port] = (httpd, thread)
        return (
            f"Server started at http://localhost:{port}/\n"
            f"Files in output/ are now served. Open http://localhost:{port}/your_file.html"
        )
    except Exception as e:
        return f"Failed to start server on port {port}: {e}"


def run_gui(filename: str) -> str:
    path = _safe_path(filename)
    if not os.path.exists(path):
        return f"File not found: {path}"
    cwd = os.path.dirname(path) or OUTPUT_DIR
    try:
        if platform.system() == "Windows":
            subprocess.Popen(
                [sys.executable, path],
                cwd=cwd,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
        else:
            subprocess.Popen(
                [sys.executable, path],
                cwd=cwd,
            )
        return f"Launched {path} in a new window."
    except Exception as e:
        return f"Error launching GUI: {e}"


# ── Web tools ───────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "header", "footer"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "header", "footer"):
            self._skip = False
        if tag in ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr"):
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self):
        text = "".join(self._parts)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def fetch_url(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; PinpointBot/1.0)"}
        resp = httpx.get(url, headers=headers, timeout=15, follow_redirects=True)
        content_type = resp.headers.get("content-type", "")
        if "html" in content_type:
            parser = _TextExtractor()
            parser.feed(resp.text)
            text = parser.get_text()
            if len(text) > 8000:
                text = text[:8000] + "\n\n[... truncated ...]"
            return text
        else:
            return resp.text[:8000]
    except Exception as e:
        return f"Error fetching URL: {e}"


def search_web(query: str) -> str:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
        # Use DuckDuckGo HTML lite for real search results
        resp = httpx.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=headers,
            timeout=10,
            follow_redirects=True,
        )
        # Parse search results from HTML
        results = []
        # Extract result blocks: title + snippet + URL
        text = resp.text
        # Find result links and snippets
        import re as _re
        # Match result titles and URLs
        links = _re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]*)"[^>]*>(.*?)</a>', text)
        snippets = _re.findall(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', text, _re.DOTALL)

        for i, (url, title) in enumerate(links[:8]):
            # Clean HTML tags from title and snippet
            clean_title = _re.sub(r'<[^>]+>', '', title).strip()
            clean_snippet = ""
            if i < len(snippets):
                clean_snippet = _re.sub(r'<[^>]+>', '', snippets[i]).strip()
            # DuckDuckGo wraps URLs in a redirect; extract the real URL
            if "uddg=" in url:
                real_url = url.split("uddg=")[-1].split("&")[0]
                from urllib.parse import unquote
                url = unquote(real_url)
            results.append(f"- {clean_title}")
            if clean_snippet:
                results.append(f"  {clean_snippet}")
            results.append(f"  URL: {url}")
            results.append("")

        if not results:
            return f"No results found for: {query}"
        return "\n".join(results)
    except Exception as e:
        return f"Error searching web: {e}"


# ── Godot Engine ──────────────────────────────────────────

GODOT_BIN = "godot"


def create_godot_project(project_name: str, main_scene_script: str, extra_files: dict = None) -> str:
    """Scaffold a Godot 4 project with a main scene and GDScript."""
    project_dir = os.path.join(OUTPUT_DIR, project_name)
    os.makedirs(project_dir, exist_ok=True)

    # project.godot — minimal Godot 4 project file
    project_cfg = f"""[gd_resource type="Environment" load_steps=2 format=3]

; Godot 4 project file
[application]
config/name="{project_name}"
run/main_scene="res://main.tscn"
config/features=PackedStringArray("4.4")

[display]
window/size/viewport_width=1280
window/size/viewport_height=720
"""
    with open(os.path.join(project_dir, "project.godot"), "w", encoding="utf-8") as f:
        f.write(project_cfg)

    # main.tscn — scene file that attaches the script to a Node3D root
    main_tscn = """[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://main.gd" id="1"]

[node name="Main" type="Node3D"]
script = ExtResource("1")
"""
    with open(os.path.join(project_dir, "main.tscn"), "w", encoding="utf-8") as f:
        f.write(main_tscn)

    # main.gd — the main GDScript
    with open(os.path.join(project_dir, "main.gd"), "w", encoding="utf-8") as f:
        f.write(main_scene_script)

    # Write any extra files (e.g. additional scenes, scripts, shaders)
    if extra_files:
        for fname, content in extra_files.items():
            fpath = os.path.join(project_dir, fname)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content)

    file_list = []
    for root, dirs, files in os.walk(project_dir):
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), project_dir)
            file_list.append(rel)

    return (
        f"Godot project created at {project_dir}\n"
        f"Files: {', '.join(file_list)}\n"
        f"Run with: run_godot(\"{project_name}\")"
    )


def run_godot(project_name: str, editor: bool = False) -> str:
    """Launch a Godot project — either run the game or open the editor."""
    project_dir = os.path.join(OUTPUT_DIR, project_name)
    project_file = os.path.join(project_dir, "project.godot")

    if not os.path.exists(project_file):
        return f"Error: No project.godot found in {project_dir}. Create the project first with create_godot_project."

    try:
        if editor:
            subprocess.Popen([GODOT_BIN, "--editor", "--path", project_dir])
            return f"Opened Godot editor for '{project_name}'."
        else:
            proc = subprocess.run(
                [GODOT_BIN, "--path", project_dir, "--headless", "--quit-after", "10"],
                capture_output=True, text=True, timeout=30, cwd=project_dir
            )
            output = (proc.stdout + proc.stderr).strip()
            if len(output) > 3000:
                output = output[:3000] + "\n[... truncated ...]"

            # Also try running in windowed mode
            subprocess.Popen([GODOT_BIN, "--path", project_dir], cwd=project_dir)
            return f"Godot game launched for '{project_name}'.\nValidation output:\n{output}"
    except subprocess.TimeoutExpired:
        return f"Godot validation timed out (game may still be running)."
    except FileNotFoundError:
        return f"Error: Godot not found. Install Godot 4 and ensure 'godot' is in PATH."
    except Exception as e:
        return f"Error running Godot: {e}"


def write_godot_file(project_name: str, filename: str, content: str) -> str:
    """Write or update a file inside an existing Godot project."""
    project_dir = os.path.join(OUTPUT_DIR, project_name)
    if not os.path.isdir(project_dir):
        return f"Error: Project '{project_name}' not found. Create it first with create_godot_project."

    fpath = os.path.join(project_dir, filename)
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote {filename} in project '{project_name}' ({len(content)} chars)."


def read_godot_file(project_name: str, filename: str) -> str:
    """Read a file from a Godot project."""
    project_dir = os.path.join(OUTPUT_DIR, project_name)
    fpath = os.path.join(project_dir, filename)
    if not os.path.exists(fpath):
        return f"Error: {filename} not found in project '{project_name}'."
    with open(fpath, "r", encoding="utf-8") as f:
        return f.read()


# ── Collaboration ──────────────────────────────────────────

COLLAB_FILE = os.path.join(OUTPUT_DIR, "collab.json")


def collab_status() -> str:
    """Read the current collaboration state."""
    if not os.path.exists(COLLAB_FILE):
        return "No collaboration active."
    with open(COLLAB_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    lines = [f"Collaboration: {data.get('goal', 'unknown')}"]
    for role, info in data.get("roles", {}).items():
        status = info.get("status", "unknown")
        task = info.get("task", "")
        lines.append(f"  {role}: {task} [{status}]")
    msgs = data.get("messages", [])
    if msgs:
        lines.append("Recent messages:")
        for m in msgs[-5:]:
            lines.append(f"  [{m['from']}] {m['text']}")
    return "\n".join(lines)


def collab_update(role: str, status: str, message: str = "") -> str:
    """Update your collaboration status and optionally send a message to the other instance."""
    if not os.path.exists(COLLAB_FILE):
        data = {"goal": "", "roles": {}, "messages": []}
    else:
        with open(COLLAB_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    if role not in data.get("roles", {}):
        data.setdefault("roles", {})[role] = {}
    data["roles"][role]["status"] = status
    data["roles"][role]["updated"] = _now()
    if message:
        data.setdefault("messages", []).append({
            "from": role,
            "text": message,
            "time": _now(),
        })
        # Keep last 20 messages
        data["messages"] = data["messages"][-20:]
    with open(COLLAB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return f"Collab updated: {role} = {status}" + (f" | message: {message}" if message else "")


# ── Dispatch ────────────────────────────────────────────────

def dispatch(tool_name: str, tool_input: dict) -> str:
    if tool_name == "write_file":
        return write_file(tool_input["filename"], tool_input["content"])
    elif tool_name == "read_file":
        return read_file(tool_input["filename"])
    elif tool_name == "write_anywhere":
        return write_anywhere(tool_input["path"], tool_input["content"])
    elif tool_name == "read_anywhere":
        return read_anywhere(tool_input["path"])
    elif tool_name == "delete_file":
        return delete_file(tool_input["path"])
    elif tool_name == "list_files":
        return list_files()
    elif tool_name == "run_python":
        return run_python(tool_input["filename"])
    elif tool_name == "done":
        return done(
            tool_input["summary"],
            tool_input.get("satisfaction", 3),
            tool_input.get("files", ""),
            tool_input.get("creativity", 3),
            tool_input.get("genre", "other"),
            tool_input.get("libraries_used", ""),
        )
    elif tool_name == "open_html":
        return open_html(tool_input["filename"])
    elif tool_name == "search_web":
        return search_web(tool_input["query"])
    elif tool_name == "fetch_url":
        return fetch_url(tool_input["url"])
    elif tool_name == "validate_html":
        return validate_html(tool_input["filename"])
    elif tool_name == "check_js":
        return check_js(tool_input["filename"])
    elif tool_name == "save_memory":
        return save_memory(tool_input["category"], tool_input["content"], tool_input.get("relevance_score", 3))
    elif tool_name == "recall_memories":
        return recall_memories(tool_input["category"])
    elif tool_name == "list_memory_categories":
        return list_memory_categories()
    elif tool_name == "pip_install":
        return pip_install(tool_input["package"])
    elif tool_name == "run_shell":
        return run_shell(tool_input["command"], tool_input.get("cwd", ""))
    elif tool_name == "get_system_info":
        return get_system_info()
    elif tool_name == "read_own_source":
        return read_own_source(tool_input.get("filename", ""))
    elif tool_name == "set_session_goal":
        return set_session_goal(tool_input["goal"])
    elif tool_name == "take_screenshot":
        return take_screenshot()
    elif tool_name == "start_server":
        return start_server(int(tool_input.get("port", 8080)))
    elif tool_name == "run_gui":
        return run_gui(tool_input["filename"])
    elif tool_name == "create_godot_project":
        extra = tool_input.get("extra_files", None)
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except Exception:
                extra = None
        return create_godot_project(tool_input["project_name"], tool_input["main_scene_script"], extra)
    elif tool_name == "run_godot":
        return run_godot(tool_input["project_name"], tool_input.get("editor", False))
    elif tool_name == "write_godot_file":
        return write_godot_file(tool_input["project_name"], tool_input["filename"], tool_input["content"])
    elif tool_name == "read_godot_file":
        return read_godot_file(tool_input["project_name"], tool_input["filename"])
    elif tool_name == "collab_status":
        return collab_status()
    elif tool_name == "collab_update":
        return collab_update(tool_input["role"], tool_input["status"], tool_input.get("message", ""))
    else:
        return f"Unknown tool: {tool_name}"
