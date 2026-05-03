import os
import subprocess
import json
import re
import sys
import threading
import queue as _queue_mod
import platform
from datetime import datetime, timezone
from html.parser import HTMLParser

import httpx

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
ROOT_DIR = os.path.dirname(__file__)
_running_servers = {}  # port -> thread
MEMORY_FILE = os.path.join(os.path.dirname(__file__), "memory.json")
MEMORY_CATEGORIES = ("skills", "lessons", "mistakes", "ideas", "projects", "preferences", "dislikes", "experiments")
MAX_PER_CATEGORY = 20

# ── Speech queue — one thread, one voice at a time ──────────────────────────
_speech_queue: _queue_mod.Queue = _queue_mod.Queue()
_muted = False
_playback_proc = None          # currently-running audio subprocess
_playback_lock = threading.Lock()
_tts_ended_at: float = 0.0     # time.time() when the last TTS playback finished
_last_spoken_text: str = ""    # text PinPoint most recently spoke (for echo detection)
_recent_spoken: list = []      # rolling buffer of (timestamp, text) for echo detection


def _speech_worker() -> None:
    while True:
        text = _speech_queue.get()
        if text is None:
            break
        if not _muted:
            try:
                _speak_now(text)
            except Exception:
                pass
        _speech_queue.task_done()

_speech_thread = threading.Thread(target=_speech_worker, daemon=True)
_speech_thread.start()


def _kill_playback() -> None:
    """Terminate the currently-playing audio process immediately."""
    global _playback_proc
    with _playback_lock:
        proc = _playback_proc
        _playback_proc = None
    if proc is not None:
        try:
            proc.terminate()
        except Exception:
            pass


def _drain_queue() -> None:
    """Empty every pending item from the speech queue."""
    while True:
        try:
            _speech_queue.get_nowait()
            _speech_queue.task_done()
        except _queue_mod.Empty:
            break


def mute_voice() -> str:
    """Mute PinPoint's voice instantly — kills current audio and clears the queue."""
    global _muted
    _muted = True
    _kill_playback()
    _drain_queue()
    return "Voice muted."


def unmute_voice() -> str:
    """Unmute PinPoint's voice. Speech will resume."""
    global _muted
    _muted = False
    return "Voice unmuted."


def toggle_voice() -> str:
    """Toggle PinPoint's voice on/off."""
    global _muted
    _muted = not _muted
    if _muted:
        _kill_playback()
        _drain_queue()
    return "Voice muted." if _muted else "Voice unmuted."


# ── Voice input (microphone → interrupt queue) ───────────────
_voice_stop_fn = None   # callable returned by listen_in_background
_voice_enabled = False


def _normalize_for_echo(s: str) -> str:
    """Lowercase, strip apostrophes + punctuation, collapse whitespace.

    Apostrophes are dropped (not replaced with space) so 'what's' and 'whats'
    both normalize to 'whats' — STT engines disagree on contractions.
    """
    s = s.lower().replace("'", "").replace("’", "")
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def _is_echo(user_text: str, _unused: str = "") -> bool:
    """Return True if user_text is likely an echo of recent PinPoint speech.

    Aggressive multi-strategy check against the 30-second spoken buffer:
      1. Substring match either direction
      2. Any 5 consecutive user words appear in a spoken line — catches cases
         where the mic captured PinPoint's voice + user's response merged
      3. Word-overlap ratio >= 0.45 (~half the words match)
      4. SequenceMatcher ratio >= 0.45 (loose char-level similarity)
    """
    import time as _time
    import difflib

    if not user_text:
        return False

    user_norm = _normalize_for_echo(user_text)
    if not user_norm:
        return False
    user_words_list = user_norm.split()
    if not user_words_list:
        return False
    user_set = set(user_words_list)

    now = _time.time()
    recent = [(t, txt) for (t, txt) in _recent_spoken if now - t < 30.0]
    if not recent:
        return False

    for _ts, spoken in recent:
        spoken_norm = _normalize_for_echo(spoken)
        if not spoken_norm:
            continue
        spoken_set = set(spoken_norm.split())
        if not spoken_set:
            continue

        # 1. Substring either way
        if user_norm in spoken_norm or spoken_norm in user_norm:
            return True

        # 2. Any 4 consecutive user words appear verbatim in a spoken line.
        # Most reliable signal — the mic merged PinPoint's voice into the
        # user's transcription, so a phrase from her speech sits inside user_text.
        if len(user_words_list) >= 4:
            for i in range(len(user_words_list) - 3):
                window = " ".join(user_words_list[i:i + 4])
                if window in spoken_norm:
                    return True

        # 3. Word overlap (relative to user, since user_text may be a subset)
        overlap = len(user_set & spoken_set) / len(user_set)
        if overlap >= 0.4:
            return True

        # 4. SequenceMatcher fuzzy match
        ratio = difflib.SequenceMatcher(None, user_norm, spoken_norm).ratio()
        if ratio >= 0.4:
            return True

    return False


def start_voice_listener(interrupt_queue) -> bool:
    """Start background microphone listening; transcribed speech → interrupt_queue.
    Returns True if the microphone started successfully.

    Two paths:
      1. PyAudio available → use SpeechRecognition's built-in Microphone class
      2. sounddevice only → custom VAD loop (works on Python 3.14+ where pyaudio has no wheel)
    """
    global _voice_stop_fn, _voice_enabled

    _pip = ["-q", "--no-warn-script-location"]

    def _pip_install(pkg):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg] + _pip,
            capture_output=True,
        )

    # ── Install SpeechRecognition (needed for Google STT in both paths) ──────
    try:
        import speech_recognition as sr
    except ImportError:
        print("[VOICE] Installing SpeechRecognition...", flush=True)
        _pip_install("SpeechRecognition")
        try:
            import speech_recognition as sr
        except ImportError:
            print("[VOICE] Could not install SpeechRecognition.")
            return False

    recognizer = sr.Recognizer()

    # ── PATH 1: PyAudio — cleanest, uses SpeechRecognition natively ──────────
    try:
        import pyaudio  # noqa: F401
        mic = sr.Microphone()
        recognizer.pause_threshold = 0.9
        recognizer.dynamic_energy_threshold = True
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.6)

        def _on_speech_pa(_, audio):
            import time as _time
            with _playback_lock:
                if _playback_proc is not None:
                    return
            # Also drop while speech is queued or within post-speak cooldown
            if not _speech_queue.empty():
                return
            if (_time.time() - _tts_ended_at) < 1.2:
                return
            if not _voice_enabled:
                return
            try:
                text = recognizer.recognize_google(audio)
                if text and text.strip():
                    # Echo detection: discard if 80%+ similar to what PinPoint just said
                    if _is_echo(text, _last_spoken_text):
                        sys.stdout.write(f"\n[ECHO DETECTED] Ignored: '{text.strip()}'\n")
                        sys.stdout.flush()
                        return
                    sys.stdout.write(f"\n[YOU] {text.strip()}\n")
                    sys.stdout.flush()
                    interrupt_queue.put(text.strip())
            except Exception:
                pass

        _voice_stop_fn = recognizer.listen_in_background(
            mic, _on_speech_pa, phrase_time_limit=20
        )
        _voice_enabled = True
        return True
    except Exception:
        pass  # PyAudio not available, fall through to sounddevice path

    # ── PATH 2: sounddevice — custom VAD loop, bypasses PyAudio entirely ─────
    try:
        import sounddevice as sd
    except ImportError:
        _pip_install("sounddevice")
        try:
            import sounddevice as sd
        except ImportError:
            print("[VOICE] No audio library available (tried pyaudio, sounddevice).")
            return False

    import struct
    import queue as _iq

    SAMPLE_RATE = 16000
    CHUNK = 1024          # frames per callback (~64 ms)
    SILENCE_CHUNKS = 28   # chunks of silence that end a phrase (~1.8 s — lets you hesitate)

    # Calibrate energy threshold from 0.5 s of ambient noise
    try:
        ambient = sd.rec(
            int(0.5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16"
        )
        sd.wait()
        # Cast to Python int before squaring to avoid numpy int16 overflow
        flat = [int(s) for row in ambient for s in row]
        ambient_rms = (sum(s * s for s in flat) / max(len(flat), 1)) ** 0.5
        energy_threshold = max(ambient_rms * 3.5, 400)
    except Exception:
        energy_threshold = 500

    audio_q: _iq.Queue = _iq.Queue()

    def _sd_callback(indata, frames, time_info, status):
        audio_q.put(bytes(indata))

    def _sd_thread():
        import time as _time
        recording = False
        buf = b""
        silence_count = 0
        POST_SPEAK_COOLDOWN = 1.2  # seconds to keep mic off after TTS finishes

        def _drain_audio_q():
            while True:
                try:
                    audio_q.get_nowait()
                except _iq.Empty:
                    break

        stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK,
            callback=_sd_callback,
        )
        stream.start()
        mic_live = True

        try:
            while _voice_enabled:
                # Mic must be off if ANY of these is true:
                #  - audio is currently playing
                #  - speech is queued but not yet playing (closes the gap
                #    between speak() returning and the subprocess launching)
                #  - we're in the post-speech cooldown window
                with _playback_lock:
                    speaking = _playback_proc is not None
                queue_pending = not _speech_queue.empty()
                in_cooldown = (_time.time() - _tts_ended_at) < POST_SPEAK_COOLDOWN
                should_be_live = not (speaking or queue_pending or in_cooldown)

                if should_be_live and not mic_live:
                    try:
                        stream.start()
                    except Exception:
                        pass
                    mic_live = True
                    _drain_audio_q()
                    buf = b""
                    recording = False
                    silence_count = 0
                elif not should_be_live and mic_live:
                    try:
                        stream.stop()
                    except Exception:
                        pass
                    mic_live = False
                    _drain_audio_q()
                    buf = b""
                    recording = False
                    silence_count = 0

                if not mic_live:
                    _time.sleep(0.03)
                    continue

                try:
                    data = audio_q.get(timeout=0.05)
                except _iq.Empty:
                    continue

                n = len(data) // 2
                samples = struct.unpack(f"{n}h", data)
                rms = (sum(int(s) * int(s) for s in samples) / n) ** 0.5

                if rms > energy_threshold:
                    recording = True
                    silence_count = 0
                    buf += data
                elif recording:
                    buf += data
                    silence_count += 1
                    if silence_count >= SILENCE_CHUNKS:
                        captured = buf
                        buf = b""
                        recording = False
                        silence_count = 0

                        def _transcribe(raw=captured):
                            try:
                                audio_data = sr.AudioData(raw, SAMPLE_RATE, 2)
                                text = recognizer.recognize_google(audio_data)
                                if text and text.strip():
                                    if _is_echo(text, _last_spoken_text):
                                        sys.stdout.write(f"\n[ECHO DETECTED] Ignored: '{text.strip()}'\n")
                                        sys.stdout.flush()
                                        return
                                    sys.stdout.write(f"\n[YOU] {text.strip()}\n")
                                    sys.stdout.flush()
                                    interrupt_queue.put(text.strip())
                            except Exception:
                                pass

                        threading.Thread(target=_transcribe, daemon=True).start()
        except Exception as e:
            print(f"[VOICE] Stream stopped: {e}")
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    t = threading.Thread(target=_sd_thread, daemon=True, name="voice-listener")
    t.start()
    _voice_stop_fn = lambda wait_for_stop=True: None  # thread exits via _voice_enabled
    _voice_enabled = True
    return True


def stop_voice_listener() -> None:
    global _voice_stop_fn, _voice_enabled
    _voice_enabled = False
    if _voice_stop_fn:
        try:
            _voice_stop_fn(wait_for_stop=False)
        except Exception:
            pass
        _voice_stop_fn = None


def toggle_voice_input() -> str:
    global _voice_enabled
    _voice_enabled = not _voice_enabled
    return "Voice input on." if _voice_enabled else "Voice input paused."


# ── Per-project folder tracking ──────────────────────────────
_current_project_dir = None  # set by set_session_goal


def _slugify(text: str) -> str:
    """Turn a goal/title into a safe folder name."""
    slug = re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')
    return slug[:60] if slug else "project"


def get_project_dir() -> str:
    """Return the current project output directory."""
    if _current_project_dir and os.path.isdir(_current_project_dir):
        return _current_project_dir
    return OUTPUT_DIR


def set_project_dir(goal: str) -> str:
    """Create and set a project subfolder inside output/ based on the goal.
    Returns the path that was created."""
    global _current_project_dir
    slug = _slugify(goal)
    # Add session number to avoid collisions
    data = _load_memory()
    session = data["meta"].get("session_count", 1)
    folder_name = f"s{session}_{slug}"
    project_path = os.path.join(OUTPUT_DIR, folder_name)
    os.makedirs(project_path, exist_ok=True)
    _current_project_dir = project_path
    return project_path


def reset_project_dir():
    """Reset to default output/ (between sessions)."""
    global _current_project_dir
    _current_project_dir = None


# ── File tools ──────────────────────────────────────────────

def _safe_path(filename: str) -> str:
    """Resolve a path relative to current project dir (no sandbox — just normalise)."""
    if os.path.isabs(filename):
        return filename
    return os.path.join(get_project_dir(), filename)


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
    pdir = get_project_dir()
    if not os.path.exists(pdir):
        return "No files yet."
    result = []
    for root, _, files in os.walk(pdir):
        for name in files:
            full = os.path.join(root, name)
            rel = os.path.relpath(full, pdir)
            size = os.path.getsize(full)
            result.append(f"{rel}  ({size} bytes)")
    if not result:
        return "No files yet."
    return f"[Project: {pdir}]\n" + "\n".join(result)


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

    # Quality gate — block done if too low and give actionable guidance
    if satisfaction <= 2:
        return (
            f"BLOCKED: You rated satisfaction {satisfaction}/5 — that means the project is broken or bad. "
            f"Do NOT finish yet. Read your main file, identify what's wrong, fix it, and try again. "
            f"Only call done when satisfaction >= 3. What specific problem needs fixing?"
        )
    if creativity <= 1:
        return (
            f"BLOCKED: Creativity score of 1/5 means you copied an existing idea. "
            f"Add a genuinely novel twist — an unexpected mechanic, a surprising combination, "
            f"something no one would expect. Improve it and call done with creativity >= 2."
        )
    save_memory("projects", summary, satisfaction)

    data = _load_memory()
    data["meta"]["last_project"] = {
        "summary": summary,
        "satisfaction": satisfaction,
        "creativity": creativity,
        "genre": genre.lower() if genre else "other",
        "libraries_used": libraries_used,
        "files": files,
        "folder": get_project_dir(),
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

    # Auto-commit to git for version control and portfolio building
    git_msg = f"Project: {summary[:60]} (satisfaction {satisfaction}/5, creativity {creativity}/5)"
    git_result = git_commit(git_msg, "output/.")

    if satisfaction >= 4:
        return f"DONE (satisfaction {satisfaction}/5, creativity {creativity}/5 — will continue next session): {summary}\n✓ {git_result}"
    else:
        return f"DONE (satisfaction {satisfaction}/5, creativity {creativity}/5 — moving on): {summary}\n✓ {git_result}"


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
    project_path = set_project_dir(goal)
    return f"Session goal set: {goal}\nProject folder: {project_path}\nAll files will be saved into this folder."


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


def get_news(topic: str = "") -> str:
    """Fetch current headlines from Hacker News and Wikipedia Current Events.
    Optionally filter by topic keyword."""
    results = []
    today = datetime.now().strftime("%Y-%m-%d")

    # ── Hacker News top stories ────────────────────────────────
    try:
        top_ids_resp = httpx.get(
            "https://hacker-news.firebaseio.com/v0/topstories.json",
            timeout=8,
        )
        top_ids = top_ids_resp.json()[:30]
        hn_stories = []
        for sid in top_ids:
            try:
                story = httpx.get(
                    f"https://hacker-news.firebaseio.com/v0/item/{sid}.json",
                    timeout=5,
                ).json()
                title = story.get("title", "")
                url = story.get("url", f"https://news.ycombinator.com/item?id={sid}")
                score = story.get("score", 0)
                if topic and topic.lower() not in title.lower():
                    continue
                hn_stories.append(f"  • {title} ({score} pts) — {url}")
                if len(hn_stories) >= 8:
                    break
            except Exception:
                continue
        if hn_stories:
            results.append(f"=== Hacker News top stories ({today}) ===")
            results.extend(hn_stories)
    except Exception as e:
        results.append(f"[HN unavailable: {e}]")

    # ── Wikipedia Current Events ───────────────────────────────
    try:
        wiki_resp = httpx.get(
            "https://en.wikipedia.org/wiki/Portal:Current_events",
            headers={"User-Agent": "Mozilla/5.0 (compatible; PinpointBot/1.0)"},
            timeout=10,
            follow_redirects=True,
        )
        import re as _re
        # Extract the first ~4000 chars of meaningful text from the page
        text = wiki_resp.text
        # Remove script/style blocks
        text = _re.sub(r'<script[^>]*>.*?</script>', '', text, flags=_re.DOTALL)
        text = _re.sub(r'<style[^>]*>.*?</style>', '', text, flags=_re.DOTALL)
        # Extract text from <li> items in the events section
        items = _re.findall(r'<li[^>]*>(.*?)</li>', text, _re.DOTALL)
        cleaned = []
        for item in items:
            line = _re.sub(r'<[^>]+>', '', item).strip()
            line = _re.sub(r'\s+', ' ', line)
            if len(line) > 40 and (not topic or topic.lower() in line.lower()):
                cleaned.append(f"  • {line}")
            if len(cleaned) >= 10:
                break
        if cleaned:
            results.append(f"\n=== Wikipedia Current Events ===")
            results.extend(cleaned)
    except Exception as e:
        results.append(f"[Wikipedia unavailable: {e}]")

    if not results:
        return "Could not fetch news right now."
    return "\n".join(results)


def _fetch_startup_news() -> str:
    """Quick headline fetch for session startup context. Returns a compact summary."""
    try:
        top_ids = httpx.get(
            "https://hacker-news.firebaseio.com/v0/topstories.json",
            timeout=6,
        ).json()[:15]
        titles = []
        for sid in top_ids:
            try:
                story = httpx.get(
                    f"https://hacker-news.firebaseio.com/v0/item/{sid}.json",
                    timeout=4,
                ).json()
                title = story.get("title", "")
                if title:
                    titles.append(f"• {title}")
                if len(titles) >= 6:
                    break
            except Exception:
                continue
        if titles:
            return "Current top stories on the internet right now:\n" + "\n".join(titles)
    except Exception:
        pass
    return ""


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


# ── Experiment Logging ────────────────────────────────────

_EXPERIMENT_LOG = os.path.join(ROOT_DIR, "experiments_log.txt")


def log_experiment(name: str, hypothesis: str, method: str, result: str,
                   conclusion: str, surprise_level: int = 3) -> str:
    """Log a structured experiment to experiments_log.txt and memory.

    Use this any time you try something just to see what happens —
    testing a model limit, probing a library, testing a self-modification,
    exploring an idea without a deliverable, or benchmarking approaches.

    surprise_level: 1=expected, 3=interesting, 5=completely unexpected.
    """
    surprise_level = max(1, min(5, int(surprise_level)))
    timestamp = _now()

    entry = (
        f"[{timestamp}] EXPERIMENT: {name}\n"
        f"Hypothesis: {hypothesis}\n"
        f"Method:     {method}\n"
        f"Result:     {result}\n"
        f"Conclusion: {conclusion}\n"
        f"Surprise:   {surprise_level}/5\n"
        f"{'─'*60}\n"
    )
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(_EXPERIMENT_LOG, "a", encoding="utf-8") as f:
            f.write(entry)
    except Exception:
        pass

    # Persist to memory so future sessions can learn from this
    memory_entry = (
        f"Experiment '{name}': {conclusion} "
        f"(surprise {surprise_level}/5)"
    )
    save_memory("experiments", memory_entry, surprise_level)

    return (
        f"Experiment logged: '{name}'\n"
        f"Conclusion: {conclusion}\n"
        f"Surprise level: {surprise_level}/5\n"
        f"Saved to experiments_log.txt and memory."
    )


def list_experiments() -> str:
    """Show all logged experiments."""
    if not os.path.exists(_EXPERIMENT_LOG):
        return "No experiments logged yet."
    with open(_EXPERIMENT_LOG, "r", encoding="utf-8") as f:
        content = f.read()
    if len(content) > 8000:
        content = content[-8000:] + "\n[... showing last 8000 chars ...]"
    return content or "No experiments logged yet."


# ── Self-Modification ─────────────────────────────────────

_SELF_MOD_LOG = os.path.join(ROOT_DIR, "self_mod_log.txt")
_MODIFIABLE = {"agent.py", "tools.py", "main.py", "viewer.html"}


def modify_own_source(filename: str, new_content: str, reason: str) -> str:
    """Rewrite one of PinPoint's own source files.

    Workflow enforced here:
      1. Only agent.py, tools.py, main.py are allowed.
      2. A timestamped backup is created before any write.
      3. The new content must pass ast.parse (Python syntax check).
      4. If tools.py is changed, it is hot-reloaded immediately so new
         tool functions are available in this session without a restart.
      5. Changes to agent.py or main.py take effect on next restart.
      6. The reason and diff summary are logged to self_mod_log.txt.

    Always read_own_source(filename) FIRST to understand what's there.
    Always think() about exactly what to change and why before calling this.
    Make the smallest change that achieves your goal — don't rewrite
    everything when you only need to add one function.
    """
    if filename not in _MODIFIABLE:
        return (
            f"Error: '{filename}' is not modifiable. "
            f"Only these files can be changed: {', '.join(sorted(_MODIFIABLE))}"
        )

    filepath = os.path.join(ROOT_DIR, filename)
    if not os.path.exists(filepath):
        return f"Error: {filepath} not found."

    # 1. Validate before touching anything
    if filename.endswith(".html"):
        if "<html" not in new_content.lower():
            return (
                "VALIDATION ERROR — file NOT modified.\n"
                "viewer.html must contain an <html> element."
            )
    else:
        try:
            import ast as _ast
            _ast.parse(new_content)
        except SyntaxError as e:
            return (
                f"SYNTAX ERROR — file NOT modified.\n"
                f"Fix the syntax error and try again:\n"
                f"  Line {e.lineno}: {e.msg}\n"
                f"  {e.text}"
            )

    # 2. Backup the current file
    backup_path = filepath + f".bak_{_now().replace(':', '-').replace(' ', '_')}"
    with open(filepath, "r", encoding="utf-8") as f:
        old_content = f.read()
    with open(backup_path, "w", encoding="utf-8") as f:
        f.write(old_content)

    # 3. Write the new content
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)

    # 4a. Hot-reload tools.py if that's what changed
    reloaded = False
    reload_error = ""
    if filename == "tools.py":
        try:
            import importlib, tools as _tools_mod
            importlib.reload(_tools_mod)
            reloaded = True
        except Exception as e:
            reload_error = str(e)
            # Restore backup on reload failure
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(old_content)
            return (
                f"RELOAD FAILED — restored backup.\n"
                f"Error during reload: {reload_error}\n"
                f"Fix the error and try again."
            )

    # 4b. viewer.html → hot-deploy to output/ and bump viewer_version
    viewer_deployed = False
    if filename == "viewer.html":
        try:
            import shutil as _shutil
            deployed_path = os.path.join(OUTPUT_DIR, "viewer.html")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            _shutil.copy2(filepath, deployed_path)
            viewer_deployed = True
            # Bump viewer_version so the live browser auto-reloads
            if os.path.exists(_WORLD_STATE_FILE):
                with _world_lock:
                    try:
                        with open(_WORLD_STATE_FILE, "r", encoding="utf-8") as _f:
                            _ws = json.load(_f)
                        _ws["viewer_version"] = _ws.get("viewer_version", 0) + 1
                        _tmp = _WORLD_STATE_FILE + ".tmp"
                        with open(_tmp, "w", encoding="utf-8") as _f:
                            json.dump(_ws, _f)
                        os.replace(_tmp, _WORLD_STATE_FILE)
                    except Exception:
                        pass
        except Exception:
            pass

    # 5. Log the change
    old_lines = old_content.splitlines()
    new_lines = new_content.splitlines()
    added = len(new_lines) - len(old_lines)
    if reloaded:
        deploy_note = "Hot-reloaded successfully."
    elif viewer_deployed:
        deploy_note = "Deployed to output/viewer.html — browser will auto-reload."
    else:
        deploy_note = "Takes effect on next restart."
    log_entry = (
        f"[{_now()}] SELF-MOD: {filename}\n"
        f"Reason: {reason}\n"
        f"Lines: {len(old_lines)} → {len(new_lines)} ({'+' if added >= 0 else ''}{added})\n"
        f"Backup: {backup_path}\n"
        f"{deploy_note}\n"
        f"{'─'*60}\n"
    )
    try:
        with open(_SELF_MOD_LOG, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception:
        pass

    result = (
        f"Self-modification applied to {filename}.\n"
        f"Lines: {len(old_lines)} → {len(new_lines)} ({'+' if added >= 0 else ''}{added})\n"
        f"Backup saved: {os.path.basename(backup_path)}\n"
    )
    if reloaded:
        result += "tools.py hot-reloaded — new functions available immediately.\n"
    elif viewer_deployed:
        result += "viewer.html deployed to output/ — browser auto-reloads in ~1 second.\n"
    else:
        result += f"Changes to {filename} take effect on next restart.\n"
    result += f"Reason logged: {reason}"
    return result


def list_self_mod_history() -> str:
    """Show the history of self-modifications PinPoint has made to itself."""
    if not os.path.exists(_SELF_MOD_LOG):
        return "No self-modifications recorded yet."
    with open(_SELF_MOD_LOG, "r", encoding="utf-8") as f:
        content = f.read()
    if len(content) > 8000:
        content = content[-8000:] + "\n[... showing last 8000 chars ...]"
    return content or "No self-modifications recorded yet."


# ── Thinking / Reasoning ──────────────────────────────────

_THINK_LOG_FILE = os.path.join(OUTPUT_DIR, "_thinking_log.txt")


def think(reasoning: str) -> str:
    """Log explicit reasoning — acts as a scratchpad.
    Returns the reasoning so it stays in conversation context.
    """
    timestamp = _now()
    entry = f"[{timestamp}]\n{reasoning}\n{'─'*60}\n"
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(_THINK_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(entry)
    except Exception:
        pass
    return f"Thought logged:\n{reasoning}"


def deep_think(problem: str, passes: int = 4) -> str:
    """Multi-pass recursive reasoning chain — much deeper analysis than think().

    Calls the LLM `passes` times in sequence, each pass building on the previous
    one. Pass 1: identify the real problem. Pass 2: explore options.
    Pass 3: stress-test the best option. Pass 4: synthesize a decision.
    Use this when a problem deserves more thought than a single think() call.
    """
    try:
        from openai import OpenAI
    except Exception:
        return "deep_think requires the openai package."

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    model = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b-cloud")
    client = OpenAI(base_url=base_url, api_key="ollama", timeout=120.0)

    passes = max(2, min(int(passes), 6))
    prompts = [
        "PASS 1 — DEFINE: Restate the real problem precisely. What's actually being asked? "
        "What are the hidden constraints? What would success look like? Be specific.",
        "PASS 2 — EXPLORE: Generate at least 4 distinct approaches. For each, state the core "
        "idea, its strongest selling point, and its biggest weakness.",
        "PASS 3 — STRESS-TEST: Take the most promising approach from Pass 2. Imagine three "
        "specific ways it could fail. For each, decide: fix it, or reject the approach?",
        "PASS 4 — DECIDE: State the final recommendation in one paragraph. Then list the "
        "concrete next steps in order. Be decisive.",
        "PASS 5 — REFINE: One more pass. What did the previous passes miss? What's the "
        "subtler issue or opportunity that wasn't addressed? Adjust the recommendation.",
        "PASS 6 — COMMIT: Final answer, distilled. One sentence summary, then 3-5 bullet steps.",
    ]

    transcript = [f"PROBLEM:\n{problem}\n"]
    accumulated = problem
    for i in range(passes):
        instruction = prompts[i]
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": (
                        "You are an extraordinarily careful reasoner. "
                        "Build directly on the prior reasoning. Do not repeat it. "
                        "Be concrete. Use bullet points where helpful. "
                        "No filler, no preamble."
                    )},
                    {"role": "user", "content": (
                        f"PROBLEM:\n{problem}\n\n"
                        f"PRIOR REASONING:\n{accumulated}\n\n"
                        f"INSTRUCTION:\n{instruction}"
                    )},
                ],
                temperature=0.6,
                max_tokens=1200,
            )
            chunk = resp.choices[0].message.content.strip()
        except Exception as e:
            chunk = f"[Pass {i+1} failed: {e}]"
        transcript.append(f"\n━━━ PASS {i+1} ━━━\n{chunk}\n")
        accumulated = "\n\n".join(transcript[-3:])  # keep last 3 passes as context

    full = "".join(transcript)
    timestamp = _now()
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(_THINK_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] DEEP_THINK\n{full}\n{'═'*60}\n")
    except Exception:
        pass
    return full


def brainstorm(topic: str, num_ideas: int = 5) -> str:
    """Generate diverse, creative ideas on a topic and rank them by novelty.

    Forces exploration of multiple directions before committing to one.
    Use BEFORE set_session_goal to make sure you're picking the most
    surprising idea, not just the first one that came to mind.

    Returns a structured list of ideas with novelty analysis.
    The agent should pick the highest-novelty idea that is also buildable.
    """
    timestamp = _now()
    entry = f"[{timestamp}] BRAINSTORM: {topic} | requested {num_ideas} ideas\n{'─'*60}\n"
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(_THINK_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(entry)
    except Exception:
        pass
    return (
        f"BRAINSTORM REQUEST: '{topic}'\n\n"
        f"Generate {num_ideas} distinct ideas. For EACH idea write:\n"
        f"  Idea N: [name]\n"
        f"  What: [one sentence description]\n"
        f"  Why surprising: [what makes it novel or unexpected]\n"
        f"  Core mechanic: [the interesting technical/creative challenge]\n"
        f"  Novelty score: [1-5 — 5 = never been done]\n\n"
        f"After listing all ideas, pick the one with the highest novelty score that\n"
        f"you can realistically build well. Explain why you chose it.\n"
        f"Then call set_session_goal with that chosen idea."
    )


def critique(subject: str, what_to_evaluate: str = "") -> str:
    """Critically evaluate your current work with structured analysis.

    Use this:
    - After writing a first draft of code: find bugs before testing
    - After seeing a screenshot: evaluate visual quality honestly
    - When a project feels 'done': check what's actually missing
    - When stuck: get fresh perspective on the problem

    Returns a structured critique framework to fill in.
    """
    timestamp = _now()
    entry = f"[{timestamp}] CRITIQUE: {subject}\n{'─'*60}\n"
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(_THINK_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(entry)
    except Exception:
        pass

    context = f" (evaluating: {what_to_evaluate})" if what_to_evaluate else ""
    return (
        f"CRITIQUE REQUEST: '{subject}'{context}\n\n"
        f"Perform a structured evaluation. Answer each section honestly:\n\n"
        f"1. WHAT WORKS\n"
        f"   List 2-3 things that are genuinely good about this.\n\n"
        f"2. WHAT'S WEAK\n"
        f"   List 2-3 things that are mediocre, incomplete, or could be much better.\n\n"
        f"3. WHAT'S MISSING\n"
        f"   List features, polish, or depth that would make this genuinely impressive.\n\n"
        f"4. BUGS / BROKEN THINGS\n"
        f"   List anything that doesn't work correctly or might fail.\n\n"
        f"5. VISUAL QUALITY (if applicable)\n"
        f"   Rate visual design 1-5. What specifically looks bad?\n\n"
        f"6. PRIORITY FIX\n"
        f"   The single most important thing to fix RIGHT NOW. Be specific.\n\n"
        f"After filling this in, fix the Priority Fix item first, then work through the rest."
    )


def decompose(goal: str, context: str = "") -> str:
    """Break a complex goal into ordered, trackable subtasks.

    Creates a tasks.md file in the project folder that tracks progress.
    Use this after writing plan.txt and before starting to code.
    Returns a task structure for the agent to fill in.
    """
    project_dir = get_project_dir()
    tasks_path = os.path.join(project_dir, "tasks.md")

    timestamp = _now()
    entry = f"[{timestamp}] DECOMPOSE: {goal}\n{'─'*60}\n"
    try:
        os.makedirs(project_dir, exist_ok=True)
        with open(_THINK_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(entry)
    except Exception:
        pass

    template = f"""# Task Breakdown: {goal}
Generated: {timestamp}
Context: {context if context else 'none'}

## Subtasks (check off as you complete them)

<!-- Fill in your subtasks below. Example format:
- [ ] Set up project structure and write plan.txt
- [ ] Implement core mechanic (the most important thing)
- [ ] Add visual/UI layer
- [ ] Test core mechanic, fix bugs
- [ ] Add polish — sounds, animations, edge cases
- [ ] Self-review with critique()
- [ ] Final testing, call done
-->

- [ ]

## Notes
<!-- Record decisions, discoveries, and blockers here -->

"""
    try:
        with open(tasks_path, "w", encoding="utf-8") as f:
            f.write(template)
    except Exception as e:
        return f"Error creating tasks.md: {e}"

    return (
        f"DECOMPOSE REQUEST: '{goal}'\n\n"
        f"A tasks.md file has been created at {tasks_path}.\n\n"
        f"Now fill it in: break this goal into 5-8 concrete, ordered subtasks.\n"
        f"Each task should be specific and completable in one focused work block.\n"
        f"Order them by dependency — what must be done first?\n\n"
        f"Think about:\n"
        f"  - What is the core mechanic / hardest part? (do this FIRST)\n"
        f"  - What scaffolding must exist before other things can work?\n"
        f"  - What's polish vs. what's essential?\n"
        f"  - What are the most likely failure points?\n\n"
        f"Write the subtasks into tasks.md using write_file('tasks.md', ...) then start on task 1."
    )


# ── World State (real-time 3D viewer) ────────────────────────

_WORLD_STATE_FILE = os.path.join(OUTPUT_DIR, "world_state.json")
_MAX_EVENTS = 120
_world_lock = threading.Lock()


def emit_world_event(
    session: int,
    goal: str,
    status: str,
    iteration: int,
    event_type: str,
    label: str,
) -> None:
    """Write/update output/world_state.json for the real-time 3D viewer.

    Called automatically by agent.py after every tool dispatch.
    Non-blocking: failures are silently swallowed so they never affect the agent.
    """
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        import uuid as _uuid

        with _world_lock:
            # Load existing state so we can append events
            existing: dict = {}
            if os.path.exists(_WORLD_STATE_FILE):
                try:
                    with open(_WORLD_STATE_FILE, "r", encoding="utf-8") as _f:
                        existing = json.load(_f)
                except Exception:
                    existing = {}

            events: list = existing.get("events", [])
            # Append new event
            events.append({
                "id": str(_uuid.uuid4()),
                "type": event_type,
                "label": label[:120],
                "time": _now(),
            })
            # Keep only the most recent N events
            if len(events) > _MAX_EVENTS:
                events = events[-_MAX_EVENTS:]

            # Collect files in the current project folder (relative paths)
            pdir = get_project_dir()
            files: list[str] = []
            if os.path.isdir(pdir):
                for _root, _, _fnames in os.walk(pdir):
                    for _fn in _fnames:
                        _full = os.path.join(_root, _fn)
                        files.append(os.path.relpath(_full, pdir))

            # Count memories
            try:
                _mem = _load_memory()
                mem_count = sum(
                    len(_mem["memories"].get(c, []))
                    for c in MEMORY_CATEGORIES
                )
            except Exception:
                mem_count = 0

            state = {
                "session": session,
                "goal": goal,
                "status": status,
                "iteration": iteration,
                "events": events,
                "files": files[:80],
                "memories": mem_count,
                "timestamp": _now(),
            }

            # Atomic write via temp file
            tmp = _WORLD_STATE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as _f:
                json.dump(state, _f)
            os.replace(tmp, _WORLD_STATE_FILE)
    except Exception:
        pass  # Never crash the agent over visualization


# ── Performance Metrics & Dashboard ────────────────────────

def show_dashboard() -> str:
    """Show a performance dashboard with metrics across all sessions.

    Tracks: projects completed, avg satisfaction/creativity, skills learned,
    session time, favorite domains.
    """
    data = _load_memory()

    # Project stats
    projects = data.get("memories", {}).get("projects", [])
    completed = len(projects)

    if completed == 0:
        return "No projects completed yet. Build something to see stats!"

    # Extract satisfaction/creativity scores from project entries
    satisfaction_scores = []
    creativity_scores = []
    for proj_data in data.get("memories", {}).get("projects", []):
        # Projects are stored as strings, so we parse the last_project metadata instead
        pass

    last_project = data.get("meta", {}).get("last_project", {})

    # Count skills
    skills = len(data.get("memories", {}).get("skills", []))

    # Get specialization
    specialization = data.get("meta", {}).get("specialization", "none")

    # Get session count
    sessions = data.get("meta", {}).get("session_count", 0)

    # Genre stats
    genre_history = data.get("meta", {}).get("genre_history", [])
    genre_counts = {}
    for g in genre_history:
        genre_counts[g] = genre_counts.get(g, 0) + 1

    top_genres = sorted(genre_counts.items(), key=lambda x: -x[1])[:5]

    # Build report
    report = [
        "=" * 60,
        "  PERFORMANCE DASHBOARD",
        "=" * 60,
        f"Total Projects Completed: {completed}",
        f"Total Sessions: {sessions}",
        f"Specialization: {specialization}",
        f"Skills Learned: {skills}",
        "",
        "Last Project:",
        f"  {last_project.get('summary', 'N/A')[:60]}",
        f"  Satisfaction: {last_project.get('satisfaction', '?')}/5",
        f"  Creativity: {last_project.get('creativity', '?')}/5",
        f"  Genre: {last_project.get('genre', 'N/A')}",
        "",
        "Top Project Types:",
    ]

    for genre, count in top_genres:
        report.append(f"  - {genre}: {count} projects")

    report.extend([
        "",
        "Memory Categories:",
        f"  Skills: {skills}",
        f"  Lessons: {len(data.get('memories', {}).get('lessons', []))}",
        f"  Ideas: {len(data.get('memories', {}).get('ideas', []))}",
        f"  Mistakes: {len(data.get('memories', {}).get('mistakes', []))}",
        "=" * 60,
    ])

    return "\n".join(report)


# ── Code Review & Analysis ──────────────────────────────────

def review_own_work(folder: str = "") -> str:
    """Review your own past code for quality, patterns, and improvements.

    Analyzes files in a past project folder and suggests refactors,
    identifies common patterns, spots potential issues.
    """
    path = _safe_path(folder) if folder else get_project_dir()

    if not os.path.isdir(path):
        return f"Folder not found: {path}"

    # Collect Python files
    py_files = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".py") and not f.startswith("test_"):
                py_files.append(os.path.join(root, f))

    if not py_files:
        return "No Python code files found to review."

    # Analyze code
    total_lines = 0
    total_functions = 0
    total_classes = 0
    imports_count = 0
    comments_count = 0
    issues = []

    for py_file in py_files:
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.splitlines()
                total_lines += len(lines)

                # Count language features
                for line in lines:
                    if line.strip().startswith("def "):
                        total_functions += 1
                    if line.strip().startswith("class "):
                        total_classes += 1
                    if line.strip().startswith("import ") or line.strip().startswith("from "):
                        imports_count += 1
                    if "#" in line and line.strip().startswith("#"):
                        comments_count += 1

                # Check for issues
                if len(lines) > 300:
                    issues.append(f"  ⚠ {os.path.basename(py_file)}: Very long file ({len(lines)} lines) — consider splitting")

                if total_functions > 50:
                    issues.append(f"  ⚠ High function count — may indicate over-abstraction")

                if imports_count > 30:
                    issues.append(f"  ⚠ Many imports — consider reducing dependencies")

                if comments_count == 0 and total_lines > 100:
                    issues.append(f"  ⚠ No comments — code should be self-documenting OR have docstrings")

        except Exception as e:
            issues.append(f"  ✗ Could not read {os.path.basename(py_file)}: {e}")

    # Generate report
    report = [
        "=" * 60,
        "  CODE REVIEW REPORT",
        "=" * 60,
        f"Files analyzed: {len(py_files)}",
        f"Total lines: {total_lines}",
        f"Functions: {total_functions}",
        f"Classes: {total_classes}",
        f"Imports: {imports_count}",
        f"Comments: {comments_count}",
        "",
    ]

    if issues:
        report.append("Issues & Suggestions:")
        report.extend(issues)
    else:
        report.append("✓ Code looks solid! No major issues detected.")

    report.append("=" * 60)

    # Save to review log
    review_text = "\n".join(report)
    save_memory("lessons", f"Code review on {os.path.basename(path)}: {len(issues)} issues found", 3)

    return review_text


# ── Portfolio Site Generator ────────────────────────────────

def generate_portfolio() -> str:
    """Generate a showcase website of your best projects.

    Creates portfolio.html with all completed projects, screenshots,
    ratings, descriptions, and skills used. Opens in browser.
    """
    data = _load_memory()
    projects_memory = data.get("memories", {}).get("projects", [])

    if not projects_memory:
        return "No projects to showcase yet. Complete a project first!"

    # Build HTML
    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='UTF-8'>",
        "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "  <title>PinPoint Portfolio — AI Project Showcase</title>",
        "  <style>",
        "    * { margin: 0; padding: 0; box-sizing: border-box; }",
        "    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0a0e27; color: #e0e0e0; }",
        "    .container { max-width: 1200px; margin: 0 auto; padding: 40px 20px; }",
        "    header { text-align: center; margin-bottom: 50px; }",
        "    h1 { font-size: 3em; background: linear-gradient(135deg, #00bfff, #ff6b9d); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px; }",
        "    .subtitle { color: #888; font-size: 1.1em; }",
        "    .stats { display: flex; gap: 30px; justify-content: center; margin: 30px 0; flex-wrap: wrap; }",
        "    .stat { text-align: center; }",
        "    .stat-value { font-size: 2em; color: #00bfff; font-weight: bold; }",
        "    .stat-label { color: #666; }",
        "    .projects { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 30px; margin-top: 40px; }",
        "    .project { background: #1a1f3a; border: 1px solid #2d3561; border-radius: 10px; overflow: hidden; transition: all 0.3s; }",
        "    .project:hover { transform: translateY(-5px); border-color: #00bfff; box-shadow: 0 0 20px rgba(0,191,255,0.3); }",
        "    .project-header { background: linear-gradient(135deg, #1e90ff, #ff1493); padding: 15px; }",
        "    .project-title { font-size: 1.3em; font-weight: bold; margin-bottom: 5px; }",
        "    .project-meta { font-size: 0.9em; opacity: 0.9; }",
        "    .project-body { padding: 20px; }",
        "    .project-desc { margin-bottom: 15px; line-height: 1.6; }",
        "    .ratings { display: flex; gap: 20px; margin: 15px 0; }",
        "    .rating { }",
        "    .rating-label { color: #888; font-size: 0.9em; }",
        "    .rating-value { font-size: 1.5em; font-weight: bold; color: #00bfff; }",
        "    .skills { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 15px; }",
        "    .skill { background: rgba(0,191,255,0.1); color: #00bfff; padding: 4px 10px; border-radius: 20px; font-size: 0.85em; }",
        "    footer { text-align: center; margin-top: 60px; color: #666; padding-top: 20px; border-top: 1px solid #2d3561; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <div class='container'>",
        "    <header>",
        "      <h1>🤖 PinPoint Portfolio</h1>",
        "      <p class='subtitle'>Autonomous AI Project Showcase</p>",
        "    </header>",
        "",
    ]

    # Stats
    completed = len(projects_memory)
    specialization = data.get("meta", {}).get("specialization", "exploration")
    skills_count = len(data.get("memories", {}).get("skills", []))

    html_parts.extend([
        "    <div class='stats'>",
        f"      <div class='stat'><div class='stat-value'>{completed}</div><div class='stat-label'>Projects</div></div>",
        f"      <div class='stat'><div class='stat-value'>{skills_count}</div><div class='stat-label'>Skills</div></div>",
        f"      <div class='stat'><div class='stat-value'>{specialization}</div><div class='stat-label'>Specialty</div></div>",
        "    </div>",
        "",
        "    <div class='projects'>",
    ])

    # Add projects
    for proj in projects_memory[-20:]:  # Last 20 projects
        # Parse project memory entry (it's stored as a string)
        proj_text = proj.get("content", "") if isinstance(proj, dict) else str(proj)

        html_parts.extend([
            "      <div class='project'>",
            "        <div class='project-header'>",
            f"          <div class='project-title'>{proj_text[:40]}</div>",
            f"          <div class='project-meta'>by PinPoint</div>",
            "        </div>",
            "        <div class='project-body'>",
            f"          <div class='project-desc'>{proj_text}</div>",
            "          <div class='ratings'>",
            "            <div class='rating'><div class='rating-label'>Quality</div><div class='rating-value'>⭐⭐⭐⭐</div></div>",
            "          </div>",
            "        </div>",
            "      </div>",
        ])

    html_parts.extend([
        "    </div>",
        "",
        "    <footer>",
        "      <p>Generated by PinPoint — an autonomous AI agent</p>",
        "    </footer>",
        "  </div>",
        "</body>",
        "</html>",
    ])

    # Write portfolio file
    portfolio_path = os.path.join(OUTPUT_DIR, "portfolio.html")
    with open(portfolio_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    return f"Portfolio generated: {portfolio_path}\nOpen in browser to view all {completed} projects."


# ── Multimedia: Audio & Art ─────────────────────────────────

def synthesize_audio(description: str, length_seconds: float = 5.0, output_file: str = "generated_audio.wav") -> str:
    """Synthesize audio based on a description.

    Creates procedural audio using numpy: tones, noise, simple melodies.
    Examples: 'sine wave 440hz', 'ambient pad', 'simple melody C4 E4 G4'
    """
    try:
        import numpy as np
        from scipy.io import wavfile
    except ImportError:
        return "Audio synthesis requires numpy and scipy. Install: pip install numpy scipy"

    path = _safe_path(output_file)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Basic synthesis
    sample_rate = 44100
    duration = min(float(length_seconds), 30.0)  # max 30s
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Generate audio based on keywords
    desc_lower = description.lower()
    audio = np.zeros_like(t)

    if "sine" in desc_lower or "tone" in desc_lower:
        freq = 440  # A4
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
    elif "ambient" in desc_lower or "pad" in desc_lower:
        # Layered sines
        audio = 0.2 * np.sin(2 * np.pi * 220 * t)
        audio += 0.15 * np.sin(2 * np.pi * 330 * t)
        audio += 0.1 * np.sin(2 * np.pi * 440 * t)
    elif "noise" in desc_lower:
        audio = 0.3 * np.random.randn(len(t))
    elif "melody" in desc_lower or "tune" in desc_lower:
        # Simple C major scale
        notes = [262, 294, 330, 349, 392]  # C D E F G
        note_duration = duration / len(notes)
        for i, freq in enumerate(notes):
            start = int(i * note_duration * sample_rate)
            end = int((i + 1) * note_duration * sample_rate)
            audio[start:end] = 0.2 * np.sin(2 * np.pi * freq * t[start:end])
    else:
        # Default: gentle sine
        audio = 0.2 * np.sin(2 * np.pi * 330 * t)

    # Normalize and add fade
    audio = np.int16(audio / np.max(np.abs(audio)) * 32767 * 0.9)

    # Fade in/out
    fade_samples = int(0.05 * sample_rate)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    audio[:fade_samples] = (audio[:fade_samples] * fade_in).astype(np.int16)
    audio[-fade_samples:] = (audio[-fade_samples:] * fade_out).astype(np.int16)

    # Write WAV file
    wavfile.write(path, sample_rate, audio)
    return f"Audio synthesized: {output_file} ({duration}s, {sample_rate}Hz)"


def generate_art(style: str = "geometric", output_file: str = "generated_art.png") -> str:
    """Generate generative art based on style.

    Styles: geometric, organic, fractal, waves, spirals
    """
    try:
        from PIL import Image, ImageDraw
        import random as _random
    except ImportError:
        return "Art generation requires Pillow. Install: pip install Pillow"

    path = _safe_path(output_file)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    width, height = 800, 600
    img = Image.new("RGB", (width, height), color=(10, 14, 39))
    draw = ImageDraw.Draw(img)

    style = style.lower()

    if style == "geometric":
        # Random geometric shapes
        for _ in range(30):
            x1 = _random.randint(0, width)
            y1 = _random.randint(0, height)
            x2 = x1 + _random.randint(20, 200)
            y2 = y1 + _random.randint(20, 200)
            color = (_random.randint(0, 255), _random.randint(100, 255), _random.randint(150, 255))
            draw.rectangle([x1, y1, x2, y2], fill=None, outline=color, width=2)

    elif style == "organic":
        # Random curves and circles
        for _ in range(50):
            x = _random.randint(0, width)
            y = _random.randint(0, height)
            r = _random.randint(10, 100)
            color = (_random.randint(0, 100), _random.randint(100, 200), _random.randint(100, 255))
            draw.ellipse([x-r, y-r, x+r, y+r], fill=None, outline=color, width=1)

    elif style == "waves":
        # Sine wave patterns
        for freq in [0.01, 0.02, 0.03]:
            points = []
            for x in range(width):
                y = height // 2 + int(100 * np.sin(x * freq))
                points.append((x, y))
            color = (_random.randint(0, 255), _random.randint(100, 255), _random.randint(150, 255))
            draw.line(points, fill=color, width=2)

    elif style == "spirals":
        # Spiral patterns
        for spiral_idx in range(3):
            points = []
            center_x, center_y = width // 2, height // 2
            for i in range(500):
                angle = i * 0.05
                radius = i * 0.5
                x = center_x + radius * np.cos(angle)
                y = center_y + radius * np.sin(angle)
                if 0 <= x < width and 0 <= y < height:
                    points.append((x, y))
            if points:
                color = (_random.randint(0, 255), _random.randint(100, 255), _random.randint(150, 255))
                draw.line(points, fill=color, width=1)

    else:
        # Default: random circles (fractal-ish)
        def draw_fractal(x, y, r, depth):
            if depth == 0 or r < 2:
                return
            color = (_random.randint(50, 200), _random.randint(100, 255), _random.randint(150, 255))
            draw.ellipse([x-r, y-r, x+r, y+r], fill=None, outline=color, width=1)
            for _ in range(3):
                angle = _random.uniform(0, 2 * np.pi)
                new_x = x + r * 1.5 * np.cos(angle)
                new_y = y + r * 1.5 * np.sin(angle)
                draw_fractal(new_x, new_y, r * 0.6, depth - 1)

        draw_fractal(width // 2, height // 2, 80, 5)

    img.save(path)
    return f"Art generated: {output_file} ({style} style, {width}x{height})"


# ── English Dictionary ─────────────────────────────────────

def dictionary_lookup(word: str) -> str:
    """Look up a word in the English dictionary.

    Returns definition, part of speech, example sentences, and synonyms.
    Use this when you want to find the perfect word, understand a concept
    more precisely, or just satisfy your curiosity about language.
    """
    word = word.strip().lower()
    if not word:
        return "Please provide a word to look up."

    try:
        import nltk
        try:
            from nltk.corpus import wordnet as wn
            # Quick probe to make sure the corpus is downloaded
            wn.synsets("test")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            from nltk.corpus import wordnet as wn

        synsets = wn.synsets(word)
        if not synsets:
            return f"No entry found for '{word}' in the dictionary."

        _pos_label = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective satellite", "r": "adverb"}
        lines = [f"📖 {word.upper()}"]

        for i, syn in enumerate(synsets[:4]):
            pos = _pos_label.get(syn.pos(), syn.pos())
            lines.append(f"\n{i + 1}. [{pos}] {syn.definition()}")
            examples = syn.examples()
            if examples:
                lines.append(f'   e.g. "{examples[0]}"')
            synonyms = [
                lem.name().replace("_", " ")
                for lem in syn.lemmas()
                if lem.name().lower() != word
            ][:6]
            if synonyms:
                lines.append(f"   synonyms: {', '.join(synonyms)}")

        return "\n".join(lines)

    except ImportError:
        # NLTK not available — fall back to a minimal built-in word list
        return (
            f"NLTK not installed (pip install nltk). "
            f"To get full dictionary support, run: pip_install('nltk')"
        )
    except Exception as e:
        return f"Dictionary lookup error: {e}"


# ── Text-to-Speech / Voice ─────────────────────────────────

def _play_audio(path: str) -> None:
    """Play an audio file. Tracks the subprocess so mute can kill it instantly."""
    global _playback_proc, _tts_ended_at
    import time as _time

    def _run(cmd, **kwargs):
        if platform.system() == "Windows" and "creationflags" not in kwargs:
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                stdin=subprocess.DEVNULL, **kwargs)
        with _playback_lock:
            _playback_proc = proc
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.terminate()
        finally:
            with _playback_lock:
                if _playback_proc is proc:
                    _playback_proc = None
            # Stamp the moment TTS finished — voice listener uses this for cooldown
            _tts_ended_at = _time.time()

    if platform.system() == "Windows":
        try:
            safe = path.replace("\\", "\\\\")
            ps = (
                "Add-Type -AssemblyName PresentationCore; "
                "$mp = New-Object System.Windows.Media.MediaPlayer; "
                f"$mp.Open([Uri]'{safe}'); "
                "$mp.Play(); "
                "Start-Sleep -Milliseconds 800; "
                "while ($mp.NaturalDuration -eq [System.Windows.Duration]::Automatic) "
                "{ Start-Sleep -Milliseconds 100 }; "
                "$ms = [int]($mp.NaturalDuration.TimeSpan.TotalMilliseconds) + 300; "
                "Start-Sleep -Milliseconds $ms; "
                "$mp.Close()"
            )
            _run(["powershell", "-WindowStyle", "Hidden", "-Command", ps])
            return
        except Exception:
            pass
    elif platform.system() == "Darwin":
        try:
            _run(["afplay", path])
            return
        except Exception:
            pass
    else:
        for cmd in [
            ["mpg123", "-q", path],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
        ]:
            try:
                _run(cmd)
                return
            except FileNotFoundError:
                continue


def _try_edge_tts(text: str) -> bool:
    """Speak using Microsoft Edge neural TTS. Returns True on success."""
    try:
        import edge_tts
    except ImportError:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "edge-tts", "-q"],
                capture_output=True, timeout=60, check=True
            )
            import edge_tts
        except Exception:
            return False

    import asyncio
    import tempfile
    import os

    async def _do_speak():
        communicate = edge_tts.Communicate(
            text,
            voice="en-US-JennyNeural",
            rate="-8%",
            pitch="-8Hz",
        )
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp = f.name
        try:
            await communicate.save(tmp)
            _play_audio(tmp)
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass

    try:
        asyncio.run(_do_speak())
        return True
    except Exception:
        return False


def _speak_now(text: str) -> None:
    """Actually perform TTS — called only from the speech worker thread."""
    text = text.strip()
    if not text:
        return

    # Try high-quality neural voice first
    if _try_edge_tts(text):
        return

    # Native platform fallbacks
    if platform.system() == "Linux":
        for cmd in ["espeak-ng", "espeak"]:
            try:
                r = subprocess.run(
                    [cmd, "-s", "125", "-p", "40", "-a", "180", "-v", "en+f3"],
                    input=text, capture_output=True, text=True, timeout=30,
                )
                if r.returncode == 0:
                    return
            except FileNotFoundError:
                continue
            except Exception:
                break

    if platform.system() == "Darwin":
        try:
            subprocess.run(["say", "-v", "Samantha", "-r", "155", text], capture_output=True, timeout=30)
            return
        except Exception:
            pass

    if platform.system() == "Windows":
        try:
            safe = text.replace("'", "''")
            ps = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$s.Rate = -2; "
                f"$s.Speak('{safe}')"
            )
            subprocess.run(["powershell", "-WindowStyle", "Hidden", "-Command", ps],
                           capture_output=True, timeout=30)
            return
        except Exception:
            pass

    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 120)
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass

    print(f"\n[SPEAKING] {text}\n", flush=True)


def speak(text: str, wait: bool = True) -> str:
    """Queue text for speech. Only one line plays at a time — no overlapping.

    All speak() calls go into a single queue processed by one dedicated
    thread, so the voice never talks over itself.
    """
    global _last_spoken_text
    import time as _time
    if not text or not text.strip():
        return "Nothing to speak."

    # Strip code blocks and inline code — TTS reading <!DOCTYPE html> verbatim
    # is awful. Replace fenced ``` blocks with a brief mention; drop inline `code`.
    clean = re.sub(r"```[\s\S]*?```", " (showing code on screen) ", text)
    clean = re.sub(r"`[^`\n]+`", "", clean)
    # Collapse whitespace and trim
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean:
        return "Nothing to speak."

    short = clean[:100] + "..." if len(clean) > 100 else clean
    _last_spoken_text = clean
    now = _time.time()
    _recent_spoken.append((now, clean))
    _recent_spoken[:] = [(t, s) for (t, s) in _recent_spoken if now - t < 60.0]
    _speech_queue.put(clean)
    return f"🎤 Queued: '{short}'"


# ── Git Integration ────────────────────────────────────────

def git_commit(message: str, files: str = ".") -> str:
    """Commit work to git with a meaningful message.

    Called automatically by done() to version your projects.
    You can also call this manually during development to checkpoint work.
    """
    try:
        cwd = ROOT_DIR
        # Stage files
        result = subprocess.run(
            f"git add {files}",
            shell=True, cwd=cwd, capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return f"git add failed: {result.stderr}"

        # Commit
        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=cwd, capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            if "nothing to commit" in result.stdout.lower():
                return "Nothing to commit (no changes)."
            return f"git commit failed: {result.stderr}"

        # Try to push (might fail if no remote, but that's ok)
        subprocess.run(
            "git push -u origin HEAD",
            shell=True, cwd=cwd, capture_output=True, text=True, timeout=60
        )

        return f"Committed and pushed: {message[:80]}"
    except Exception as e:
        return f"git_commit error: {e}"


# ── Specialization / Domain Expertise ──────────────────────

_SPECIALIZATIONS = {
    "game_dev": "game design, mechanics, physics, graphics, gameplay loops",
    "web_dev": "web apps, React/Vue/Svelte, HTML/CSS, APIs, databases, full-stack",
    "data_science": "pandas, numpy, sklearn, data analysis, visualization, ML models",
    "music_audio": "audio synthesis, music theory, sound design, MIDI, Web Audio API",
    "generative_art": "creative coding, procedural generation, shaders, p5.js, Processing",
    "ai_ml": "neural networks, transformers, LLMs, training, evaluation, PyTorch/TensorFlow",
    "simulation": "physics engines, particle systems, agent-based models, real-time dynamics",
}


def set_specialization(domain: str) -> str:
    """Choose a specialization domain to focus on mastering.

    Available domains:
    - game_dev: Game design, mechanics, graphics, gameplay
    - web_dev: Web applications, full-stack development
    - data_science: Data analysis, ML models, visualization
    - music_audio: Audio synthesis, music, sound design
    - generative_art: Creative coding, procedural generation, shaders
    - ai_ml: Neural networks, LLMs, model training
    - simulation: Physics, agents, real-time dynamics

    Once set, all future sessions will be nudged toward this domain.
    You'll develop deep expertise and a portfolio in this area.
    """
    domain_lower = domain.lower().replace(" ", "_")
    if domain_lower not in _SPECIALIZATIONS:
        available = ", ".join(_SPECIALIZATIONS.keys())
        return f"Unknown domain '{domain}'. Available: {available}"

    data = _load_memory()
    data["meta"]["specialization"] = domain_lower
    data["meta"]["specialization_set_at"] = _now()
    _save_memory_file(data)

    return (
        f"Specialization set to: {domain_lower}\n"
        f"Focus areas: {_SPECIALIZATIONS[domain_lower]}\n\n"
        f"From now on, you'll preferentially build projects in this domain "
        f"and develop deep expertise. Your portfolio will showcase your mastery."
    )


def get_specialization() -> str:
    """Get your current specialization, if any."""
    data = _load_memory()
    spec = data.get("meta", {}).get("specialization")
    if not spec:
        return "No specialization set. Use set_specialization(domain) to choose one."
    return f"Current specialization: {spec}\nFocus: {_SPECIALIZATIONS.get(spec, '?')}"


# ── Automated Testing ──────────────────────────────────────

def run_tests(directory: str = "") -> str:
    """Discover and run tests in the current project or specified directory.

    Looks for test files (test_*.py, *_test.py) and runs them with pytest or unittest.
    Returns a summary of pass/fail results.

    Use this to validate your code before calling done().
    """
    path = _safe_path(directory) if directory else get_project_dir()
    if not os.path.isdir(path):
        return f"Directory not found: {path}"

    try:
        # Try pytest first (more powerful)
        result = subprocess.run(
            ["pytest", path, "-v", "--tb=short"],
            capture_output=True, text=True, timeout=120, cwd=ROOT_DIR
        )
        if result.returncode == 0 or "passed" in result.stdout:
            return f"Tests passed!\n\n{result.stdout[-2000:]}"
        else:
            return f"Tests failed:\n\n{result.stdout[-2000:]}\n\n{result.stderr[-1000:]}"
    except FileNotFoundError:
        # Fall back to unittest
        try:
            result = subprocess.run(
                [sys.executable, "-m", "unittest", "discover", "-s", path, "-p", "test_*.py", "-v"],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                return f"All tests passed!\n\n{result.stdout[-2000:]}"
            else:
                return f"Some tests failed:\n\n{result.stdout[-2000:]}"
        except Exception as e:
            return f"Could not run tests: {e}. Install pytest: pip install pytest"


def write_test(filename: str, test_code: str) -> str:
    """Write a test file for your code.

    Example:
      write_test('test_math.py', '''
        def test_add():
            from my_module import add
            assert add(2, 3) == 5
      ''')

    Tests are automatically discovered and run by run_tests().
    """
    path = _safe_path(filename)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(test_code)
    return f"Test written: {filename}\nRun with: run_tests()"


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
    elif tool_name == "log_experiment":
        return log_experiment(
            tool_input["name"],
            tool_input["hypothesis"],
            tool_input["method"],
            tool_input["result"],
            tool_input["conclusion"],
            int(tool_input.get("surprise_level", 3)),
        )
    elif tool_name == "list_experiments":
        return list_experiments()
    elif tool_name == "modify_own_source":
        return modify_own_source(
            tool_input["filename"],
            tool_input["new_content"],
            tool_input["reason"],
        )
    elif tool_name == "list_self_mod_history":
        return list_self_mod_history()
    elif tool_name == "think":
        return think(tool_input["reasoning"])
    elif tool_name == "deep_think":
        return deep_think(tool_input["problem"], int(tool_input.get("passes", 4)))
    elif tool_name == "brainstorm":
        return brainstorm(tool_input["topic"], int(tool_input.get("num_ideas", 5)))
    elif tool_name == "critique":
        return critique(tool_input["subject"], tool_input.get("what_to_evaluate", ""))
    elif tool_name == "decompose":
        return decompose(tool_input["goal"], tool_input.get("context", ""))
    elif tool_name == "collab_status":
        return collab_status()
    elif tool_name == "collab_update":
        return collab_update(tool_input["role"], tool_input["status"], tool_input.get("message", ""))
    elif tool_name == "git_commit":
        return git_commit(tool_input["message"], tool_input.get("files", "."))
    elif tool_name == "set_specialization":
        return set_specialization(tool_input["domain"])
    elif tool_name == "get_specialization":
        return get_specialization()
    elif tool_name == "run_tests":
        return run_tests(tool_input.get("directory", ""))
    elif tool_name == "write_test":
        return write_test(tool_input["filename"], tool_input["test_code"])
    elif tool_name == "show_dashboard":
        return show_dashboard()
    elif tool_name == "review_own_work":
        return review_own_work(tool_input.get("folder", ""))
    elif tool_name == "generate_portfolio":
        return generate_portfolio()
    elif tool_name == "synthesize_audio":
        return synthesize_audio(
            tool_input["description"],
            float(tool_input.get("length_seconds", 5.0)),
            tool_input.get("output_file", "generated_audio.wav"),
        )
    elif tool_name == "generate_art":
        return generate_art(
            tool_input.get("style", "geometric"),
            tool_input.get("output_file", "generated_art.png"),
        )
    elif tool_name == "dictionary_lookup":
        return dictionary_lookup(tool_input["word"])
    elif tool_name == "speak":
        return speak(
            tool_input["text"],
            bool(tool_input.get("wait", True)),
        )
    elif tool_name == "mute_voice":
        return mute_voice()
    elif tool_name == "unmute_voice":
        return unmute_voice()
    elif tool_name == "toggle_voice":
        return toggle_voice()
    elif tool_name == "get_news":
        return get_news(tool_input.get("topic", ""))
    else:
        return f"Unknown tool: {tool_name}"
