import os
import subprocess
import json
import re
from html.parser import HTMLParser

import httpx

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def _safe_path(filename: str) -> str:
    """Resolve path inside output/ and reject any traversal attempts."""
    resolved = os.path.realpath(os.path.join(OUTPUT_DIR, filename))
    if not resolved.startswith(os.path.realpath(OUTPUT_DIR)):
        raise ValueError(f"Path traversal not allowed: {filename}")
    return resolved


def write_file(filename: str, content: str) -> str:
    path = _safe_path(filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Written {len(content)} chars to output/{filename}"


def read_file(filename: str) -> str:
    path = _safe_path(filename)
    if not os.path.exists(path):
        return f"File not found: output/{filename}"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


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
        return f"File not found: output/{filename}"
    try:
        result = subprocess.run(
            ["python3", path],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=OUTPUT_DIR,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        parts = []
        if out:
            parts.append(f"stdout:\n{out}")
        if err:
            parts.append(f"stderr:\n{err}")
        parts.append(f"exit code: {result.returncode}")
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return "Execution timed out after 30 seconds."
    except Exception as e:
        return f"Error running script: {e}"


def open_html(filename: str) -> str:
    import webbrowser
    path = _safe_path(filename)
    if not os.path.exists(path):
        return f"File not found: output/{filename}"
    url = "file:///" + path.replace("\\", "/")
    webbrowser.open(url)
    return f"Opened output/{filename} in default browser."


def done(summary: str) -> str:
    return f"DONE: {summary}"


class _TextExtractor(HTMLParser):
    """Strip HTML tags and return plain text."""
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
        # Collapse excessive whitespace
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
            # Truncate to keep response manageable
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
            "User-Agent": "Mozilla/5.0 (compatible; PinpointBot/1.0)",
            "Accept": "application/json",
        }
        params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
        resp = httpx.get(
            "https://api.duckduckgo.com/",
            params=params,
            headers=headers,
            timeout=10,
        )
        data = resp.json()
        results = []

        # Abstract (best direct answer)
        if data.get("AbstractText"):
            results.append(f"Summary: {data['AbstractText']}")
            if data.get("AbstractURL"):
                results.append(f"Source: {data['AbstractURL']}")
            results.append("")

        # Related topics
        for topic in data.get("RelatedTopics", [])[:8]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(f"- {topic['Text']}")
                if topic.get("FirstURL"):
                    results.append(f"  URL: {topic['FirstURL']}")

        if not results:
            return f"No results found for: {query}"
        return "\n".join(results)
    except Exception as e:
        return f"Error searching web: {e}"


# Tool definitions for the Anthropic API
TOOL_DEFINITIONS = [
    {
        "name": "write_file",
        "description": (
            "Write content to a file inside the output/ directory. "
            "Use this to create programs, scripts, data files, stories, or anything else you want to build."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Filename (optionally with subdirectory, e.g. 'game.py' or 'data/config.json'). Stays inside output/.",
                },
                "content": {
                    "type": "string",
                    "description": "Full text content to write to the file.",
                },
            },
            "required": ["filename", "content"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file you have previously written inside output/.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Filename relative to output/.",
                }
            },
            "required": ["filename"],
        },
    },
    {
        "name": "list_files",
        "description": "List all files you have created in the output/ directory.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "run_python",
        "description": (
            "Execute a Python script you have written inside output/ and see its output. "
            "Use this to test your code, run simulations, generate data, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Python filename relative to output/ (e.g. 'simulation.py').",
                }
            },
            "required": ["filename"],
        },
    },
    {
        "name": "done",
        "description": (
            "Call this when you are completely finished with your creative work. "
            "Provide a summary of everything you created."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "A description of everything you built and why you chose to create it.",
                }
            },
            "required": ["summary"],
        },
    },
    {
        "name": "open_html",
        "description": (
            "Open an HTML file you created in the user's default web browser. "
            "Use this after writing an HTML game or webpage to let the user see and interact with it. "
            "The file must be inside output/."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "HTML filename relative to output/ (e.g. 'game.html').",
                }
            },
            "required": ["filename"],
        },
    },
    {
        "name": "search_web",
        "description": "Search the web using DuckDuckGo and return relevant results for a query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": "Fetch the text content of any web page by URL.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL to fetch (e.g. 'https://example.com/article').",
                }
            },
            "required": ["url"],
        },
    },
]


# OpenAI-compatible tool definitions (for AgentRouter / OpenAI SDK)
OPENAI_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write content to a file inside the output/ directory. "
                "Use this to create programs, scripts, data files, stories, or anything else you want to build."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Filename (optionally with subdirectory, e.g. 'game.py' or 'data/config.json'). Stays inside output/.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full text content to write to the file.",
                    },
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file you have previously written inside output/.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Filename relative to output/.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files you have created in the output/ directory.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute a Python script you have written inside output/ and see its output. "
                "Use this to test your code, run simulations, generate data, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Python filename relative to output/ (e.g. 'simulation.py').",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": (
                "Call this when you are completely finished with your creative work. "
                "Provide a summary of everything you created."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "A description of everything you built and why you chose to create it.",
                    }
                },
                "required": ["summary"],
            },
        },
    },
]


def dispatch(tool_name: str, tool_input: dict) -> str:
    if tool_name == "write_file":
        return write_file(tool_input["filename"], tool_input["content"])
    elif tool_name == "read_file":
        return read_file(tool_input["filename"])
    elif tool_name == "list_files":
        return list_files()
    elif tool_name == "run_python":
        return run_python(tool_input["filename"])
    elif tool_name == "done":
        return done(tool_input["summary"])
    elif tool_name == "open_html":
        return open_html(tool_input["filename"])
    elif tool_name == "search_web":
        return search_web(tool_input["query"])
    elif tool_name == "fetch_url":
        return fetch_url(tool_input["url"])
    else:
        return f"Unknown tool: {tool_name}"
