"""Crash-safe JSON persistence shared by the v3 subsystems.

Writes go through a temp file + ``os.replace`` so a crash mid-write can never
leave a half-written state file behind. Reads never raise: a missing or corrupt
file yields the supplied default, because losing state must not take the agent
down with it.
"""

import json
import os
import tempfile
from typing import Any


def read_json(path: str, default: Any = None) -> Any:
    """Load JSON from ``path``; return ``default`` if missing or unreadable."""
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeDecodeError):
        return default


def write_json(path: str, data: Any) -> bool:
    """Atomically write ``data`` as JSON. Returns True on success."""
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=parent or ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp, path)
            return True
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception:
        return False


def append_jsonl(path: str, record: dict) -> bool:
    """Append one JSON object as a line. Used for append-only logs."""
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        return True
    except Exception:
        return False


def read_jsonl(path: str, limit: int = 0) -> list:
    """Read a JSONL file into a list, skipping unparseable lines.

    ``limit`` > 0 returns only the last N records.
    """
    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return []
    return records[-limit:] if limit > 0 else records
