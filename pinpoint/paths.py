"""Filesystem locations for PinPoint state.

Every path is resolved *at call time* from ``PINPOINT_HOME`` so that tests can
redirect all state into a temporary directory without reloading modules. The
default is the repository root, which is where PinPoint v2 already writes.
"""

import os

# The repo root — where agent.py / tools.py / memory.json live.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def home() -> str:
    """Root directory for all PinPoint state. Override with PINPOINT_HOME."""
    return os.environ.get("PINPOINT_HOME") or REPO_ROOT


def output_dir(*parts: str) -> str:
    """A path under ``<home>/output``, creating parent directories."""
    path = os.path.join(home(), "output", *parts)
    os.makedirs(os.path.dirname(path) if parts else path, exist_ok=True)
    return path


def state_path(name: str) -> str:
    """A JSON state file directly under ``<home>``."""
    return os.path.join(home(), name)
