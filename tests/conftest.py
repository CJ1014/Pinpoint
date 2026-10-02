"""Shared test fixtures.

Every test runs against a throwaway PINPOINT_HOME so no test can read or
clobber CJ's real memory.json, output/, or audit log.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Redirect all PinPoint state into a temp dir for the duration of a test."""
    monkeypatch.setenv("PINPOINT_HOME", str(tmp_path))
    os.makedirs(tmp_path / "output", exist_ok=True)
    yield tmp_path
