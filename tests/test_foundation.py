"""Stage 1: paths + crash-safe JSON store."""

import json
import os

from pinpoint import jsonstore, paths


def test_home_follows_env(isolated_home):
    assert paths.home() == str(isolated_home)


def test_home_defaults_to_repo_root(monkeypatch):
    monkeypatch.delenv("PINPOINT_HOME", raising=False)
    assert os.path.isfile(os.path.join(paths.home(), "agent.py"))


def test_output_dir_creates_parents(isolated_home):
    p = paths.output_dir("checkpoints", "a.json")
    assert os.path.isdir(os.path.dirname(p))


def test_write_then_read_roundtrip(isolated_home):
    p = paths.state_path("x.json")
    assert jsonstore.write_json(p, {"a": 1})
    assert jsonstore.read_json(p) == {"a": 1}


def test_read_json_survives_corruption(isolated_home):
    p = paths.state_path("bad.json")
    with open(p, "w", encoding="utf-8") as f:
        f.write("{not json")
    assert jsonstore.read_json(p, {"fallback": True}) == {"fallback": True}


def test_read_json_missing_returns_default(isolated_home):
    assert jsonstore.read_json(paths.state_path("nope.json"), []) == []


def test_write_json_is_atomic_no_temp_left(isolated_home):
    p = paths.state_path("atomic.json")
    jsonstore.write_json(p, {"v": 1})
    leftovers = [f for f in os.listdir(isolated_home) if f.endswith(".tmp")]
    assert leftovers == []


def test_jsonl_append_and_read(isolated_home):
    p = paths.output_dir("log.jsonl")
    jsonstore.append_jsonl(p, {"n": 1})
    jsonstore.append_jsonl(p, {"n": 2})
    assert [r["n"] for r in jsonstore.read_jsonl(p)] == [1, 2]


def test_jsonl_limit_returns_tail(isolated_home):
    p = paths.output_dir("log.jsonl")
    for i in range(5):
        jsonstore.append_jsonl(p, {"n": i})
    assert [r["n"] for r in jsonstore.read_jsonl(p, limit=2)] == [3, 4]


def test_jsonl_skips_corrupt_lines(isolated_home):
    p = paths.output_dir("log.jsonl")
    jsonstore.append_jsonl(p, {"n": 1})
    with open(p, "a", encoding="utf-8") as f:
        f.write("garbage\n")
    jsonstore.append_jsonl(p, {"n": 2})
    assert len(jsonstore.read_jsonl(p)) == 2


def test_write_json_serializes_unknown_types(isolated_home):
    from datetime import datetime
    p = paths.state_path("dt.json")
    assert jsonstore.write_json(p, {"t": datetime(2026, 1, 1)})
    assert "2026-01-01" in json.dumps(jsonstore.read_json(p))
