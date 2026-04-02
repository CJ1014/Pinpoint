"""Run this once to remove fireworks-related entries from memory.json"""
import json, os, re

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "memory.json")
BANNED = ["firework", "fireworks", "spark", "explosion", "particle burst", "confetti"]

if not os.path.exists(MEMORY_FILE):
    print("memory.json not found.")
    raise SystemExit

with open(MEMORY_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

removed = 0
for category, entries in data.get("memories", {}).items():
    before = len(entries)
    data["memories"][category] = [
        e for e in entries
        if not any(b in e.get("content", "").lower() for b in BANNED)
    ]
    removed += before - len(data["memories"][category])

# Also clear last_project if it was fireworks
last = data.get("meta", {}).get("last_project", {})
if last and any(b in last.get("summary", "").lower() for b in BANNED):
    data["meta"].pop("last_project", None)
    print("Cleared last_project (was fireworks).")

with open(MEMORY_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Done. Removed {removed} fireworks-related memory entries.")
