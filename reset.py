"""
PinPoint Reset — wipes all memories, projects, and logs.
Run with:  python reset.py
"""
import os
import shutil

ROOT = os.path.dirname(__file__)
MEMORY_FILE = os.path.join(ROOT, "memory.json")
OUTPUT_DIR = os.path.join(ROOT, "output")
LOGS = [
    os.path.join(ROOT, "self_mod_log.txt"),
    os.path.join(ROOT, "experiments_log.txt"),
]

print("PinPoint Reset")
print("=" * 40)

# Delete memory.json
if os.path.exists(MEMORY_FILE):
    os.remove(MEMORY_FILE)
    print("Deleted memory.json")
else:
    print("memory.json not found (already clean)")

# Delete output/ folder (all projects, logs, world state)
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
    print(f"Deleted output/ ({OUTPUT_DIR})")
else:
    print("output/ not found (already clean)")

# Delete root-level logs
for log in LOGS:
    if os.path.exists(log):
        os.remove(log)
        print(f"Deleted {os.path.basename(log)}")

print("=" * 40)
print("Done. PinPoint will start completely fresh next run.")
