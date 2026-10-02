# checkpoint.py — Phase 5: Long-Horizon Continuity

import json
import os
from typing import Dict, List, Optional
from datetime import datetime

class Checkpoint:
    """Compressed session snapshot for resuming later."""

    def __init__(self, root_goal: str, session_num: int):
        self.id = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.root_goal = root_goal
        self.session_num = session_num
        self.completed_tasks: List[str] = []
        self.pending_tasks: List[str] = []
        self.reasoning_summary: str = ""
        self.lessons_learned: List[str] = []
        self.files_created: List[str] = []
        self.files_modified: List[str] = []
        self.next_steps: List[str] = []
        self.blockers: List[Dict] = []
        self.memory_snapshot: Dict = {}
        self.goal_tree_snapshot: Dict = {}

    def add_completed(self, task: str):
        self.completed_tasks.append(task)
        if task in self.pending_tasks:
            self.pending_tasks.remove(task)

    def add_pending(self, task: str):
        if task not in self.pending_tasks:
            self.pending_tasks.append(task)

    def add_lesson(self, lesson: str):
        if lesson not in self.lessons_learned:
            self.lessons_learned.append(lesson)

    def add_blocker(self, description: str, reason: str):
        self.blockers.append({
            "description": description,
            "reason": reason,
            "logged_at": datetime.now().isoformat()
        })

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": datetime.now().isoformat(),
            "root_goal": self.root_goal,
            "session_num": self.session_num,
            "completed_tasks": self.completed_tasks,
            "pending_tasks": self.pending_tasks,
            "reasoning_summary": self.reasoning_summary,
            "lessons_learned": self.lessons_learned,
            "files_created": self.files_created,
            "files_modified": self.files_modified,
            "next_steps": self.next_steps,
            "blockers": self.blockers,
        }

    def save(self, output_dir: str = "output/checkpoints") -> str:
        """Write checkpoint to disk."""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{self.id}.json")
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return filepath

    @staticmethod
    def load(checkpoint_id: str, output_dir: str = "output/checkpoints") -> Optional["Checkpoint"]:
        """Load a checkpoint from disk."""
        filepath = os.path.join(output_dir, f"{checkpoint_id}.json")
        if not os.path.exists(filepath):
            return None

        with open(filepath, "r") as f:
            data = json.load(f)

        cp = Checkpoint(data["root_goal"], data["session_num"])
        cp.id = data["id"]
        cp.completed_tasks = data.get("completed_tasks", [])
        cp.pending_tasks = data.get("pending_tasks", [])
        cp.reasoning_summary = data.get("reasoning_summary", "")
        cp.lessons_learned = data.get("lessons_learned", [])
        cp.files_created = data.get("files_created", [])
        cp.files_modified = data.get("files_modified", [])
        cp.next_steps = data.get("next_steps", [])
        cp.blockers = data.get("blockers", [])
        return cp

    @staticmethod
    def list_all(output_dir: str = "output/checkpoints") -> List[str]:
        """List all checkpoint IDs."""
        if not os.path.exists(output_dir):
            return []
        return sorted([
            f[:-5] for f in os.listdir(output_dir)
            if f.endswith(".json") and f.startswith("checkpoint_")
        ])

    @staticmethod
    def get_latest(output_dir: str = "output/checkpoints") -> Optional["Checkpoint"]:
        """Load the most recent checkpoint."""
        checkpoints = Checkpoint.list_all(output_dir)
        if not checkpoints:
            return None
        return Checkpoint.load(checkpoints[-1], output_dir)

    def to_resume_text(self) -> str:
        """Human-readable summary for resuming."""
        text = f"""
╔═══════════════════════════════════════════════════════╗
║              RESUMING FROM CHECKPOINT                 ║
╚═══════════════════════════════════════════════════════╝

Checkpoint: {self.id}
Session: {self.session_num}

GOAL: {self.root_goal}

PROGRESS: {len(self.completed_tasks)}/{len(self.completed_tasks) + len(self.pending_tasks)} tasks done

COMPLETED:
{chr(10).join(f'  ✓ {t}' for t in self.completed_tasks[:10])}
{f"  ... and {len(self.completed_tasks) - 10} more" if len(self.completed_tasks) > 10 else ""}

PENDING:
{chr(10).join(f'  → {t}' for t in self.pending_tasks[:10])}

NEXT STEPS:
{chr(10).join(f'  1. {s}' for s in self.next_steps[:3])}

LESSONS LEARNED:
{chr(10).join(f'  • {l}' for l in self.lessons_learned[:5])}

{f"BLOCKERS: {len(self.blockers)}" if self.blockers else ""}
{chr(10).join(f"  ⚠ {b['description']}" for b in self.blockers[:3]) if self.blockers else ""}

═════════════════════════════════════════════════════════
"""
        return text
