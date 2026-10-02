# goal_tree.py — Phase 1: Hierarchical Goal Decomposition

import json
from typing import Dict, List, Optional, Set
from datetime import datetime

class GoalNode:
    """Single node in the goal tree."""
    def __init__(self, goal: str, depth: int = 0, parent_goal: Optional[str] = None):
        self.goal = goal
        self.depth = depth
        self.parent_goal = parent_goal
        self.subgoals: List["GoalNode"] = []
        self.confidence: float = 0.8
        self.is_impossible: bool = False
        self.impossible_reason: str = ""
        self.fallback: Optional[str] = None
        self.created_at = datetime.now().isoformat()
        self.completed: bool = False
        self.attempts: int = 0

    def to_dict(self) -> dict:
        """Serialize to JSON."""
        return {
            "goal": self.goal,
            "depth": self.depth,
            "parent": self.parent_goal,
            "confidence": self.confidence,
            "is_impossible": self.is_impossible,
            "impossible_reason": self.impossible_reason,
            "fallback": self.fallback,
            "completed": self.completed,
            "attempts": self.attempts,
            "subgoals": [s.to_dict() for s in self.subgoals],
        }

class GoalTree:
    """Hierarchical decomposition of a goal into executable subgoals."""

    def __init__(self, root_goal: str):
        self.root = GoalNode(root_goal, depth=0)
        self.all_nodes: List[GoalNode] = [self.root]
        self.pruned_branches: List[Dict] = []
        self.execution_order: List[str] = []
        self.dependencies: Dict[str, Set[str]] = {}

    def add_subgoal(self, parent_goal: str, subgoal: str, confidence: float = 0.8,
                   depends_on: List[str] = None):
        """Add a subgoal under a parent goal."""
        for node in self.all_nodes:
            if node.goal == parent_goal:
                child = GoalNode(subgoal, depth=node.depth + 1, parent_goal=parent_goal)
                child.confidence = confidence
                node.subgoals.append(child)
                self.all_nodes.append(child)

                # Track dependencies
                if depends_on:
                    self.dependencies[subgoal] = set(depends_on)

                return True
        return False

    def mark_impossible(self, goal_str: str, reason: str):
        """Mark a branch as impossible and log why."""
        for node in self.all_nodes:
            if node.goal == goal_str:
                node.is_impossible = True
                node.impossible_reason = reason
                self.pruned_branches.append({
                    "goal": goal_str,
                    "reason": reason,
                    "depth": node.depth,
                    "timestamp": datetime.now().isoformat()
                })
                return True
        return False

    def mark_completed(self, goal_str: str):
        """Mark a goal as done."""
        for node in self.all_nodes:
            if node.goal == goal_str:
                node.completed = True
                return True
        return False

    def record_attempt(self, goal_str: str):
        """Track that we tried a goal (useful for detecting stuck loops)."""
        for node in self.all_nodes:
            if node.goal == goal_str:
                node.attempts += 1
                return True
        return False

    def get_executable_actions(self) -> List[str]:
        """Get leaf nodes that aren't completed or impossible."""
        leaves = []
        def traverse(node):
            if node.is_impossible or node.completed:
                return
            # Only add leaf nodes (no subgoals)
            if not node.subgoals:
                leaves.append(node.goal)
            else:
                for child in node.subgoals:
                    traverse(child)
        traverse(self.root)
        return leaves

    def get_completion_status(self) -> Dict:
        """How much of the tree is done? Counted over executable leaf nodes —
        parent/phase nodes are structure, not work items."""
        leaves = [n for n in self.all_nodes if not n.subgoals]
        total = len(leaves)
        completed = sum(1 for n in leaves if n.completed)
        impossible = sum(1 for n in leaves if n.is_impossible and not n.completed)
        active = total - completed - impossible

        return {
            "total": total,
            "completed": completed,
            "impossible": impossible,
            "active": active,
            "percent_done": (completed / total * 100) if total > 0 else 0,
        }

    def get_next_priority_action(self) -> Optional[str]:
        """Get the NEXT action to take (respects dependencies)."""
        executable = self.get_executable_actions()
        if not executable:
            return None

        # Prefer actions whose dependencies are all completed
        completed_goals = {n.goal for n in self.all_nodes if n.completed}
        for action in executable:
            deps = self.dependencies.get(action, set())
            if deps.issubset(completed_goals):
                return action
        return executable[0]

    def to_dict(self) -> dict:
        """Full tree as JSON."""
        status = self.get_completion_status()
        return {
            "root_goal": self.root.goal,
            "tree": self.root.to_dict(),
            "pruned_branches": self.pruned_branches,
            "executable_actions": self.get_executable_actions(),
            "next_priority": self.get_next_priority_action(),
            "status": status,
            "total_nodes": len(self.all_nodes),
            "dependencies": {k: list(v) for k, v in self.dependencies.items()},
        }

    def to_markdown(self) -> str:
        """Render tree as markdown for display."""
        lines = [f"# Goal Tree: {self.root.goal}\n"]

        status = self.get_completion_status()
        lines.append(f"**Progress: {status['completed']}/{status['total']} done ({status['percent_done']:.1f}%)**\n")

        def render_node(node, prefix=""):
            if node.is_impossible:
                status_str = f"❌ IMPOSSIBLE ({node.impossible_reason})"
            elif node.completed:
                status_str = "✅ DONE"
            elif node.subgoals:
                completed_sub = sum(1 for s in node.subgoals if s.completed)
                status_str = f"→ [{completed_sub}/{len(node.subgoals)}]"
            else:
                status_str = f"▪ EXECUTABLE (conf: {node.confidence:.0%})"

            lines.append(f"{prefix}• {node.goal} {status_str}")

            for i, child in enumerate(node.subgoals):
                is_last = i == len(node.subgoals) - 1
                child_prefix = prefix + ("  " if is_last else "│ ")
                render_node(child, child_prefix)

        render_node(self.root)

        if self.get_executable_actions():
            lines.append("\n**Next executable actions:**")
            for action in self.get_executable_actions()[:5]:
                lines.append(f"→ {action}")

        return "\n".join(lines)

    def to_text_summary(self) -> str:
        """Simple text summary for logging."""
        executable = self.get_executable_actions()
        status = self.get_completion_status()
        next_action = self.get_next_priority_action()

        text = f"""GOAL TREE STATUS
Root: {self.root.goal}
Progress: {status['completed']}/{status['total']} ({status['percent_done']:.0f}%)
Active: {status['active']} | Blocked: {status['impossible']}

Next Action: {next_action if next_action else 'None (tree complete)'}

Executable Queue: {len(executable)} actions
"""
        if executable:
            for a in executable[:5]:
                text += f"  → {a}\n"
            if len(executable) > 5:
                text += f"  ... and {len(executable) - 5} more\n"

        return text


# ── Compatibility layer ───────────────────────────────────────────────────────
# Existing integration (agent.py auto-decompose, tools.decompose_goal) imports
# decompose_and_save(goal, context). Builds the standard phase decomposition
# deterministically (no LLM round-trip) and persists it for the live viewer.

def build_auto_tree(goal: str, depth: int = 2) -> GoalTree:
    """Standard phase decomposition used by decompose_goal_auto."""
    tree = GoalTree(goal)
    if depth >= 1:
        tree.add_subgoal(goal, "[1] Plan & Research", 0.9)
        tree.add_subgoal(goal, "[2] Design", 0.85)
        tree.add_subgoal(goal, "[3] Implement", 0.8)
        tree.add_subgoal(goal, "[4] Test & Validate", 0.75)
        tree.add_subgoal(goal, "[5] Documentation", 0.7)
    if depth >= 2:
        tree.add_subgoal("[2] Design", "Architecture sketch")
        tree.add_subgoal("[2] Design", "Data model")
        tree.add_subgoal("[2] Design", "API specification")
    return tree


def _save_tree_to_memory(tree: GoalTree) -> None:
    try:
        from tools import _load_memory, _save_memory_file  # lazy — avoids circular import
        mem = _load_memory()
        mem["active_goal_decomposition"] = {
            "root_goal": tree.root.goal,
            "tree": tree.root.to_dict(),
            "pruned_branches": tree.pruned_branches,
            "execution_order": tree.get_executable_actions(),
        }
        hist = mem.get("decomposition_history", [])
        hist.append({
            "goal": tree.root.goal,
            "total_nodes": len(tree.all_nodes),
            "branches_pruned": len(tree.pruned_branches),
            "timestamp": datetime.now().isoformat(),
        })
        mem["decomposition_history"] = hist[-25:]
        _save_memory_file(mem)
    except Exception:
        pass


def decompose_and_save(goal: str, context: str = "") -> str:
    """Build the auto tree, persist for the viewer, return JSON string."""
    tree = build_auto_tree(goal, depth=2)
    _save_tree_to_memory(tree)
    return json.dumps(tree.to_dict(), indent=2)
