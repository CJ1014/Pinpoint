"""
goal_tree.py — GoalTree module for PinPoint

Decomposes a root goal into a structured, executable tree using the LLM.
Stores the result in memory.json under "active_goal_decomposition".
"""

import json
import re
import logging
from collections import deque

from agent import MODEL, AGENT_MODEL, get_llm_client, chat_completion

logger = logging.getLogger(__name__)

_DECOMPOSE_PROMPT = """\
You are a goal decomposition engine for an autonomous AI agent called PinPoint.
Break the following goal into a structured execution plan.

Goal: {goal}
{context_line}
Output ONLY valid JSON with this exact structure — no markdown, no explanation, no extra keys:
{{
  "goal": "<the root goal>",
  "confidence": <float 0.0-1.0>,
  "impossible": <true or false>,
  "reason": "<why impossible, or empty string if not impossible>",
  "subgoals": [
    {{
      "goal": "<subgoal description>",
      "confidence": <float 0.0-1.0>,
      "tasks": [
        {{
          "action": "<concrete action to take>",
          "depends_on": ["<action string this task depends on>"]
        }}
      ]
    }}
  ]
}}

Rules:
- Maximum 3 subgoals.
- Maximum 3 tasks per subgoal.
- "depends_on" is a list of "action" strings from OTHER tasks (can be in earlier subgoals).
- If no dependencies, use an empty list [].
- If the goal is truly impossible or self-contradictory, set impossible=true and explain in reason.
- Keep tasks concrete and individually executable.
- Output ONLY the JSON object — nothing else.
"""


def _flat_fallback(goal: str) -> dict:
    """Return a minimal valid tree when the LLM fails or produces unparseable output."""
    return {
        "goal": goal,
        "confidence": 0.5,
        "impossible": False,
        "reason": "",
        "subgoals": [
            {
                "goal": goal,
                "confidence": 0.5,
                "tasks": [
                    {
                        "action": f"Complete: {goal}",
                        "depends_on": [],
                    }
                ],
            }
        ],
    }


class GoalTree:
    """
    Represents a decomposed goal as a tree of subgoals and tasks.

    Attributes:
        root_goal       The original goal string.
        tree            The decomposed tree dict (populated by decompose()).
        pruned_branches List of {branch_path, reason} dicts marking pruned branches.
    """

    def __init__(self, root_goal: str):
        self.root_goal = root_goal
        self.tree: dict = {}
        self.pruned_branches: list = []

    # ------------------------------------------------------------------
    # decompose
    # ------------------------------------------------------------------

    def decompose(self, context: str = "") -> dict:
        """
        Use the LLM (AGENT_MODEL) to break the root goal into a JSON tree.

        Returns the parsed tree dict. On any parse/network error, falls back to
        a flat single-task tree so the caller always gets a usable structure.

        Tree schema:
          {
            goal: str,
            confidence: float,
            impossible: bool,
            reason: str,
            subgoals: [
              {
                goal: str,
                confidence: float,
                tasks: [
                  { action: str, depends_on: [str] }
                ]
              }
            ]
          }
        """
        context_line = f"Context: {context}" if context else ""
        prompt = _DECOMPOSE_PROMPT.format(
            goal=self.root_goal,
            context_line=context_line,
        )

        try:
            client = get_llm_client(timeout=120.0)
            messages = [{"role": "user", "content": prompt}]
            response = chat_completion(
                client=client,
                messages=messages,
                temperature=0.3,
                max_tokens=1500,
                model=AGENT_MODEL,
            )
            raw = (response.choices[0].message.content or "").strip()

            # Strip markdown code fences if the model wrapped the JSON
            if raw.startswith("```"):
                # Remove opening fence line
                raw = raw.split("\n", 1)[-1]
                # Remove closing fence
                raw = raw.rsplit("```", 1)[0].strip()

            # If there's surrounding prose, extract the JSON object
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                raw = json_match.group(0)

            tree = json.loads(raw)

            if not isinstance(tree, dict):
                raise ValueError("LLM returned non-dict JSON")

            # Normalize top-level keys
            tree.setdefault("goal", self.root_goal)
            tree.setdefault("confidence", 0.7)
            tree.setdefault("impossible", False)
            tree.setdefault("reason", "")
            if not isinstance(tree.get("subgoals"), list):
                tree["subgoals"] = []

            # Normalize subgoals and tasks
            for sg in tree["subgoals"]:
                if not isinstance(sg, dict):
                    continue
                sg.setdefault("goal", "unnamed subgoal")
                sg.setdefault("confidence", 0.7)
                if not isinstance(sg.get("tasks"), list):
                    sg["tasks"] = []
                for task in sg["tasks"]:
                    if not isinstance(task, dict):
                        continue
                    task.setdefault("action", "unnamed task")
                    if not isinstance(task.get("depends_on"), list):
                        task["depends_on"] = []
                    # Ensure every depends_on entry is a string
                    task["depends_on"] = [str(d) for d in task["depends_on"] if d]

            self.tree = tree
            return tree

        except Exception as exc:
            logger.warning(
                "GoalTree.decompose failed for %r: %s — using flat fallback",
                self.root_goal,
                exc,
            )
            self.tree = _flat_fallback(self.root_goal)
            return self.tree

    # ------------------------------------------------------------------
    # prune
    # ------------------------------------------------------------------

    def prune(self, branch_path: str, reason: str) -> None:
        """
        Mark a branch as pruned.

        branch_path is a human-readable identifier for the branch — typically
        the subgoal goal string, a subgoal index, or any label the caller
        wants to use.  Pruned branches are excluded from get_execution_order().
        """
        self.pruned_branches.append(
            {
                "branch_path": branch_path,
                "reason": reason,
            }
        )

    # ------------------------------------------------------------------
    # get_execution_order
    # ------------------------------------------------------------------

    def get_execution_order(self) -> list:
        """
        Return a flat list of tasks in dependency order (topological sort).

        Each item is a dict:
          {
            "task":    str   — the action string,
            "subgoal": str   — the parent subgoal's goal string,
            "fallback": str  — what to do if this task is blocked,
          }

        Pruned subgoals are excluded.  Dependency cycles are broken by
        appending cycle members at the end in their original order.
        """
        if not self.tree:
            return []

        # Build a set of pruned identifiers (goal strings and index strings)
        pruned_ids = set()
        for p in self.pruned_branches:
            pruned_ids.add(p["branch_path"])

        # --- Collect tasks ---------------------------------------------------
        all_tasks: list = []  # [{action, subgoal, depends_on}]

        for sg_idx, sg in enumerate(self.tree.get("subgoals", [])):
            if not isinstance(sg, dict):
                continue
            sg_goal = sg.get("goal", f"subgoal_{sg_idx}")
            # Skip if pruned by goal string or by index
            if sg_goal in pruned_ids or str(sg_idx) in pruned_ids:
                continue
            for task in sg.get("tasks", []):
                if not isinstance(task, dict):
                    continue
                action = task.get("action", "").strip()
                if not action:
                    continue
                all_tasks.append(
                    {
                        "action": action,
                        "subgoal": sg_goal,
                        "depends_on": list(task.get("depends_on", [])),
                        "fallback": (
                            f"Skip '{action}' if blocked — attempt remaining "
                            f"independent tasks and revisit when dependencies resolve"
                        ),
                    }
                )

        if not all_tasks:
            return []

        # --- Topological sort (Kahn's algorithm) -----------------------------
        n = len(all_tasks)
        # Map action string → index (first occurrence wins for duplicate actions)
        action_to_idx: dict = {}
        for i, t in enumerate(all_tasks):
            if t["action"] not in action_to_idx:
                action_to_idx[t["action"]] = i

        # adjacency[i] = list of task indices that must come AFTER i
        adjacency: list = [[] for _ in range(n)]
        in_degree: list = [0] * n

        for i, t in enumerate(all_tasks):
            for dep_action in t["depends_on"]:
                dep_idx = action_to_idx.get(dep_action)
                if dep_idx is not None and dep_idx != i:
                    adjacency[dep_idx].append(i)
                    in_degree[i] += 1

        # BFS queue seeded with zero-in-degree nodes
        queue: deque = deque(i for i in range(n) if in_degree[i] == 0)
        sorted_indices: list = []

        while queue:
            node = queue.popleft()
            sorted_indices.append(node)
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Append any remaining nodes involved in a cycle (preserves them)
        visited = set(sorted_indices)
        for i in range(n):
            if i not in visited:
                sorted_indices.append(i)

        return [
            {
                "task": all_tasks[i]["action"],
                "subgoal": all_tasks[i]["subgoal"],
                "fallback": all_tasks[i]["fallback"],
            }
            for i in sorted_indices
        ]

    # ------------------------------------------------------------------
    # to_json
    # ------------------------------------------------------------------

    def to_json(self) -> dict:
        """
        Return the full serializable representation of this GoalTree, including:
          - root_goal
          - tree (the decomposed structure)
          - pruned_branches
          - execution_order (pre-computed flat task list)
        """
        return {
            "root_goal": self.root_goal,
            "tree": self.tree,
            "pruned_branches": self.pruned_branches,
            "execution_order": self.get_execution_order(),
        }

    # ------------------------------------------------------------------
    # save_to_memory
    # ------------------------------------------------------------------

    def save_to_memory(self) -> None:
        """
        Persist this GoalTree to memory.json.

        Loads memory.json via tools._load_memory(), writes
        memory["active_goal_decomposition"] = self.to_json(), then saves
        back via tools._save_memory_file().

        Uses a lazy import of tools to avoid circular imports at module load.
        """
        # Lazy import — tools imports agent, avoid circular dependency at top level
        from tools import _load_memory, _save_memory_file  # noqa: PLC0415

        memory = _load_memory()
        memory["active_goal_decomposition"] = self.to_json()
        _save_memory_file(memory)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def decompose_and_save(goal: str, context: str = "") -> str:
    """
    Create a GoalTree for *goal*, decompose it (with optional *context*),
    persist the result to memory.json, and return the full tree as a
    JSON-formatted string.

    This is the primary entry point for external callers that want a
    one-shot decompose-and-persist operation.
    """
    tree = GoalTree(goal)
    tree.decompose(context)
    tree.save_to_memory()
    return json.dumps(tree.to_json(), indent=2, ensure_ascii=False)
