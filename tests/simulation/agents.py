"""Stand-in step functions.

A ``step_fn`` answers one question: *given this task, which tool should run?*
That is the model's job in production. These stand-ins answer it with keyword
matching over the task description and the current strategy.

They deliberately contain **no failure handling**. Nothing here inspects an
error, decides to retry, or picks a recovery. If a scenario recovers, the
recovery came from the planner and the recovery engine — which is the whole
point of testing it this way.
"""

import re
from typing import Callable, List, Optional, Tuple

Step = Tuple[str, dict]

# A package name inside a prerequisite task the recovery engine wrote.
_PACKAGE = re.compile(r"package '([\w.\-]+)'")
_COMMAND = re.compile(r"command '([\w.\-]+)'")


def keyword_agent(world, *, artifact: str = "app.py",
                  source: str = "print('hello')",
                  test_file: str = "test_app.py",
                  test_source: str = "print('test ok')",
                  strategy_sources: Optional[dict] = None) -> Callable:
    """Map a task description to a tool call.

    ``strategy_sources`` lets a scenario give different code for different
    strategies, which is how a "first approach can't work" case is set up —
    the agent writes what the current strategy says to write.
    """

    def step(task, context) -> List[Step]:
        description = task.description.lower()
        strategy = (task.strategy or "").lower()
        blob = f"{description} {strategy}"

        # Work the recovery engine inserted: install what was missing.
        if "install" in blob:
            match = _PACKAGE.search(task.description) or _COMMAND.search(task.description)
            if match:
                return [("pip_install", {"package": match.group(1)})]

        if "find or create the missing path" in blob:
            return [("write_file", {"filename": artifact, "content": source})]

        if any(word in blob for word in ("implement", "apply the change",
                                         "develop and apply a fix", "write it",
                                         "build it")):
            content = source
            if strategy_sources:
                for marker, alternative in strategy_sources.items():
                    if marker in strategy:
                        content = alternative
                        break
            return [("write_file", {"filename": artifact, "content": content})]

        if ("run the tests" in blob or "test it" in blob
                or ("test" in blob and "write" in blob)):
            # Write the tests before running them if they don't exist yet.
            steps: List[Step] = []
            if not world.exists(test_file):
                steps.append(("write_file", {"filename": test_file,
                                             "content": test_source}))
            steps.append(("run_tests", {}))
            return steps

        if any(word in blob for word in ("reproduce", "run it", "verify", "inspect",
                                         "locate the failing", "check")):
            return [("run_python", {"filename": artifact})]

        if "search" in blob or "sources" in blob:
            return [("search_web", {"url": "https://docs.example"})]

        return [("think", {"reasoning": task.instruction()[:160]})]

    return step


def messaging_agent(recipient: str, body: str) -> Callable:
    """A step function for a 'text someone' objective."""

    def step(task, context) -> List[Step]:
        description = task.description.lower()
        if "resolve the recipient" in description:
            return [("resolve_contact", {"name": recipient})]
        if description == "send it":
            return [("send_message", {"to": recipient, "body": body})]
        return [("think", {"reasoning": task.instruction()[:160]})]

    return step


def recording_agent(inner: Callable, log: list) -> Callable:
    """Wrap a step function so a scenario can see exactly what it was asked."""

    def step(task, context) -> List[Step]:
        steps = inner(task, context)
        log.append({"task": task.id, "description": task.description,
                    "strategy": task.strategy, "attempt": task.attempts,
                    "steps": [name for name, _ in steps]})
        return steps

    return step
