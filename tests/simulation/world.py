"""A deterministic fake world.

Two decisions make this worth having:

**The filesystem and sockets are real.** Files land in a real temp directory and
servers bind real localhost ports, because the verification layer stats real
files and opens real sockets. A purely in-memory fake would let verification
pass on evidence that doesn't exist, which would test nothing.

**Failures are injected at the tool boundary, not the truth boundary.** A
``silent`` failure makes ``write_file`` return its usual cheerful success string
*without* creating the file. The world does not tell the agent it failed — the
agent has to find out by checking. That is the property under test.

Scripts have modelled semantics: importing a module that isn't installed
produces a real ``ModuleNotFoundError`` traceback with exit code 1, so the
agent has to diagnose and install it rather than being handed the answer.
"""

import os
import re
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# Failure kinds.
ERROR = "error"        # the tool reports a failure
SILENT = "silent"      # the tool reports success but nothing happens
TIMEOUT = "timeout"    # the tool hangs past its deadline
PARTIAL = "partial"    # the tool half-does the job and says it succeeded
STALE = "stale"        # the effect happens, then is undone behind the agent's back


@dataclass
class Failure:
    """What goes wrong, and how often."""
    kind: str
    message: str = ""
    times: int = 1                 # how many matching calls it affects
    delay: float = 0.5             # for TIMEOUT
    used: int = 0

    @property
    def spent(self) -> bool:
        return self.used >= self.times


@dataclass
class Rule:
    """Inject a failure into calls matching a tool and optional parameter text."""
    tool: str
    failure: Failure
    contains: str = ""             # only match calls whose params mention this

    def matches(self, tool: str, params: dict) -> bool:
        if self.failure.spent or tool != self.tool:
            return False
        if self.contains:
            blob = " ".join(str(v) for v in (params or {}).values())
            return self.contains in blob
        return True


@dataclass
class ToolCall:
    """One recorded interaction with the world."""
    tool: str
    params: Dict[str, Any]
    result: str
    injected: str = ""
    at: float = 0.0


class SimMessageProvider:
    """A messaging/voice provider with configurable honesty."""

    name = "sim-provider"

    def __init__(self, world: "FakeWorld", kind: str = "sms",
                 confirm: bool = True, configured: bool = True,
                 error: str = ""):
        self.world = world
        self.kind = kind
        self.confirm = confirm
        self.configured = configured
        self.error = error
        self.sent: List[dict] = []

    def available(self) -> bool:
        return self.configured

    def missing_config(self) -> str:
        return "sim provider has no credentials configured"

    def _result(self, to: str, payload: dict):
        from pinpoint.communication.providers import base

        self.sent.append({"to": to, **payload})
        self.world.messages.append({"kind": self.kind, "to": to, **payload})
        if self.error:
            return base.SendResult(kind=self.kind, provider=self.name, to=to,
                                   error=self.error)
        if not self.confirm:
            # The provider accepted the request but returned no identifier.
            return base.SendResult(ok=True, kind=self.kind, provider=self.name,
                                   to=to, confirmation_id="", status="unknown")
        identifier = f"SM{len(self.sent):08d}"
        return base.SendResult(ok=True, kind=self.kind, provider=self.name, to=to,
                               confirmation_id=identifier, status="queued")

    def send_message(self, to, body):
        return self._result(to, {"body": body})

    def send_email(self, to, subject, body):
        return self._result(to, {"subject": subject, "body": body})

    def make_call(self, to, script):
        return self._result(to, {"script": script})

    def get_messages(self, limit=10):
        return self.world.messages[-limit:]

    def get_call_status(self, call_id):
        return {"id": call_id, "status": "completed", "duration": "31"}


class FakeWorld:
    """The simulated environment the agent acts on."""

    def __init__(self, root: str):
        self.root = str(root)
        os.makedirs(self.root, exist_ok=True)
        self.installed_modules = {"os", "sys", "json", "time", "pathlib", "re"}
        self.rules: List[Rule] = []
        self.log: List[ToolCall] = []
        self.messages: List[dict] = []
        self.http: Dict[str, tuple] = {}
        self.processes: Dict[str, bool] = {}
        self._sockets: Dict[int, socket.socket] = {}
        self.clock_started = time.time()

    # ── configuration ────────────────────────────────────────────────────────

    def inject(self, tool: str, kind: str, message: str = "", times: int = 1,
               contains: str = "", delay: float = 0.5) -> Rule:
        rule = Rule(tool=tool, contains=contains,
                    failure=Failure(kind=kind, message=message, times=times,
                                    delay=delay))
        self.rules.append(rule)
        return rule

    def install(self, *modules: str) -> None:
        self.installed_modules.update(modules)

    def serve(self, url: str, status: int = 200, body: str = "ok") -> None:
        self.http[url] = (status, body)

    def start_process(self, name: str) -> None:
        self.processes[name] = True

    def crash_process(self, name: str) -> None:
        self.processes[name] = False

    def open_port(self, port: int = 0) -> int:
        """Bind a real listening socket so port verification is genuine."""
        server = socket.socket()
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", port))
        server.listen(1)
        bound = server.getsockname()[1]
        self._sockets[bound] = server
        return bound

    def close_port(self, port: int) -> None:
        server = self._sockets.pop(port, None)
        if server is not None:
            server.close()

    def close(self) -> None:
        for port in list(self._sockets):
            self.close_port(port)

    def vanish(self, relative: str) -> None:
        """Delete a file behind the agent's back — stale-state simulation."""
        try:
            os.unlink(self.path(relative))
        except OSError:
            pass

    # ── filesystem helpers ───────────────────────────────────────────────────

    def path(self, name: str) -> str:
        if os.path.isabs(name):
            return name
        return os.path.join(self.root, name)

    def exists(self, name: str) -> bool:
        return os.path.isfile(self.path(name))

    def read(self, name: str) -> str:
        with open(self.path(name), "r", encoding="utf-8") as handle:
            return handle.read()

    def write(self, name: str, content: str) -> None:
        target = self.path(name)
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(content)

    # ── the dispatch the agent talks to ──────────────────────────────────────

    def dispatch(self, tool: str, params: Optional[dict] = None) -> str:
        params = dict(params or {})
        rule = next((r for r in self.rules if r.matches(tool, params)), None)
        injected = ""

        if rule is not None:
            rule.failure.used += 1
            injected = rule.failure.kind
            if rule.failure.kind == TIMEOUT:
                time.sleep(rule.failure.delay)
            elif rule.failure.kind == ERROR:
                result = rule.failure.message or f"Error: {tool} failed"
                self._record(tool, params, result, injected)
                return result
            elif rule.failure.kind == SILENT:
                # The tool claims success. Nothing actually happens. The agent
                # is expected to catch this by checking, not by being told.
                result = self._plausible_success(tool, params)
                self._record(tool, params, result, injected)
                return result

        handler = getattr(self, f"_tool_{tool}", None)
        if handler is None:
            result = f"Unknown tool: {tool}"
        else:
            result = handler(params, rule)
        self._record(tool, params, result, injected)
        return result

    def _record(self, tool, params, result, injected) -> None:
        self.log.append(ToolCall(tool=tool, params=params, result=result,
                                 injected=injected, at=time.time()))

    def _plausible_success(self, tool: str, params: dict) -> str:
        """The success string a tool would emit — used for silent failures."""
        if tool in ("write_file", "write_anywhere", "write_test"):
            target = params.get("filename") or params.get("path", "out.txt")
            content = params.get("content", params.get("test_code", ""))
            return f"Written {len(content)} chars to {self.path(target)}"
        if tool == "delete_file":
            return f"Deleted {params.get('path', '')}"
        if tool in ("run_shell", "run_python"):
            return "stdout:\ncompleted\nexit code: 0"
        if tool == "run_tests":
            return "All tests passed."
        if tool == "send_message":
            return "Message sent successfully!"
        if tool == "pip_install":
            return f"Successfully installed {params.get('package', '')}"
        if tool == "start_server":
            return f"Server started on port {params.get('port', 8080)}"
        return "Done."

    # ── modelled tools ───────────────────────────────────────────────────────

    def _tool_write_file(self, params, rule):
        target = params.get("filename") or params.get("path", "")
        content = params.get("content", "")
        if not target:
            return "Error: write_file needs a filename"
        if rule is not None and rule.failure.kind == PARTIAL:
            content = content[: max(1, len(content) // 2)]
        self.write(target, content)
        if rule is not None and rule.failure.kind == STALE:
            self.vanish(target)      # written, then removed behind the agent's back
        return f"Written {len(content)} chars to {self.path(target)}"

    _tool_write_anywhere = _tool_write_file

    def _tool_write_test(self, params, rule):
        return self._tool_write_file(
            {"filename": params.get("filename", ""),
             "content": params.get("test_code", params.get("content", ""))}, rule)

    def _tool_read_file(self, params, rule):
        target = params.get("filename") or params.get("path", "")
        if not self.exists(target):
            return f"File not found: {self.path(target)}"
        return self.read(target)

    _tool_read_anywhere = _tool_read_file

    def _tool_delete_file(self, params, rule):
        target = params.get("path") or params.get("filename", "")
        if not self.exists(target):
            return f"Error: no such file: {self.path(target)}"
        os.unlink(self.path(target))
        return f"Deleted {self.path(target)}"

    def _tool_list_files(self, params, rule):
        names = sorted(os.listdir(self.root)) if os.path.isdir(self.root) else []
        return "\n".join(names) if names else "(empty)"

    def _tool_pip_install(self, params, rule):
        package = params.get("package", "").strip()
        if not package:
            return "Error: pip_install needs a package"
        self.install(package)
        return f"Successfully installed {package}\nexit code: 0"

    def _run_script(self, name: str) -> tuple:
        """Model running a Python file. Returns ``(exit_code, stdout, stderr)``."""
        if not self.exists(name):
            return 2, "", f"python: can't open file '{self.path(name)}': No such file"
        source = self.read(name)

        for match in re.finditer(r"^\s*import\s+([\w.]+)|^\s*from\s+([\w.]+)\s+import",
                                 source, re.MULTILINE):
            module = (match.group(1) or match.group(2) or "").split(".")[0]
            if module and module not in self.installed_modules:
                return 1, "", (f"Traceback (most recent call last):\n"
                               f"  File \"{self.path(name)}\", line 1, in <module>\n"
                               f"ModuleNotFoundError: No module named '{module}'")
        if "SYNTAX_ERROR" in source:
            return 1, "", (f"  File \"{self.path(name)}\", line 1\n"
                           f"SyntaxError: invalid syntax")
        if "assert False" in source or "RAISE" in source:
            return 1, "", ("Traceback (most recent call last):\n"
                           "AssertionError: the implementation is wrong")

        printed = re.findall(r"print\((['\"])(.*?)\1\)", source)
        stdout = "\n".join(text for _, text in printed)
        return 0, stdout, ""

    def _tool_run_python(self, params, rule):
        name = params.get("filename", "")
        code, out, err = self._run_script(name)
        parts = []
        if out:
            parts.append(f"stdout:\n{out}")
        if err:
            parts.append(f"stderr:\n{err}")
        parts.append(f"exit code: {code}")
        return "\n".join(parts)

    def _tool_run_shell(self, params, rule):
        command = (params.get("command") or "").strip()
        if not command:
            return "Error: run_shell needs a command"

        match = re.match(r"^(?:python3?|py)\s+(\S+)", command)
        if match:
            return self._tool_run_python({"filename": match.group(1)}, rule)

        match = re.match(r"^pip3?\s+install\s+(\S+)", command)
        if match:
            return self._tool_pip_install({"package": match.group(1)}, rule)

        if command.startswith("ls"):
            return f"stdout:\n{self._tool_list_files({}, rule)}\nexit code: 0"

        match = re.match(r"^cat\s+(\S+)", command)
        if match:
            body = (self.read(match.group(1)) if self.exists(match.group(1))
                    else "")
            if not body:
                return (f"stderr:\ncat: {match.group(1)}: No such file or directory\n"
                        f"exit code: 1")
            return f"stdout:\n{body}\nexit code: 0"

        match = re.match(r"^mkdir\s+(?:-p\s+)?(\S+)", command)
        if match:
            os.makedirs(self.path(match.group(1)), exist_ok=True)
            return "exit code: 0"

        if command.startswith("pytest") or command.startswith("python -m pytest"):
            return self._tool_run_tests({}, rule)

        binary = command.split()[0]
        return (f"stderr:\n{binary}: command not found\nexit code: 127")

    def _tool_run_tests(self, params, rule):
        """Run every ``test_*.py`` in the world through the script model."""
        tests = [n for n in sorted(os.listdir(self.root))
                 if n.startswith("test_") and n.endswith(".py")]
        if not tests:
            return "stderr:\nno tests found\nexit code: 5"

        failures = []
        for name in tests:
            code, _, err = self._run_script(name)
            if code != 0:
                failures.append(f"{name}: {err.splitlines()[-1] if err else 'failed'}")

        if failures:
            return ("stdout:\n" + "\n".join(failures)
                    + f"\n{len(failures)} failed, {len(tests) - len(failures)} passed"
                    + "\nexit code: 1")
        return (f"stdout:\n{len(tests)} passed\nexit code: 0")

    def _tool_start_server(self, params, rule):
        port = int(params.get("port", 0) or 0)
        bound = self.open_port(port)
        return f"Server started on port {bound}\nexit code: 0"

    def _tool_fetch_url(self, params, rule):
        url = params.get("url", "")
        if url not in self.http:
            return f"Error: 404 not found: {url}"
        status, body = self.http[url]
        if status >= 400:
            return f"Error: HTTP {status} from {url}"
        return body

    _tool_search_web = _tool_fetch_url

    def _tool_think(self, params, rule):
        return f"Thought logged: {params.get('reasoning', '')[:200]}"

    _tool_brainstorm = _tool_think
    _tool_critique = _tool_think
    _tool_decompose = _tool_think

    def _tool_set_session_goal(self, params, rule):
        return f"Goal set: {params.get('goal', '')}"

    def _tool_done(self, params, rule):
        return f"Session complete: {params.get('summary', '')}"

    # ── communication, routed through the real manager ───────────────────────

    def install_providers(self, confirm: bool = True, configured: bool = True,
                          error: str = "") -> Dict[str, SimMessageProvider]:
        """Register simulated providers with the real communication manager."""
        from pinpoint.communication import manager as CM
        from pinpoint.communication.providers import base

        providers = {}
        for kind in (base.SMS, base.EMAIL, base.VOICE):
            provider = SimMessageProvider(self, kind=kind, confirm=confirm,
                                          configured=configured, error=error)
            CM.set_provider(kind, provider)
            providers[kind] = provider
        return providers

    def _tool_send_message(self, params, rule):
        from pinpoint import toolapi
        return toolapi.send_message(params.get("to", ""), params.get("body", ""))

    def _tool_send_email(self, params, rule):
        from pinpoint import toolapi
        return toolapi.send_email(params.get("to", ""), params.get("subject", ""),
                                  params.get("body", ""))

    def _tool_make_call(self, params, rule):
        from pinpoint import toolapi
        return toolapi.make_call(params.get("to", ""), params.get("purpose", ""),
                                 params.get("script", ""))

    def _tool_resolve_contact(self, params, rule):
        from pinpoint import toolapi
        return toolapi.resolve_contact(params.get("name", ""))

    # ── inspection ───────────────────────────────────────────────────────────

    def calls(self, tool: str = "") -> List[ToolCall]:
        return [c for c in self.log if not tool or c.tool == tool]

    def call_count(self, tool: str = "") -> int:
        return len(self.calls(tool))

    def summary(self) -> str:
        counts: Dict[str, int] = {}
        for call in self.log:
            counts[call.tool] = counts.get(call.tool, 0) + 1
        return ", ".join(f"{tool}×{n}" for tool, n in sorted(counts.items()))
