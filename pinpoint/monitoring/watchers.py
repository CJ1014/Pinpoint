"""Watchers — the things that notice.

Each watcher polls one observable and emits events on *transitions*, not on
every poll: a server that has been down for an hour should produce one
``port_down`` event, not three thousand. State changes are what carry
information.

Every watcher takes an injectable probe, so the polling logic can be tested
without needing a real crashed server to hand.
"""

import os
import socket
import subprocess
import threading
import time
from typing import Callable, Dict, List, Optional

from pinpoint.monitoring import events as E

UP, DOWN, UNKNOWN = "up", "down", "unknown"


class Watcher:
    """Base watcher: poll a probe, emit an event when the state changes."""

    kind = "watcher"

    def __init__(self, name: str, interval: float = 30.0):
        self.name = name
        self.interval = interval
        self.state: str = UNKNOWN
        self.last_polled: float = 0.0
        self.polls: int = 0
        self.active: bool = True

    def probe(self) -> str:
        """Return the current state. Subclasses implement this."""
        return UNKNOWN

    def _transition_events(self, previous: str, current: str) -> List[E.Event]:
        return []

    def poll(self) -> List[E.Event]:
        """Poll once; return any events the transition produced."""
        if not self.active:
            return []
        self.polls += 1
        self.last_polled = time.time()
        try:
            current = self.probe()
        except Exception as exc:
            current = DOWN
            if self.state != DOWN:
                previous, self.state = self.state, current
                return [E.Event(type=self._down_type(), source=self.name,
                                detail=f"probe failed: {exc}", severity=E.WARNING)]
            return []

        if current == self.state:
            return []
        previous, self.state = self.state, current
        return self._transition_events(previous, current)

    def _down_type(self) -> str:
        return E.COMMAND_FAILED

    def describe(self) -> str:
        return f"{self.kind}:{self.name} = {self.state} (polled {self.polls}x)"


class PortWatcher(Watcher):
    """Watches whether something is listening on a TCP port."""

    kind = "port"

    def __init__(self, host: str, port: int, interval: float = 30.0,
                 probe: Optional[Callable[[], str]] = None, timeout: float = 2.0):
        super().__init__(f"{host}:{port}", interval)
        self.host = host
        self.port = int(port)
        self.timeout = timeout
        self._probe = probe

    def probe(self) -> str:
        if self._probe is not None:
            return self._probe()
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout):
                return UP
        except OSError:
            return DOWN

    def _down_type(self) -> str:
        return E.PORT_DOWN

    def _transition_events(self, previous: str, current: str) -> List[E.Event]:
        if current == DOWN:
            return [E.Event(type=E.PORT_DOWN, source=self.name,
                            detail=f"nothing is listening on {self.name}",
                            severity=E.CRITICAL if previous == UP else E.WARNING,
                            data={"host": self.host, "port": self.port})]
        if current == UP and previous == DOWN:
            return [E.Event(type=E.PORT_UP, source=self.name,
                            detail=f"{self.name} is accepting connections again",
                            severity=E.INFO)]
        return []


class ProcessWatcher(Watcher):
    """Watches whether a named process is running."""

    kind = "process"

    def __init__(self, process_name: str, interval: float = 30.0,
                 probe: Optional[Callable[[], str]] = None):
        super().__init__(process_name, interval)
        self._probe = probe

    def probe(self) -> str:
        if self._probe is not None:
            return self._probe()
        try:
            output = subprocess.run(["pgrep", "-f", self.name],
                                    capture_output=True, text=True, timeout=10)
            return UP if output.stdout.strip() else DOWN
        except Exception:
            return UNKNOWN

    def _down_type(self) -> str:
        return E.PROCESS_CRASHED

    def _transition_events(self, previous: str, current: str) -> List[E.Event]:
        if current == DOWN and previous == UP:
            return [E.Event(type=E.PROCESS_CRASHED, source=self.name,
                            detail=f"'{self.name}' is no longer running",
                            severity=E.CRITICAL)]
        if current == UP and previous == DOWN:
            return [E.Event(type=E.PROCESS_STARTED, source=self.name,
                            detail=f"'{self.name}' is running again", severity=E.INFO)]
        return []


class FileWatcher(Watcher):
    """Watches a file for modification."""

    kind = "file"

    def __init__(self, path: str, interval: float = 10.0,
                 probe: Optional[Callable[[], str]] = None):
        super().__init__(path, interval)
        self.path = path
        self._probe = probe

    def probe(self) -> str:
        if self._probe is not None:
            return self._probe()
        try:
            stat = os.stat(self.path)
            return f"{stat.st_mtime}:{stat.st_size}"
        except OSError:
            return "missing"

    def _transition_events(self, previous: str, current: str) -> List[E.Event]:
        if previous == UNKNOWN:
            return []      # the first poll establishes the baseline
        if current == "missing":
            return [E.Event(type=E.FILE_CHANGED, source=self.path,
                            detail=f"{self.path} disappeared", severity=E.WARNING)]
        return [E.Event(type=E.FILE_CHANGED, source=self.path,
                        detail=f"{self.path} changed", severity=E.INFO)]


class HttpWatcher(Watcher):
    """Watches whether a URL responds successfully."""

    kind = "http"

    def __init__(self, url: str, interval: float = 60.0,
                 probe: Optional[Callable[[], str]] = None, timeout: float = 10.0):
        super().__init__(url, interval)
        self.url = url
        self.timeout = timeout
        self._probe = probe

    def probe(self) -> str:
        if self._probe is not None:
            return self._probe()
        try:
            import httpx
            response = httpx.get(self.url, timeout=self.timeout,
                                 follow_redirects=True)
            return UP if response.status_code < 500 else DOWN
        except Exception:
            return DOWN

    def _down_type(self) -> str:
        return E.HTTP_DOWN

    def _transition_events(self, previous: str, current: str) -> List[E.Event]:
        if current == DOWN:
            return [E.Event(type=E.HTTP_DOWN, source=self.url,
                            detail=f"{self.url} is not responding",
                            severity=E.CRITICAL if previous == UP else E.WARNING)]
        if current == UP and previous == DOWN:
            return [E.Event(type=E.HTTP_UP, source=self.url,
                            detail=f"{self.url} is responding again", severity=E.INFO)]
        return []


class CommandWatcher(Watcher):
    """Watches whether a command still exits cleanly — a build or test suite."""

    kind = "command"

    def __init__(self, command: str, interval: float = 300.0,
                 probe: Optional[Callable[[], str]] = None,
                 cwd: str = "", timeout: float = 300.0):
        super().__init__(command, interval)
        self.command = command
        self.cwd = cwd
        self.timeout = timeout
        self._probe = probe

    def probe(self) -> str:
        if self._probe is not None:
            return self._probe()
        try:
            completed = subprocess.run(self.command, shell=True, capture_output=True,
                                       text=True, timeout=self.timeout,
                                       cwd=self.cwd or None)
            return UP if completed.returncode == 0 else DOWN
        except Exception:
            return DOWN

    def _transition_events(self, previous: str, current: str) -> List[E.Event]:
        if current == DOWN:
            return [E.Event(type=E.COMMAND_FAILED, source=self.command,
                            detail=f"`{self.command}` is failing", severity=E.WARNING)]
        if current == UP and previous == DOWN:
            return [E.Event(type=E.COMMAND_RECOVERED, source=self.command,
                            detail=f"`{self.command}` passes again", severity=E.INFO)]
        return []


class Monitor:
    """Runs a set of watchers and feeds what they see into the event pipeline."""

    def __init__(self, pipeline: Optional[E.EventPipeline] = None):
        self.pipeline = pipeline or E.EventPipeline()
        self.watchers: Dict[str, Watcher] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def add(self, watcher: Watcher) -> Watcher:
        self.watchers[f"{watcher.kind}:{watcher.name}"] = watcher
        return watcher

    def remove(self, key: str) -> bool:
        return self.watchers.pop(key, None) is not None

    def poll_once(self) -> List[E.Decision]:
        """Poll every watcher and run what they emit through the pipeline."""
        decisions: List[E.Decision] = []
        for watcher in list(self.watchers.values()):
            for event in watcher.poll():
                decisions.append(self.pipeline.handle(event))
        return decisions

    def _loop(self, interval: float) -> None:
        while not self._stop.wait(interval):
            try:
                self.poll_once()
            except Exception:
                continue      # a watcher failure must not kill the monitor

    def start(self, interval: float = 30.0) -> bool:
        """Begin polling in the background."""
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, args=(interval,),
                                        daemon=True, name="pinpoint-monitor")
        self._thread.start()
        return True

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict:
        return {
            "running": self.running,
            "watchers": [w.describe() for w in self.watchers.values()],
            "pipeline": self.pipeline.stats(),
        }
