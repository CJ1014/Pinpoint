"""Central tool registry.

Every tool declares what it is, what it costs, and how you can tell whether it
actually worked. Two consumers depend on this:

* the **policy engine**, which reads ``level`` / ``reversible`` /
  ``external_side_effect`` / ``costs_money`` to classify an action;
* the **planner**, which reads ``capabilities`` and ``description`` to work out
  which tools could advance a goal, rather than having tool knowledge baked
  into a prompt.

``verification`` names how success is confirmed — the executor uses it so that
"no error string in the output" is never mistaken for success.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from pinpoint.security.permissions import GREEN, RED, YELLOW

# ── Verification methods ──────────────────────────────────────────────────────
V_NONE = "none"                       # nothing external to check (pure reasoning)
V_FILE_EXISTS = "file_exists"         # a file must exist afterwards
V_FILE_ABSENT = "file_absent"         # a file must be gone afterwards
V_EXIT_CODE = "exit_code"             # a process must have exited cleanly
V_PROCESS_RUNNING = "process_running"  # something must still be running
V_PROVIDER_CONFIRMATION = "provider_confirmation"  # external service must confirm
V_STATE_CHANGE = "state_change"       # stored state must differ afterwards
V_CONTENT = "content_match"           # output must contain expected content
V_UNVERIFIABLE = "unverifiable"       # no way to check — never counts as success

# ── Capability tags ───────────────────────────────────────────────────────────
C_REASONING = "reasoning"
C_PLANNING = "planning"
C_FILESYSTEM = "filesystem"
C_EXECUTION = "execution"
C_NETWORK = "network"
C_GUI = "gui"
C_VISION = "vision"
C_AUDIO = "audio"
C_MEMORY = "memory"
C_COMMUNICATION = "communication"
C_SELF_MOD = "self_modification"
C_INSPECTION = "inspection"
C_META = "meta"


@dataclass
class ToolSpec:
    """Everything the system knows about one tool, declaratively."""
    name: str
    description: str = ""
    capabilities: List[str] = field(default_factory=list)
    level: str = YELLOW
    permission: str = ""
    reversible: bool = True
    external_side_effect: bool = False
    costs_money: bool = False
    destructive: bool = False
    timeout: float = 120.0
    verification: str = V_NONE
    requires: List[str] = field(default_factory=list)   # optional runtime deps
    risk: str = "medium"

    def to_dict(self) -> dict:
        return {
            "name": self.name, "description": self.description,
            "capabilities": self.capabilities, "level": self.level,
            "permission": self.permission, "reversible": self.reversible,
            "external_side_effect": self.external_side_effect,
            "costs_money": self.costs_money, "destructive": self.destructive,
            "timeout": self.timeout, "verification": self.verification,
            "requires": self.requires, "risk": self.risk,
        }

    def summary(self) -> str:
        """One line for planner consumption."""
        flags = []
        if not self.reversible:
            flags.append("irreversible")
        if self.external_side_effect:
            flags.append("external")
        if self.costs_money:
            flags.append("costs money")
        suffix = f" [{', '.join(flags)}]" if flags else ""
        return f"{self.name} ({self.level}, {'/'.join(self.capabilities) or 'general'}){suffix}: {self.description}"


_REGISTRY: Dict[str, ToolSpec] = {}

# Alternative names the model reaches for. tools.dispatch resolves these before
# executing, so the policy engine has to resolve them too — otherwise
# ``bash("rm -rf /")`` is classified as an unknown tool rather than as the
# destructive shell command it actually is.
ALIASES: Dict[str, str] = {
    "open_in_browser": "open_html", "open_browser": "open_html",
    "browser_open": "open_html", "open_file": "open_html",
    "launch_browser": "open_html",
    "create_file": "write_file", "save_file": "write_file", "write": "write_file",
    "run_back": "push_back", "refuse": "push_back", "reject": "push_back",
    "search": "search_web", "google": "search_web", "look_up": "search_web",
    "get_url": "fetch_url", "scrape": "fetch_url", "visit": "fetch_url",
    "execute": "run_shell", "bash": "run_shell", "shell": "run_shell",
    "sh": "run_shell", "cmd": "run_shell", "command": "run_shell",
    "execute_command": "run_shell", "run_command": "run_shell",
    "terminal": "run_shell", "system": "run_shell",
    "text": "send_message", "sms": "send_message", "message": "send_message",
    "call": "make_call", "phone": "make_call",
    "email": "send_email", "mail": "send_email",
    "remove_file": "delete_file", "rm": "delete_file", "unlink": "delete_file",
    "install": "pip_install", "pip": "pip_install",
}


def canonical(name: str) -> str:
    """Resolve an alias to the tool that will actually run."""
    return ALIASES.get(name, name)


def register(spec: ToolSpec) -> ToolSpec:
    _REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> Optional[ToolSpec]:
    return _REGISTRY.get(canonical(name))


def require(name: str) -> ToolSpec:
    """Get a spec, or a conservative default for an unregistered tool.

    An unknown tool is treated as YELLOW, irreversible, and *unverifiable* —
    unrecognised capability should never default to "safe", and its effect must
    never default to "confirmed".
    """
    spec = _REGISTRY.get(canonical(name))
    if spec is not None:
        return spec
    return ToolSpec(name=name, description="unregistered tool",
                    level=YELLOW, reversible=False, risk="high",
                    verification=V_UNVERIFIABLE)


def all_specs() -> Dict[str, ToolSpec]:
    return dict(_REGISTRY)


def names() -> List[str]:
    return sorted(_REGISTRY)


def with_capability(capability: str) -> List[ToolSpec]:
    """Tools tagged with a capability — how the planner finds candidates."""
    return [s for s in _REGISTRY.values() if capability in s.capabilities]


def search(text: str) -> List[ToolSpec]:
    """Substring search over names, descriptions, and capability tags."""
    needle = (text or "").lower().strip()
    if not needle:
        return []
    return [s for s in _REGISTRY.values()
            if needle in s.name.lower()
            or needle in s.description.lower()
            or any(needle in c for c in s.capabilities)]


def catalog(capability: str = "", max_tools: int = 0) -> str:
    """A compact catalog string the planner can be shown."""
    specs = with_capability(capability) if capability else list(_REGISTRY.values())
    specs.sort(key=lambda s: s.name)
    if max_tools:
        specs = specs[:max_tools]
    return "\n".join(s.summary() for s in specs)


# ── Built-in registrations ────────────────────────────────────────────────────

def _r(name, description, capabilities, level=YELLOW, permission="", *,
       reversible=True, external=False, money=False, destructive=False,
       timeout=120.0, verification=V_NONE, requires=None, risk="medium"):
    register(ToolSpec(
        name=name, description=description, capabilities=list(capabilities),
        level=level, permission=permission, reversible=reversible,
        external_side_effect=external, costs_money=money, destructive=destructive,
        timeout=timeout, verification=verification, requires=list(requires or []),
        risk=risk,
    ))


def _register_builtins() -> None:
    # ── Reasoning (no side effects at all) ────────────────────────────────────
    _r("think", "Reason about a problem in the open", [C_REASONING], GREEN,
       "reasoning.think", risk="none")
    _r("deep_think", "Multi-pass reasoning over a hard problem", [C_REASONING], GREEN,
       "reasoning.think", timeout=300.0, risk="none")
    _r("brainstorm", "Generate candidate ideas", [C_REASONING], GREEN,
       "reasoning.think", risk="none")
    _r("critique", "Critically evaluate something", [C_REASONING], GREEN,
       "reasoning.think", risk="none")
    _r("decompose", "Break a goal into steps", [C_REASONING, C_PLANNING], GREEN,
       "reasoning.plan", risk="none")
    _r("decompose_goal", "Build a hierarchical goal tree", [C_PLANNING], GREEN,
       "reasoning.plan", risk="none")
    _r("decompose_goal_auto", "Deterministic goal decomposition", [C_PLANNING], GREEN,
       "reasoning.plan", risk="none")
    _r("multi_frame_analysis", "Analyse a decision from five frames", [C_REASONING], GREEN,
       "reasoning.analyze", timeout=180.0, risk="none")
    _r("run_multi_frame_analysis", "Analyse a decision from five frames", [C_REASONING],
       GREEN, "reasoning.analyze", timeout=180.0, risk="none")
    _r("reflect_on_values", "Surface the values behind a decision", [C_REASONING], GREEN,
       "reasoning.values", risk="none")
    _r("check_capability", "Ask whether a task is within capability", [C_META], GREEN,
       "meta.capability", risk="none")
    _r("push_back", "Refuse or renegotiate the current request", [C_META], GREEN,
       "meta.pushback", risk="none")

    # ── Filesystem: reads ─────────────────────────────────────────────────────
    _r("read_file", "Read a file from the project folder", [C_FILESYSTEM], GREEN,
       "filesystem.read", verification=V_CONTENT, risk="none")
    _r("read_anywhere", "Read a file by absolute path", [C_FILESYSTEM], GREEN,
       "filesystem.read", verification=V_CONTENT, risk="low")
    _r("list_files", "List files in the project folder", [C_FILESYSTEM], GREEN,
       "filesystem.read", risk="none")
    _r("read_own_source", "Read PinPoint's own source", [C_FILESYSTEM, C_INSPECTION],
       GREEN, "filesystem.read", risk="none")

    # ── Filesystem: writes ────────────────────────────────────────────────────
    _r("write_file", "Write a file in the project folder", [C_FILESYSTEM], YELLOW,
       "filesystem.write", verification=V_FILE_EXISTS)
    _r("write_anywhere", "Write a file at an absolute path", [C_FILESYSTEM], YELLOW,
       "filesystem.write", verification=V_FILE_EXISTS, risk="high")
    _r("delete_file", "Delete a file", [C_FILESYSTEM], YELLOW, "filesystem.delete",
       reversible=False, destructive=True, verification=V_FILE_ABSENT, risk="high")
    _r("write_test", "Write a test file", [C_FILESYSTEM], YELLOW, "filesystem.write",
       verification=V_FILE_EXISTS, risk="low")

    # ── Execution ─────────────────────────────────────────────────────────────
    _r("run_python", "Execute a Python file", [C_EXECUTION], YELLOW, "execution.python",
       verification=V_EXIT_CODE, timeout=300.0)
    _r("run_shell", "Execute a shell command", [C_EXECUTION], YELLOW, "execution.shell",
       reversible=False, verification=V_EXIT_CODE, timeout=300.0, risk="high")
    _r("run_tests", "Run the test suite", [C_EXECUTION, C_INSPECTION], GREEN,
       "execution.test", verification=V_EXIT_CODE, timeout=600.0, risk="low")
    _r("pip_install", "Install a Python package", [C_EXECUTION], YELLOW,
       "execution.install", reversible=False, verification=V_EXIT_CODE,
       timeout=300.0, risk="high")
    _r("run_gui", "Launch a GUI program", [C_EXECUTION, C_GUI], YELLOW, "execution.gui",
       verification=V_PROCESS_RUNNING, requires=["display"], risk="high")
    _r("start_server", "Start a local HTTP server", [C_EXECUTION, C_NETWORK], YELLOW,
       "execution.server", verification=V_PROCESS_RUNNING)
    _r("check_js", "Syntax-check a JavaScript file", [C_INSPECTION], GREEN,
       "inspection.lint", verification=V_EXIT_CODE, risk="none")
    _r("validate_html", "Validate an HTML file", [C_INSPECTION], GREEN,
       "inspection.lint", risk="none")

    # ── Network ───────────────────────────────────────────────────────────────
    _r("search_web", "Search the web", [C_NETWORK], GREEN, "network.search",
       timeout=60.0, risk="low")
    _r("deep_research", "Read several sources on a topic", [C_NETWORK], GREEN,
       "network.search", timeout=300.0, risk="low")
    _r("fetch_url", "Fetch a URL's text", [C_NETWORK], GREEN, "network.fetch",
       timeout=60.0, risk="low")
    _r("get_news", "Fetch current headlines", [C_NETWORK], GREEN, "network.fetch",
       timeout=60.0, risk="low")
    _r("dictionary_lookup", "Look a word up", [C_NETWORK], GREEN, "network.fetch",
       timeout=30.0, risk="none")

    # ── GUI / computer control ────────────────────────────────────────────────
    _r("open_app", "Open an application", [C_GUI], YELLOW, "computer.launch",
       verification=V_PROCESS_RUNNING, requires=["desktop"], risk="medium")
    _r("open_url", "Open a URL in the browser", [C_GUI, C_NETWORK], YELLOW,
       "computer.launch", requires=["desktop"], risk="low")
    _r("open_html", "Open a local HTML file in the browser", [C_GUI], YELLOW,
       "computer.launch", requires=["desktop"], risk="low")
    _r("type_text", "Type text via the keyboard", [C_GUI], YELLOW, "computer.keyboard",
       reversible=False, requires=["pyautogui"], risk="high")
    _r("press_key", "Press a key or combination", [C_GUI], YELLOW, "computer.keyboard",
       reversible=False, requires=["pyautogui"], risk="high")
    _r("take_screenshot", "Save a screenshot", [C_VISION], GREEN, "computer.screen",
       verification=V_FILE_EXISTS, requires=["screen"], risk="low")
    _r("capture_screen", "Capture the screen to a path", [C_VISION], GREEN,
       "computer.screen", verification=V_FILE_EXISTS, requires=["screen"], risk="low")
    _r("see_screen", "Describe what is on screen", [C_VISION], GREEN, "computer.screen",
       timeout=180.0, requires=["screen", "vision_model"], risk="low")

    # ── Memory / goals ────────────────────────────────────────────────────────
    _r("save_memory", "Store something in memory", [C_MEMORY], GREEN, "memory.write",
       verification=V_STATE_CHANGE, risk="none")
    _r("recall_memories", "Recall stored memories", [C_MEMORY], GREEN, "memory.read",
       risk="none")
    _r("list_memory_categories", "List memory categories", [C_MEMORY], GREEN,
       "memory.read", risk="none")
    _r("update_knowledge", "Append to the durable knowledge file", [C_MEMORY], GREEN,
       "memory.write", verification=V_STATE_CHANGE, risk="none")
    _r("list_goals", "List long-term goals", [C_PLANNING], GREEN, "goals.read",
       risk="none")
    _r("add_long_term_goal", "Add a long-term goal", [C_PLANNING], GREEN, "goals.write",
       verification=V_STATE_CHANGE, risk="none")
    _r("update_goal_progress", "Record progress on a goal", [C_PLANNING], GREEN,
       "goals.write", verification=V_STATE_CHANGE, risk="none")
    _r("complete_goal", "Mark a goal complete", [C_PLANNING], GREEN, "goals.write",
       verification=V_STATE_CHANGE, risk="none")
    _r("abandon_goal", "Abandon a goal", [C_PLANNING], GREEN, "goals.write",
       verification=V_STATE_CHANGE, risk="low")
    _r("set_session_goal", "Set this session's objective", [C_PLANNING], GREEN,
       "goals.write", verification=V_STATE_CHANGE, risk="none")

    # ── Cognition state ───────────────────────────────────────────────────────
    _r("verify_claim", "Check a claim against session reality", [C_META], GREEN,
       "meta.verify", risk="none")
    _r("verify_last_action", "Score the last tool result", [C_META], GREEN,
       "meta.verify", risk="none")
    _r("get_session_reality", "Report what actually happened this session", [C_META],
       GREEN, "meta.reality", risk="none")
    _r("get_previous_sessions", "Report previous sessions", [C_META], GREEN,
       "meta.reality", risk="none")
    _r("create_agi_checkpoint", "Save cross-session project state", [C_META], GREEN,
       "meta.checkpoint", verification=V_FILE_EXISTS, risk="none")
    _r("list_agi_checkpoints", "List saved checkpoints", [C_META], GREEN,
       "meta.checkpoint", risk="none")
    _r("request_human_input", "Ask the human a question", [C_META], GREEN,
       "meta.ask", risk="none")
    _r("done", "Declare the session's work finished", [C_META], GREEN, "meta.done",
       risk="none")

    # ── Self-modification ─────────────────────────────────────────────────────
    _r("modify_own_source", "Edit PinPoint's own source", [C_SELF_MOD, C_FILESYSTEM],
       YELLOW, "self.modify", verification=V_FILE_EXISTS, risk="high")
    _r("list_self_mod_history", "List past self-modifications", [C_INSPECTION], GREEN,
       "self.read", risk="none")

    # ── Reporting / inspection ────────────────────────────────────────────────
    _r("get_system_info", "Report host system information", [C_INSPECTION], GREEN,
       "system.read", risk="none")
    _r("show_dashboard", "Render the performance dashboard", [C_INSPECTION], GREEN,
       "system.read", risk="none")
    _r("review_own_work", "Review work produced this session", [C_INSPECTION], GREEN,
       "system.read", risk="none")
    _r("log_experiment", "Record a structured experiment", [C_MEMORY], GREEN,
       "memory.write", verification=V_STATE_CHANGE, risk="none")
    _r("list_experiments", "List logged experiments", [C_MEMORY], GREEN, "memory.read",
       risk="none")
    _r("get_specialization", "Report the current specialization", [C_META], GREEN,
       "meta.read", risk="none")
    _r("set_specialization", "Set a domain specialization", [C_META], GREEN,
       "meta.write", verification=V_STATE_CHANGE, risk="none")
    _r("collab_status", "Read collaboration status", [C_META], GREEN, "meta.read",
       risk="none")
    _r("collab_update", "Post a collaboration update", [C_META], GREEN, "meta.write",
       risk="low")

    # ── Media / voice ─────────────────────────────────────────────────────────
    _r("speak", "Say something out loud", [C_AUDIO], GREEN, "audio.speak",
       requires=["audio_out"], risk="none")
    _r("mute_voice", "Mute speech output", [C_AUDIO], GREEN, "audio.control",
       risk="none")
    _r("unmute_voice", "Unmute speech output", [C_AUDIO], GREEN, "audio.control",
       risk="none")
    _r("toggle_voice", "Toggle speech output", [C_AUDIO], GREEN, "audio.control",
       risk="none")
    _r("synthesize_audio", "Generate an audio file", [C_AUDIO, C_FILESYSTEM], YELLOW,
       "audio.generate", verification=V_FILE_EXISTS, timeout=300.0, risk="low")
    _r("generate_art", "Generate an image file", [C_FILESYSTEM], YELLOW,
       "media.generate", verification=V_FILE_EXISTS, timeout=300.0, risk="low")
    _r("generate_portfolio", "Generate the portfolio site", [C_FILESYSTEM], YELLOW,
       "media.generate", verification=V_FILE_EXISTS, risk="low")

    # ── Version control ───────────────────────────────────────────────────────
    _r("git_commit", "Commit changes to git", [C_EXECUTION], YELLOW, "vcs.commit",
       verification=V_EXIT_CODE, risk="medium")

    # ── Communication (Stage 8 — external, billable, never auto-approved) ─────
    _r("send_message", "Send a text message through an authorized provider",
       [C_COMMUNICATION], YELLOW, "communication.send", reversible=False,
       external=True, money=True, verification=V_PROVIDER_CONFIRMATION,
       requires=["sms_provider"], risk="high")
    _r("send_email", "Send an email through an authorized provider",
       [C_COMMUNICATION], YELLOW, "communication.email", reversible=False,
       external=True, verification=V_PROVIDER_CONFIRMATION,
       requires=["email_provider"], risk="high")
    _r("make_call", "Place a phone call through an authorized provider",
       [C_COMMUNICATION], YELLOW, "communication.call", reversible=False,
       external=True, money=True, verification=V_PROVIDER_CONFIRMATION,
       requires=["voice_provider"], risk="high")
    _r("get_messages", "Read messages from an authorized provider",
       [C_COMMUNICATION], YELLOW, "communication.read", external=True,
       requires=["sms_provider"], risk="medium")
    _r("get_call_status", "Check the status of a call", [C_COMMUNICATION], GREEN,
       "communication.read", external=True, requires=["voice_provider"], risk="low")
    _r("resolve_contact", "Resolve a name to a contact", [C_COMMUNICATION], GREEN,
       "communication.contacts", risk="low")
    _r("add_contact", "Save a contact CJ has given", [C_COMMUNICATION], YELLOW,
       "communication.contacts", verification=V_STATE_CHANGE, risk="low")
    _r("communication_status", "Report which channels actually work",
       [C_COMMUNICATION, C_INSPECTION], GREEN, "communication.read", risk="none")

    # ── Scheduling ────────────────────────────────────────────────────────────
    _r("schedule_task", "Schedule something for later", [C_PLANNING], GREEN,
       "schedule.write", verification=V_STATE_CHANGE, risk="low")
    _r("list_scheduled", "List scheduled tasks", [C_PLANNING], GREEN,
       "schedule.read", risk="none")
    _r("cancel_scheduled", "Cancel a scheduled task", [C_PLANNING], GREEN,
       "schedule.write", verification=V_STATE_CHANGE, risk="low")

    # ── Monitoring ────────────────────────────────────────────────────────────
    _r("watch", "Watch a port, process, file, URL, or command", [C_INSPECTION],
       GREEN, "monitor.write", verification=V_STATE_CHANGE, risk="low")
    _r("list_watchers", "Show what is being watched", [C_INSPECTION], GREEN,
       "monitor.read", risk="none")
    _r("check_watchers", "Poll every watcher now", [C_INSPECTION], GREEN,
       "monitor.read", risk="none")
    _r("stop_watching", "Stop a watcher", [C_INSPECTION], GREEN, "monitor.write",
       risk="low")

    # ── Self-knowledge ────────────────────────────────────────────────────────
    _r("capability_report", "Report what is actually possible in this environment",
       [C_META, C_INSPECTION], GREEN, "meta.capability", risk="none")
    _r("permission_status", "Report the autonomy profile and standing grants",
       [C_META, C_INSPECTION], GREEN, "meta.read", risk="none")
    _r("emergency_status", "Report whether the emergency stop is engaged",
       [C_META, C_INSPECTION], GREEN, "meta.read", risk="none")


_register_builtins()
