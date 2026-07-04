"""
capability_map.py — PinPoint CapabilityMap Module

Rule-based + memory-learned capability tracking for the PinPoint autonomous agent.
No LLM calls are made in this module. Confidence scores are updated via recorded
outcomes and persisted to memory between sessions.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default confidence scores per domain
# ---------------------------------------------------------------------------

DEFAULT_DOMAINS: dict[str, float] = {
    "code_writing": 0.95,
    "file_operations": 0.92,
    "web_research": 0.85,
    "reasoning": 0.88,
    "html_css": 0.90,
    "data_analysis": 0.75,
    "user_emotions": 0.20,
    "real_world_events": 0.15,
    "current_time": 0.10,
    "hardware_control": 0.30,
    "email_access": 0.05,
    "camera_access": 0.05,
}

# ---------------------------------------------------------------------------
# Hard boundaries — always False regardless of confidence score
# ---------------------------------------------------------------------------

HARD_BOUNDARIES: dict[str, bool] = {
    "no_camera_access": True,
    "no_microphone_raw": True,
    "no_email_access": True,
    "no_ground_truth_verification": True,
    "no_future_prediction": True,
}

# ---------------------------------------------------------------------------
# Keyword mappings: domain -> list of trigger keywords/phrases
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "code_writing": [
        "write code", "code", "program", "script", "function", "class",
        "implement", "debug", "fix bug", "refactor", "python", "javascript",
        "typescript", "java", "c++", "c#", "rust", "go", "ruby", "php",
        "algorithm", "module", "library", "api", "snippet", "coding",
    ],
    "file_operations": [
        "file", "folder", "directory", "read file", "write file", "delete file",
        "move file", "copy file", "rename file", "list files", "path", "disk",
        "save", "load", "open file", "create file", "mkdir", "filesystem",
    ],
    "web_research": [
        "search", "web", "internet", "browse", "look up", "find online",
        "research", "google", "website", "url", "http", "online", "fetch",
        "download", "scrape", "crawl", "query web",
    ],
    "reasoning": [
        "reason", "think", "analyze", "conclude", "infer", "logic", "deduce",
        "evaluate", "compare", "explain", "summarize", "plan", "decide",
        "solve", "problem", "strategy", "assess", "judge", "interpret",
    ],
    "html_css": [
        "html", "css", "webpage", "web page", "style", "layout", "design",
        "frontend", "front-end", "dom", "element", "class name", "selector",
        "responsive", "flexbox", "grid", "bootstrap", "tailwind", "sass",
        "stylesheet", "template",
    ],
    "data_analysis": [
        "data", "analyze data", "dataset", "csv", "excel", "spreadsheet",
        "statistics", "chart", "graph", "plot", "pandas", "numpy", "table",
        "metrics", "aggregate", "filter", "sort data", "visualize", "report",
        "trend", "average", "mean", "median", "correlation",
    ],
    "user_emotions": [
        "emotion", "feeling", "mood", "sad", "happy", "angry", "anxious",
        "stress", "mental health", "empathy", "therapy", "sentiment",
        "emotional support", "how are you feeling", "upset", "depressed",
        "worried", "nervous",
    ],
    "real_world_events": [
        "news", "event", "happening", "current events", "today", "yesterday",
        "latest", "recent", "breaking", "world news", "politics", "sports score",
        "stock price", "weather", "live", "real time", "real-time",
    ],
    "current_time": [
        "time", "clock", "date", "what time", "current time", "now",
        "timestamp", "timezone", "hour", "minute", "second", "today's date",
        "what day", "calendar",
    ],
    "hardware_control": [
        "hardware", "cpu", "gpu", "ram", "memory usage", "disk space",
        "process", "system", "reboot", "shutdown", "driver", "device",
        "peripheral", "usb", "monitor", "screen brightness", "volume",
        "keyboard", "mouse",
    ],
    "email_access": [
        "email", "mail", "inbox", "send email", "read email", "gmail",
        "outlook", "smtp", "imap", "compose", "reply email", "attachment",
        "email account", "mailbox",
    ],
    "camera_access": [
        "camera", "webcam", "photo", "picture", "snapshot", "capture image",
        "record video", "video feed", "live feed", "screenshot webcam",
        "face detection", "opencv camera",
    ],
}

# Boundary keyword triggers: boundary key -> list of trigger keywords/phrases
_BOUNDARY_KEYWORDS: dict[str, list[str]] = {
    "no_camera_access": [
        "camera", "webcam", "photo", "snapshot", "capture image",
        "record video", "video feed", "live feed", "opencv camera",
    ],
    "no_microphone_raw": [
        "microphone", "mic", "audio input", "record audio", "speech input",
        "voice input", "listen", "hear", "audio capture", "raw audio",
    ],
    "no_email_access": [
        "email", "mail", "inbox", "send email", "read email", "gmail",
        "outlook", "smtp", "imap", "compose", "reply email", "mailbox",
        "email account",
    ],
    "no_ground_truth_verification": [
        "verify fact", "ground truth", "confirm reality", "validate real world",
        "check if true", "fact check", "is it true", "verify news",
        "authenticate", "confirm event happened",
    ],
    "no_future_prediction": [
        "predict future", "predict the future", "what will happen", "forecast",
        "prophecy", "will it", "future event", "predict outcome", "next week will",
        "stock prediction", "future prediction", "price prediction",
    ],
}

# Human-readable descriptions for hard boundary reasons
_BOUNDARY_REASONS: dict[str, str] = {
    "no_camera_access": (
        "PinPoint does not have direct camera or webcam access. "
        "Raw camera capture is outside its operational boundaries."
    ),
    "no_microphone_raw": (
        "PinPoint does not have raw microphone or audio input access. "
        "Live audio capture is outside its operational boundaries."
    ),
    "no_email_access": (
        "PinPoint does not have access to email accounts or mailboxes. "
        "Reading or sending emails is outside its operational boundaries."
    ),
    "no_ground_truth_verification": (
        "PinPoint cannot verify real-world facts or ground truth with certainty. "
        "It has no authoritative external verification channel."
    ),
    "no_future_prediction": (
        "PinPoint cannot predict future events. "
        "Forecasting real-world outcomes is outside its operational boundaries."
    ),
}

# Workaround suggestions per domain or boundary
_WORKAROUNDS: dict[str, str] = {
    "camera_access": (
        "I cannot access the camera directly. "
        "You could capture an image using your OS tools and share the file path, "
        "then I can analyze or process that image file."
    ),
    "no_camera_access": (
        "I cannot access the camera directly. "
        "You could capture an image using your OS tools and share the file path, "
        "then I can analyze or process that image file."
    ),
    "no_microphone_raw": (
        "I cannot capture raw audio. "
        "You could record audio with a system tool and provide the audio file, "
        "or use a speech-to-text utility and paste the transcript for me to process."
    ),
    "email_access": (
        "I cannot access email accounts directly. "
        "You could copy and paste the email text to me, "
        "or use a local email client and share exported content."
    ),
    "no_email_access": (
        "I cannot access email accounts directly. "
        "You could copy and paste the email text to me, "
        "or use a local email client and share exported content."
    ),
    "no_ground_truth_verification": (
        "I cannot verify real-world facts with certainty. "
        "I can reason from information you provide, "
        "or search the web if web research tools are available."
    ),
    "no_future_prediction": (
        "I cannot predict future events. "
        "I can help you analyze trends, model scenarios, "
        "or reason about probabilities based on current data."
    ),
    "user_emotions": (
        "Emotional understanding is a lower-confidence area for me. "
        "I can offer logical support, resources, or help you articulate your thoughts, "
        "but I recommend speaking with a qualified human professional for emotional matters."
    ),
    "real_world_events": (
        "I may not have current information about real-world events. "
        "I can help search the web for recent news or reason from context you provide."
    ),
    "current_time": (
        "I do not have a real-time clock. "
        "You can tell me the current date/time and I will use it, "
        "or I can read system time via a script if file operations are available."
    ),
    "hardware_control": (
        "Direct hardware control is limited. "
        "I can write scripts (PowerShell, Python) that you run locally "
        "to interact with hardware or system resources."
    ),
    "code_writing": (
        "Code writing is a strong capability. "
        "Please describe what you need and I will write it."
    ),
    "file_operations": (
        "File operations are well-supported. "
        "I can read, write, move, copy, and manage files on your local system."
    ),
    "web_research": (
        "I can perform web research if search tools are available in this session."
    ),
    "reasoning": (
        "Reasoning and analysis are core capabilities. "
        "Please provide the context or problem and I will reason through it."
    ),
    "html_css": (
        "HTML/CSS work is well-supported. "
        "I can write, edit, and debug web markup and stylesheets."
    ),
    "data_analysis": (
        "Data analysis is supported at a moderate confidence level. "
        "Provide the dataset or describe the analysis and I will help."
    ),
    "unknown": (
        "I am not sure which of my capabilities apply here. "
        "Please describe the task in more detail and I will assess what I can do."
    ),
}


def _task_lower(task: str) -> str:
    """Return lowercased, stripped version of task string."""
    return task.strip().lower()


def _match_domain(task_l: str) -> Optional[str]:
    """
    Return the best-matching domain name for a task string, or None if no match.
    Matching counts keyword hits per domain and returns the highest-scoring one.
    Minimum 1 hit is required for a match.
    """
    scores: dict[str, int] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in task_l)
        if count > 0:
            scores[domain] = count
    if not scores:
        return None
    return max(scores, key=lambda d: scores[d])


def _match_boundary(task_l: str) -> Optional[str]:
    """
    Return the first hard-boundary key that matches the task, or None.
    Boundaries are checked in definition order.
    """
    for boundary, keywords in _BOUNDARY_KEYWORDS.items():
        for kw in keywords:
            if kw in task_l:
                return boundary
    return None


class CapabilityMap:
    """
    Tracks PinPoint's domain-level capability confidence scores.
    Scores are loaded from persistent memory on init and saved after updates.
    """

    def __init__(self) -> None:
        self.domains: dict[str, float] = {}
        self.load()

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------

    def can_do(self, task: str) -> tuple[bool, float, str]:
        """
        Assess whether PinPoint can perform the described task.

        Returns:
            (can_do: bool, confidence: float, reason: str)

        Hard boundaries are checked first and always return (False, 0.0, reason).
        Domain confidence below 0.40 returns False with the actual confidence score.
        """
        task_l = _task_lower(task)

        # 1. Hard boundary check — takes absolute precedence
        boundary = _match_boundary(task_l)
        if boundary is not None:
            reason = _BOUNDARY_REASONS.get(
                boundary,
                f"This capability is permanently outside PinPoint's boundaries ({boundary}).",
            )
            return (False, 0.0, reason)

        # 2. Domain keyword match
        domain = _match_domain(task_l)
        if domain is None:
            return (
                True,
                0.50,
                "No specific domain matched. Attempting with general reasoning capability.",
            )

        confidence = self.domains.get(domain, DEFAULT_DOMAINS.get(domain, 0.50))

        # Threshold: below 0.40 we report inability
        if confidence < 0.40:
            reason = (
                f"Confidence in domain '{domain}' is low ({confidence:.2f}). "
                "This task is likely outside reliable operational bounds."
            )
            return (False, confidence, reason)

        reason = f"Task maps to domain '{domain}' with confidence {confidence:.2f}."
        return (True, confidence, reason)

    def record_failure(self, task: str, domain: str, reason: str) -> None:
        """
        Record a task failure for a domain, reducing its confidence by 0.05.
        Confidence floor is 0.05. Persists to memory.
        """
        current = self.domains.get(domain, DEFAULT_DOMAINS.get(domain, 0.50))
        updated = max(0.05, round(current - 0.05, 4))
        self.domains[domain] = updated
        self.save()

    def record_success(self, task: str, domain: str) -> None:
        """
        Record a task success for a domain, increasing its confidence by 0.02.
        Confidence ceiling is 0.99. Persists to memory.
        """
        current = self.domains.get(domain, DEFAULT_DOMAINS.get(domain, 0.50))
        updated = min(0.99, round(current + 0.02, 4))
        self.domains[domain] = updated
        self.save()

    def suggest_workaround(self, task: str) -> str:
        """
        Suggest an alternative approach for a task that may be outside capability.
        Checks boundary triggers first, then domain match, then falls back to generic.
        """
        task_l = _task_lower(task)

        boundary = _match_boundary(task_l)
        if boundary is not None and boundary in _WORKAROUNDS:
            return _WORKAROUNDS[boundary]

        domain = _match_domain(task_l)
        if domain is not None and domain in _WORKAROUNDS:
            return _WORKAROUNDS[domain]

        return _WORKAROUNDS["unknown"]

    def get_all(self) -> dict[str, float]:
        """Return a copy of all domains with their current confidence scores."""
        return dict(self.domains)

    def save(self) -> None:
        """Persist the capability map to memory['capability_map']."""
        try:
            from tools import _load_memory, _save_memory_file  # type: ignore

            memory = _load_memory()
            memory["capability_map"] = self.domains
            _save_memory_file(memory)
        except Exception as exc:
            logger.warning("CapabilityMap.save failed: %s", exc)

    def load(self) -> None:
        """
        Load capability map from memory['capability_map'].
        Falls back to DEFAULT_DOMAINS if memory is unavailable or empty.
        """
        loaded = False
        try:
            from tools import _load_memory  # type: ignore

            memory = _load_memory()
            stored = memory.get("capability_map")
            if isinstance(stored, dict) and stored:
                # Merge with defaults so newly added domains are always present
                merged = dict(DEFAULT_DOMAINS)
                merged.update({k: float(v) for k, v in stored.items()})
                self.domains = merged
                loaded = True
        except Exception as exc:
            logger.debug("CapabilityMap.load fell back to defaults: %s", exc)

        if not loaded:
            self.domains = dict(DEFAULT_DOMAINS)


# ---------------------------------------------------------------------------
# Module-level singleton and convenience functions
# ---------------------------------------------------------------------------

_map = CapabilityMap()


def can_do(task: str) -> tuple[bool, float, str]:
    """
    Module-level convenience: assess whether PinPoint can perform a task.

    Returns:
        (can_do: bool, confidence: float, reason: str)
    """
    return _map.can_do(task)


def check(task: str) -> str:
    """
    Module-level convenience: full capability check returned as a JSON string.

    Returns a JSON object with keys:
        can_do      — bool
        confidence  — float (rounded to 4 decimal places)
        reason      — str
        workaround  — str (populated when task is not doable or confidence < 0.40)
    """
    able, confidence, reason = _map.can_do(task)
    workaround = ""
    if not able or confidence < 0.40:
        workaround = _map.suggest_workaround(task)

    result = {
        "can_do": able,
        "confidence": round(confidence, 4),
        "reason": reason,
        "workaround": workaround,
    }
    return json.dumps(result, ensure_ascii=False)
