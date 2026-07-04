"""
reasoning_frames.py — MultiFrameReasoning module for PinPoint.

Analyzes a problem through five cognitive lenses (technical, economic,
temporal, social, creative) in parallel, detects cross-frame conflicts,
and synthesizes a unified recommendation.

Lazy imports from `agent` (inside methods) to avoid circular dependencies.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Any

# ── Constants ────────────────────────────────────────────────────────────────

FRAME_TYPES: list[str] = ["technical", "economic", "temporal", "social", "creative"]


# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class FrameAnalysis:
    frame_type: str
    analysis: str
    confidence: float


# ── Analyzer class ───────────────────────────────────────────────────────────

class MultiFrameAnalyzer:
    """Runs multi-frame reasoning across five cognitive lenses."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, problem: str) -> dict:
        """Analyze *problem* through all five frames concurrently.

        Returns a dict with keys:
            problem   - the original problem string
            frames    - {frame_type: {analysis, confidence}, ...}
            conflicts - list of {frame_a, frame_b, conflict, resolution}
            synthesis - one-paragraph recommendation
        """
        # 1. Parallel frame analysis
        frames_raw: dict[str, FrameAnalysis] = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_frame = {
                executor.submit(self._analyze_frame, problem, frame): frame
                for frame in FRAME_TYPES
            }
            for future in as_completed(future_to_frame):
                frame_name = future_to_frame[future]
                try:
                    result = future.result()
                    frames_raw[frame_name] = result
                except Exception as exc:
                    # Fallback on individual frame failure — keep the run alive
                    frames_raw[frame_name] = FrameAnalysis(
                        frame_type=frame_name,
                        analysis=f"Analysis unavailable: {exc}",
                        confidence=0.0,
                    )

        # Serialize frames to a plain dict for downstream steps
        frames_dict: dict[str, dict] = {
            ft: {"analysis": fa.analysis, "confidence": fa.confidence}
            for ft, fa in frames_raw.items()
        }

        # 2. Detect cross-frame conflicts
        conflicts = self._detect_conflicts(frames_dict)

        # 3. Synthesize a single recommendation
        synthesis = self._synthesize(problem, frames_dict, conflicts)

        return {
            "problem": problem,
            "frames": frames_dict,
            "conflicts": conflicts,
            "synthesis": synthesis,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyze_frame(self, problem: str, frame_type: str) -> FrameAnalysis:
        """Call the LLM once for a single frame perspective.

        Asks for 2-3 sentences.  Parses a trailing confidence value if
        the model includes one (e.g. "Confidence: 0.85"), otherwise falls
        back to 0.7.
        """
        from agent import AGENT_MODEL, get_llm_client, chat_completion  # lazy import

        prompt = (
            f"Analyze this problem from the {frame_type} perspective in 2-3 sentences.\n\n"
            f"Problem: {problem}\n\n"
            "At the very end of your response, on its own line, optionally include:\n"
            "Confidence: <float between 0.0 and 1.0>\n"
            "If you omit it, a default of 0.7 will be used."
        )

        client = get_llm_client(timeout=60.0)
        response = chat_completion(
            client,
            messages=[{"role": "user", "content": prompt}],
            model=AGENT_MODEL,
            max_tokens=300,
            temperature=0.7,
        )
        raw_text: str = response.choices[0].message.content or ""

        # Extract optional confidence line
        confidence = 0.7
        lines = raw_text.strip().splitlines()
        cleaned_lines = []
        for line in lines:
            m = re.match(r"(?i)confidence\s*[:\-]\s*([0-9.]+)", line.strip())
            if m:
                try:
                    confidence = max(0.0, min(1.0, float(m.group(1))))
                except ValueError:
                    pass
            else:
                cleaned_lines.append(line)

        analysis_text = "\n".join(cleaned_lines).strip()

        return FrameAnalysis(
            frame_type=frame_type,
            analysis=analysis_text,
            confidence=confidence,
        )

    def _detect_conflicts(self, frames: dict) -> list:
        """Ask the LLM to identify direct conflicts between frames.

        Returns a list of dicts, each with:
            frame_a, frame_b, conflict, resolution
        """
        from agent import AGENT_MODEL, get_llm_client, chat_completion  # lazy import

        frames_text = "\n\n".join(
            f"[{ft.upper()}]\n{data['analysis']}"
            for ft, data in frames.items()
        )

        prompt = (
            "Here are analyses of a problem from five different perspectives:\n\n"
            f"{frames_text}\n\n"
            "List any DIRECT conflicts — cases where two frames examine the same issue "
            "but reach opposing conclusions or recommendations.\n\n"
            "For each conflict, respond with a JSON object on one line:\n"
            '{"frame_a": "<frame>", "frame_b": "<frame>", '
            '"conflict": "<what they disagree on>", "resolution": "<suggested resolution>"}\n\n'
            "Output ONLY these JSON objects, one per line, with no other text. "
            "If there are no conflicts, output the single word: NONE"
        )

        client = get_llm_client(timeout=60.0)
        response = chat_completion(
            client,
            messages=[{"role": "user", "content": prompt}],
            model=AGENT_MODEL,
            max_tokens=500,
            temperature=0.4,
        )
        raw: str = (response.choices[0].message.content or "").strip()

        if raw.upper() == "NONE" or not raw:
            return []

        conflicts: list[dict] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            # Strip leading list markers the model might add
            line = re.sub(r"^\s*[-*\d.]+\s*", "", line)
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    # Normalise keys — fill missing ones with empty strings
                    conflicts.append({
                        "frame_a": str(obj.get("frame_a", "")),
                        "frame_b": str(obj.get("frame_b", "")),
                        "conflict": str(obj.get("conflict", "")),
                        "resolution": str(obj.get("resolution", "")),
                    })
            except (json.JSONDecodeError, ValueError):
                # Attempt to extract a JSON object buried in prose
                m = re.search(r"\{.*\}", line)
                if m:
                    try:
                        obj = json.loads(m.group())
                        if isinstance(obj, dict):
                            conflicts.append({
                                "frame_a": str(obj.get("frame_a", "")),
                                "frame_b": str(obj.get("frame_b", "")),
                                "conflict": str(obj.get("conflict", "")),
                                "resolution": str(obj.get("resolution", "")),
                            })
                    except (json.JSONDecodeError, ValueError):
                        pass

        return conflicts

    def _synthesize(self, problem: str, frames: dict, conflicts: list) -> str:
        """Combine all frame analyses and detected conflicts into one recommendation.

        Returns a 1-2 sentence synthesis string.
        """
        from agent import AGENT_MODEL, get_llm_client, chat_completion  # lazy import

        frames_text = "\n".join(
            f"- {ft.capitalize()} ({data['confidence']:.0%} confidence): {data['analysis']}"
            for ft, data in frames.items()
        )

        conflicts_text = ""
        if conflicts:
            conflict_lines = [
                f"- {c['frame_a']} vs {c['frame_b']}: {c['conflict']} (resolution: {c['resolution']})"
                for c in conflicts
            ]
            conflicts_text = "\nKey conflicts identified:\n" + "\n".join(conflict_lines)

        prompt = (
            f"Problem: {problem}\n\n"
            f"Frame analyses:\n{frames_text}"
            f"{conflicts_text}\n\n"
            "Write a 1-2 sentence synthesis recommendation that combines the insights from "
            "all frames and accounts for any conflicts. Be concrete and actionable. "
            "No preamble — start directly with the recommendation."
        )

        client = get_llm_client(timeout=60.0)
        response = chat_completion(
            client,
            messages=[{"role": "user", "content": prompt}],
            model=AGENT_MODEL,
            max_tokens=200,
            temperature=0.6,
        )
        return (response.choices[0].message.content or "").strip()

    def save_analysis(self, problem: str, result: dict) -> None:
        """Persist this analysis to memory["multi_frame_analyses"], keeping last 20.

        Uses the same memory store as the rest of PinPoint.
        """
        from tools import _load_memory, _save_memory_file  # lazy import

        memory = _load_memory()

        entry: dict[str, Any] = {
            "problem": problem[:500],  # cap to avoid bloating memory
            "synthesis": result.get("synthesis", ""),
            "conflict_count": len(result.get("conflicts", [])),
            "frame_confidences": {
                ft: data.get("confidence", 0.0)
                for ft, data in result.get("frames", {}).items()
            },
        }

        analyses: list = memory.get("multi_frame_analyses", [])
        analyses.append(entry)
        analyses = analyses[-20:]  # keep only the most recent 20
        memory["multi_frame_analyses"] = analyses

        _save_memory_file(memory)


# ── Module-level convenience function ────────────────────────────────────────

def analyze(problem: str) -> str:
    """Run multi-frame analysis on *problem* and return the full result as JSON.

    This is the primary public entry point for external callers.
    """
    analyzer = MultiFrameAnalyzer()
    result = analyzer.analyze(problem)
    analyzer.save_analysis(problem, result)
    return json.dumps(result, indent=2, ensure_ascii=False)
