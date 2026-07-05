# reasoning_frames.py — Phase 3: Multi-Frame Analysis

import json
from typing import Dict, List
from datetime import datetime

FRAME_TYPES = ["technical", "economic", "temporal", "social", "creative"]

class FrameAnalysis:
    """One perspective on a problem."""
    def __init__(self, frame_type: str):
        if frame_type not in FRAME_TYPES:
            raise ValueError(f"Frame type must be one of {FRAME_TYPES}")

        self.frame_type = frame_type
        self.analysis: str = ""
        self.confidence: float = 0.5
        self.key_points: List[str] = []
        self.warnings: List[str] = []
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "frame_type": self.frame_type,
            "analysis": self.analysis,
            "confidence": self.confidence,
            "key_points": self.key_points,
            "warnings": self.warnings,
            "timestamp": self.timestamp,
        }

class ConflictDetector:
    """Find where frames disagree."""

    def __init__(self):
        # Opposing keyword PAIRS (must be 2-tuples — they unpack into a/b sides)
        self.conflict_keywords = {
            "speed": ("fast", "slow"),
            "pace": ("quick", "delay"),
            "cost": ("cheap", "expensive"),
            "time": ("now", "later"),
            "urgency": ("urgent", "long"),
            "horizon": ("now", "months"),
            "complexity": ("simple", "complex"),
            "difficulty": ("easy", "hard"),
            "impact": ("major", "minor"),
            "criticality": ("critical", "trivial"),
        }

    def detect(self, frame_a: str, analysis_a: str, frame_b: str, analysis_b: str) -> List[Dict]:
        """Find conflicts between two frames."""
        conflicts = []

        for topic, (keyword_a, keyword_b) in self.conflict_keywords.items():
            has_a_1 = keyword_a.lower() in analysis_a.lower()
            has_a_2 = keyword_b.lower() in analysis_a.lower()
            has_b_1 = keyword_a.lower() in analysis_b.lower()
            has_b_2 = keyword_b.lower() in analysis_b.lower()

            # Detect opposing positions
            if (has_a_1 and has_b_2) or (has_a_2 and has_b_1):
                conflicts.append({
                    "topic": topic,
                    "frame_a": frame_a,
                    "frame_b": frame_b,
                    "conflict": f"{frame_a} emphasizes {keyword_a if has_a_1 else keyword_b}, "
                               f"{frame_b} emphasizes {keyword_b if has_b_2 else keyword_a}",
                    "severity": "high" if topic in ["cost", "time", "urgency"] else "medium"
                })

        return conflicts

class MultiFrameAnalyzer:
    """Analyze a problem from 5 perspectives simultaneously."""

    def __init__(self):
        self.analyses: Dict[str, FrameAnalysis] = {}
        self.conflicts: List[Dict] = []
        self.synthesis: str = ""
        self.detector = ConflictDetector()

    def add_frame(self, frame_type: str, analysis: str, confidence: float,
                 key_points: List[str], warnings: List[str] = None):
        """Add one frame's analysis."""
        if frame_type not in FRAME_TYPES:
            raise ValueError(f"Frame type must be one of {FRAME_TYPES}")

        fa = FrameAnalysis(frame_type)
        fa.analysis = analysis
        fa.confidence = min(1.0, max(0.0, confidence))
        fa.key_points = key_points
        fa.warnings = warnings or []
        self.analyses[frame_type] = fa

    def detect_conflicts(self):
        """Find where frames disagree."""
        frame_list = list(self.analyses.items())
        self.conflicts = []

        for i, (frame_a, analysis_a) in enumerate(frame_list):
            for frame_b, analysis_b in frame_list[i+1:]:
                detected = self.detector.detect(
                    frame_a, analysis_a.analysis,
                    frame_b, analysis_b.analysis
                )
                self.conflicts.extend(detected)

    def synthesize(self) -> str:
        """Create a unified recommendation from all frames."""
        if not self.analyses:
            return "No frames analyzed yet."

        # Refresh conflicts so synthesis reflects the current frame set
        self.detect_conflicts()

        synthesis = "# Synthesis\n\n"

        # Find consensus
        high_confidence_frames = [
            (f, a) for f, a in self.analyses.items()
            if a.confidence >= 0.8
        ]

        if high_confidence_frames:
            synthesis += f"**Strong consensus from {len(high_confidence_frames)} frames:** "
            synthesis += ", ".join(f[0] for f in high_confidence_frames) + "\n\n"

        # Highlight conflicts
        if self.conflicts:
            synthesis += f"**{len(self.conflicts)} conflicts detected:**\n"
            for conflict in self.conflicts:
                synthesis += f"- {conflict['conflict']}\n"
            synthesis += "\nResolution: Need to choose between the conflicting priorities.\n"
        else:
            synthesis += "**No major conflicts detected** — frames agree on core points.\n"

        # Recommendation
        synthesis += "\n**Recommendation:** "
        if len(high_confidence_frames) >= 3:
            synthesis += "Follow the high-confidence frames."
        elif self.conflicts:
            synthesis += "Resolve identified conflicts before proceeding."
        else:
            synthesis += "Proceed with the analyzed approach."

        return synthesis

    def to_dict(self) -> dict:
        self.detect_conflicts()
        synthesis = self.synthesize()

        return {
            "frames": {name: analysis.to_dict() for name, analysis in self.analyses.items()},
            "conflicts": self.conflicts,
            "synthesis": synthesis,
            "timestamp": datetime.now().isoformat(),
        }

    def to_markdown(self) -> str:
        """Render as markdown."""
        self.detect_conflicts()
        self.synthesis = self.synthesize()

        lines = ["# Multi-Frame Analysis\n"]

        for frame_type in FRAME_TYPES:
            if frame_type in self.analyses:
                fa = self.analyses[frame_type]
                lines.append(f"## {frame_type.capitalize()} Frame")
                lines.append(f"**Confidence: {fa.confidence:.0%}**\n")
                lines.append(fa.analysis + "\n")

                if fa.key_points:
                    lines.append("**Key Points:**")
                    for kp in fa.key_points:
                        lines.append(f"- {kp}")
                    lines.append("")

                if fa.warnings:
                    lines.append("**⚠ Warnings:**")
                    for w in fa.warnings:
                        lines.append(f"- {w}")
                    lines.append("")

        lines.append(self.synthesis)

        return "\n".join(lines)


# ── Compatibility layer ───────────────────────────────────────────────────────
# Existing integration (tools.run_multi_frame_analysis) imports analyze(problem).
# Builds the standard 5-frame analysis, persists it for the live viewer panel,
# and returns pretty JSON.

def build_standard_analysis(problem: str) -> MultiFrameAnalyzer:
    analyzer = MultiFrameAnalyzer()
    analyzer.add_frame("technical",
        f"From a technical perspective, '{problem}' uses proven patterns and has manageable complexity.",
        0.85,
        ["Standard libraries available", "Clear architecture", "Known failure modes"],
        ["May need optimization"])
    analyzer.add_frame("economic",
        "Economically, the cost (time/resources) vs. benefit must be carefully weighed. Initial cost is expensive.",
        0.70,
        ["Implementation cost up front", "Value realization later", "Marginal ROI in short term"],
        ["Long payoff period", "Initial cost high"])
    analyzer.add_frame("temporal",
        "Temporally, deadlines are critical. We should start now if deadlines exist.",
        0.75,
        ["Critical path exists", "Deadline buffer essential", "Start now or delay"],
        ["Time pressure may cause errors"])
    analyzer.add_frame("social",
        "Socially, this affects stakeholders. Coordination and communication matter.",
        0.65,
        ["Multiple parties affected", "Knowledge transfer needed", "Alignment required"],
        ["Buy-in uncertain", "Resistance possible"])
    analyzer.add_frame("creative",
        "Creatively, a hybrid approach could reduce time and cost significantly.",
        0.60,
        ["Novel 80/20 solution", "Untested but promising", "High upside potential"],
        ["Unproven approach", "Failure mode unknown"])
    return analyzer


def _save_analysis_to_memory(problem: str, result: dict) -> None:
    try:
        from tools import _load_memory, _save_memory_file  # lazy — avoids circular import
        mem = _load_memory()
        analyses = mem.get("multi_frame_analyses", [])
        analyses.append({"problem": problem[:200], **result})
        mem["multi_frame_analyses"] = analyses[-20:]
        _save_memory_file(mem)
    except Exception:
        pass


def analyze(problem: str) -> str:
    """Run the 5-frame analysis, persist for the viewer, return JSON string."""
    analyzer = build_standard_analysis(problem)
    result = analyzer.to_dict()
    _save_analysis_to_memory(problem, result)
    return json.dumps(result, indent=2)
