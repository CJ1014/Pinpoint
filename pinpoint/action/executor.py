"""The single path through which actions actually happen.

Order of operations, and none of it is skippable:

1. **Emergency stop** — halted means nothing runs.
2. **Policy** — RED is refused outright; YELLOW may need approval.
3. **Approval** — asked once, honoured, audited.
4. **Capability preflight** — a missing capability is reported honestly.
5. **Execute** — with a timeout, exceptions captured as failures.
6. **Verify** — independently, per the registry's declared method.
7. **Audit** — one structured record per attempt.

The reasoning layer talks to this, not to ``tools.dispatch``. That is what
makes "the model decided it was fine" a non-event.
"""

import threading
import time
from typing import Any, Callable, Dict, List, Optional

from pinpoint import epistemics
from pinpoint.action import capabilities, result as R, verify as V
from pinpoint.security import approval, audit, emergency_stop, permissions
from pinpoint.tools import registry


class Executor:
    """Executes actions under policy, with verification and auditing."""

    def __init__(self, dispatch: Optional[Callable[[str, dict], str]] = None,
                 session: Any = "", goal: str = "",
                 on_result: Optional[Callable[[R.ActionResult], None]] = None):
        self._dispatch = dispatch
        self.session = session
        self.goal = goal
        self.on_result = on_result
        self.history: List[R.ActionResult] = []

    # ── dispatch plumbing ────────────────────────────────────────────────────

    def _resolve_dispatch(self) -> Callable[[str, dict], str]:
        if self._dispatch is not None:
            return self._dispatch
        import tools  # imported lazily: heavy, and optional in tests
        return tools.dispatch

    def _run_tool(self, tool: str, params: dict, timeout: float) -> tuple:
        """Run the tool with a wall-clock timeout. Returns ``(output, error)``."""
        box: Dict[str, Any] = {}

        def _call():
            try:
                box["out"] = self._resolve_dispatch()(tool, params)
            except Exception as exc:
                box["err"] = f"{type(exc).__name__}: {exc}"

        thread = threading.Thread(target=_call, daemon=True,
                                  name=f"pinpoint-tool-{tool}")
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            return "", f"timed out after {timeout:.0f}s"
        if "err" in box:
            return "", box["err"]
        output = box.get("out", "")
        return (output if isinstance(output, str) else str(output)), ""

    # ── the pipeline ─────────────────────────────────────────────────────────

    def execute(self, tool: str, params: Optional[dict] = None, *,
                goal: str = "", attempt: int = 1,
                expected: str = "", before: Any = None,
                timeout: Optional[float] = None) -> R.ActionResult:
        params = dict(params or {})
        goal = goal or self.goal
        spec = registry.require(tool)

        # 1. Emergency stop.
        if emergency_stop.is_engaged():
            info = emergency_stop.status()
            res = R.blocked(tool, params,
                            f"emergency stop engaged ({info.get('reason', '')}) — "
                            f"a human must clear it")
            return self._finish(res, goal, attempt)

        # 2. Policy.
        decision = permissions.check(tool, params)
        if decision.blocked:
            res = R.blocked(tool, params, decision.reason, decision.level)
            return self._finish(res, goal, attempt)

        # 3. Approval.
        approval_state = "not_required"
        if decision.requires_approval:
            outcome = approval.request(tool, params, decision, goal=goal)
            if not outcome.approved:
                res = R.denied(tool, params,
                               outcome.reason or "the human declined this action",
                               decision.level)
                return self._finish(res, goal, attempt)
            approval_state = "approved"

        # 4. Capability preflight.
        available, detail = capabilities.check_all(spec.requires)
        if not available:
            res = R.unavailable(tool, params, detail)
            res.permission_level = decision.level
            res.approval = approval_state
            return self._finish(res, goal, attempt)

        # 5. Execute.
        started = time.perf_counter()
        output, error = self._run_tool(tool, params,
                                       timeout if timeout is not None else spec.timeout)
        duration_ms = (time.perf_counter() - started) * 1000.0

        res = R.ActionResult(
            tool=tool, params=params, output=output, error=error,
            duration_ms=duration_ms, reversible=spec.reversible,
            side_effects=self._side_effects(spec), permission_level=decision.level,
            approval=approval_state, goal=goal, attempt=attempt,
        )

        if error:
            res.status = R.FAILED
            res.confidence = 0.95
            res.epistemic = epistemics.OBSERVED
            res.verification = V.VerificationOutcome(
                spec.verification, False, error, 0.95, epistemics.OBSERVED).to_dict()
            return self._finish(res, goal, attempt)

        # 6. Verify — independently, not by absence of an error string.
        outcome = V.verify(tool, params, output, method=spec.verification,
                           before=before, after=None, expected=expected)
        res.verification = outcome.to_dict()
        res.confidence = outcome.confidence
        res.epistemic = outcome.epistemic

        if outcome.verified is True:
            res.status = R.SUCCESS
        elif outcome.verified is False:
            res.status = R.FAILED
            res.error = outcome.detail
        else:
            res.status = R.UNVERIFIED

        return self._finish(res, goal, attempt)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _side_effects(spec: registry.ToolSpec) -> List[str]:
        effects = []
        if spec.external_side_effect:
            effects.append("external")
        if spec.costs_money:
            effects.append("billable")
        if spec.destructive:
            effects.append("destructive")
        if not spec.reversible:
            effects.append("irreversible")
        return effects

    def _finish(self, res: R.ActionResult, goal: str, attempt: int) -> R.ActionResult:
        res.goal = goal
        res.attempt = attempt
        self.history.append(res)
        audit.record(
            "action", tool=res.tool, params=res.params, result=res.output,
            error=res.error, goal=goal, session=self.session,
            permission=res.permission_level, approval=res.approval,
            verification=res.verification, duration_ms=res.duration_ms,
            extra={"action_id": res.action_id, "status": res.status,
                   "attempt": attempt, "confidence": res.confidence},
        )
        if self.on_result is not None:
            try:
                self.on_result(res)
            except Exception:
                pass
        return res

    # ── reporting ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        counts: Dict[str, int] = {}
        for res in self.history:
            counts[res.status] = counts.get(res.status, 0) + 1
        return {
            "actions": len(self.history),
            "by_status": counts,
            "verified_successes": counts.get(R.SUCCESS, 0),
            "unverified": counts.get(R.UNVERIFIED, 0),
            "failures": counts.get(R.FAILED, 0),
            "blocked": counts.get(R.BLOCKED, 0) + counts.get(R.DENIED, 0),
        }

    def last(self, tool: str = "") -> Optional[R.ActionResult]:
        for res in reversed(self.history):
            if not tool or res.tool == tool:
                return res
        return None
