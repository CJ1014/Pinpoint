"""The action layer: structured results, independent verification, execution.

Reasoning code asks the executor to perform an action and gets an
:class:`~pinpoint.action.result.ActionResult` back. It never sees a bare string,
so "did that work?" stops being a substring search over prose.
"""
