"""Security subsystem: permissions, approval, audit, emergency stop.

These modules are deliberately free of LLM calls. Every decision they make is a
deterministic function of the tool name, its parameters, and stored human
grants — never of model-generated prose. That is what makes the policy
unbypassable by the agent talking itself into an exception.
"""
