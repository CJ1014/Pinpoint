"""Monitoring: events, watchers, scheduling.

PinPoint can wait for the world to change rather than only responding when
spoken to. The pipeline that decides what to do about a change is deterministic
Python — a monitor that woke a language model on every file write would be both
expensive and slow to react.
"""
