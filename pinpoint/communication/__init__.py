"""Communication subsystem — messaging, calling, email.

Provider-agnostic by design. PinPoint talks to this layer; this layer talks to
whatever real provider is configured. If none is configured, the honest answer
is that the capability is unavailable — never a simulated send that reads like
a real one.

Credentials come from the environment only. Nothing in this package writes a
credential to memory.json, to the audit log, or to any other file.
"""
