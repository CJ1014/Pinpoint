"""PinPoint v3 — autonomous personal agent subsystems.

This package layers new architecture (security, action, planning, memory,
communication, monitoring) on top of the existing PinPoint v2 modules that live
at the repository root. Nothing here replaces a working v2 module; the root
modules stay importable exactly as before.

Import cost is kept near zero: subpackages are not imported here, so
`import pinpoint` does not drag in the whole system.
"""

__version__ = "3.0.0"
