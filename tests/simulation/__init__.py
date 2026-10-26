"""A deterministic fake world for testing autonomous behaviour.

The point of this package is to model *tool semantics*, not to make the agent
look good. A simulated `write_file` returns whatever the simulated filesystem
actually did — so if the world is configured to accept the call but not create
the file, the agent still has to catch that through verification.

Nothing here shortcuts the agent's own checks.
"""

from tests.simulation.world import (  # noqa: F401
    FakeWorld, Failure, Rule, ToolCall,
)
