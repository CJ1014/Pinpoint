"""Fixtures for the simulation suite."""

import pytest

from pinpoint.action import executor as EX
from pinpoint.agent import orchestrator as O
from pinpoint.communication import manager as CM
from pinpoint.security import approval, emergency_stop, permissions
from tests.simulation.world import FakeWorld


@pytest.fixture
def world(tmp_path):
    w = FakeWorld(tmp_path / "world")
    yield w
    w.close()


@pytest.fixture(autouse=True)
def clean_state():
    emergency_stop.reset_for_tests()
    approval.reset_for_tests()
    CM.reset_providers()
    permissions.set_profile(permissions.AUTONOMOUS, source=permissions.HUMAN)
    yield
    emergency_stop.reset_for_tests()
    approval.reset_for_tests()
    CM.reset_providers()


@pytest.fixture
def agent(world):
    """Build an orchestrator wired to the fake world.

    Usage: ``agent(step_fn)`` returns a ready orchestrator.
    """
    def build(step_fn, **kwargs):
        executor = EX.Executor(dispatch=world.dispatch, session="sim")
        return O.Orchestrator(executor, step_fn, session="sim",
                              max_iterations=kwargs.pop("max_iterations", 40),
                              **kwargs)
    return build
