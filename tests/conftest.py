from unittest.mock import MagicMock

import numpy as np
import pytest

from marioai.core import environment as _environment


@pytest.fixture(autouse=True)
def _no_java_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent ``Environment.__init__`` from spawning the Java server.

    Several tests construct ``MarioEnv``/``ShapedPixelMarioEnv`` to exercise
    Python-side logic and then swap ``env._env`` for a ``MagicMock``. Without
    this patch the real server subprocess is spawned and then orphaned when
    the reference is replaced, leaving ``java ch.idsia.scenarios.MainRun``
    processes alive after the test run.
    """

    def _fake_run_server(self: _environment.Environment) -> MagicMock:
        self._server_process = None
        self._stdout_log = None
        self._stderr_log = None
        return MagicMock(name='TCPClient')

    monkeypatch.setattr(_environment.Environment, '_run_server', _fake_run_server)


@pytest.fixture
def zero_level_scene() -> np.ndarray:
    """22x22 grid of zeros (empty world)."""
    return np.zeros((22, 22), dtype=int)


@pytest.fixture
def o_message() -> bytes:
    """Minimal ``O`` observation: empty grid, Mario at (1.5, 2.5), no enemies."""
    tokens = ['O', 'true', 'false'] + ['0'] * 484 + ['1.5', '2.5']
    return ' '.join(tokens).encode()


@pytest.fixture
def fit_message() -> bytes:
    """Minimal ``FIT`` message: won, distance 123.45, 99 seconds left."""
    return b'FIT 1 123.45 99 2 5'
