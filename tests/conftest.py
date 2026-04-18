import numpy as np
import pytest


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
