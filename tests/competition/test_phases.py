import dataclasses

import pytest

from marioai.competition import PHASES, PhaseConfig


def test_phases_match_competition_readme() -> None:
    assert len(PHASES) == 5
    expected = [
        PhaseConfig('1-easy', 0, 0, 1001, 2, 60),
        PhaseConfig('2-medium-A', 5, 0, 2042, 2, 80),
        PhaseConfig('3-medium-B', 8, 1, 2077, 2, 100),
        PhaseConfig('4-hard-A', 15, 2, 3013, 1, 120),
        PhaseConfig('5-hard-B', 20, 3, 3099, 0, 120),
    ]
    assert PHASES == expected


def test_phase_config_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        PHASES[0].level_seed = 42  # type: ignore[misc]
