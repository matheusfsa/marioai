import numpy as np
import pytest

from marioai.core.utils import (
    FitnessResult,
    Observation,
    decode,
    extract_observation,
)


class TestExtractObservation:
    def test_fit_message_returns_fitness_result(self, fit_message: bytes) -> None:
        result = extract_observation(fit_message)
        assert isinstance(result, FitnessResult)
        assert result.status == 1
        assert result.distance == pytest.approx(123.45)
        assert result.time_left == 99
        assert result.mario_mode == 2
        assert result.coins == 5

    def test_o_message_returns_observation(self, o_message: bytes) -> None:
        result = extract_observation(o_message)
        assert isinstance(result, Observation)
        assert result.may_jump is True
        assert result.on_ground is False
        assert result.mario_floats == (1.5, 2.5)
        assert result.enemies_floats == []
        assert result.level_scene.shape == (22, 22)
        assert result.level_scene.dtype.kind == 'i'

    def test_o_message_parses_enemy_positions(self) -> None:
        tokens = ['O', 'false', 'true'] + ['0'] * 484 + ['10.0', '20.0', '3.5', '4.5']
        result = extract_observation(' '.join(tokens).encode())
        assert isinstance(result, Observation)
        assert result.mario_floats == (10.0, 20.0)
        assert result.enemies_floats == [3.5, 4.5]

    def test_o_message_parses_level_scene_values(self) -> None:
        grid_values = [str(i % 10) for i in range(484)]
        tokens = ['O', 'false', 'false'] + grid_values + ['0.0', '0.0']
        result = extract_observation(' '.join(tokens).encode())
        assert isinstance(result, Observation)
        assert result.level_scene[0, 0] == 0
        assert result.level_scene[0, 1] == 1
        assert result.level_scene[0, 9] == 9
        assert result.level_scene[0, 10] == 0

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError):
            extract_observation(b'XYZ some other payload')


class TestDecode:
    def test_decode_rejects_wrong_length(self) -> None:
        with pytest.raises(ValueError):
            decode('short')

    def test_decode_returns_22x22_grid(self) -> None:
        payload = '\x00' * 31
        grid, checksum = decode(payload)
        assert grid.shape == (22, 22)
        assert checksum == 0
        assert np.all(grid == 0)


class TestChecksumMismatch:
    def test_warns_but_does_not_raise(self, caplog: pytest.LogCaptureFixture) -> None:
        payload = 'E11' + ('\x00' * 31) + '99999'
        with caplog.at_level('WARNING'):
            result = extract_observation(payload.encode())
        assert isinstance(result, Observation)
        assert any('checksum mismatch' in rec.message for rec in caplog.records)
