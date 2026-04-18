import logging
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    'FitnessResult',
    'Observation',
    'decode',
    'extract_observation',
    'extractObservation',
]

logger = logging.getLogger(__name__)

_POWERS_OF_2 = tuple(1 << i for i in range(18))
_ENCODED_SCENE_LEN = 31
_LEVEL_SHAPE = (22, 22)
_TOTAL_CELLS = _LEVEL_SHAPE[0] * _LEVEL_SHAPE[1]


@dataclass
class Observation:
    """Per-frame observation decoded from the server's ``O`` or ``E`` message.

    Attributes:
      may_jump: whether Mario may jump this frame.
      on_ground: whether Mario is touching the ground.
      mario_floats: Mario's (x, y) position in the level. ``None`` for the
        bit-packed ``E`` format, which only carries the grid.
      enemies_floats: flat list of enemy coordinates (x, y interleaved).
        Empty for the ``E`` format.
      level_scene: 22x22 grid of tile/entity ids around Mario.
    """

    may_jump: bool
    on_ground: bool
    level_scene: np.ndarray
    mario_floats: tuple[float, float] | None = None
    enemies_floats: list[float] = field(default_factory=list)


@dataclass
class FitnessResult:
    """End-of-episode summary decoded from the server's ``FIT`` message."""

    status: int
    distance: float
    time_left: int
    mario_mode: int
    coins: int


def decode(estate: str) -> tuple[np.ndarray, int]:
    """Decode the bit-packed 31-char scene used by the ``E`` observation format.

    Returns:
      The 22x22 grid as a numpy array, and the checksum computed from the
      input string.
    """
    if len(estate) != _ENCODED_SCENE_LEN:
        raise ValueError(f'Encoded scene must have {_ENCODED_SCENE_LEN} chars, got {len(estate)}')

    dstate = np.full(_LEVEL_SHAPE, 2, dtype=int)
    row = col = 0
    total_bits = 0
    check_sum = 0

    for cur_char in estate:
        code = ord(cur_char)
        if code != 0:
            check_sum += code
        for j in range(16):
            total_bits += 1
            if col > 21:
                row += 1
                col = 0
            dstate[row, col] = 1 if (_POWERS_OF_2[j] & code) else 0
            col += 1
            if total_bits == _TOTAL_CELLS:
                break
        if total_bits == _TOTAL_CELLS:
            break

    logger.debug('decoded %d bits from E-format scene', total_bits)
    return dstate, check_sum


def extract_observation(data: bytes) -> Observation | FitnessResult:
    """Parse a raw server message into an :class:`Observation` or :class:`FitnessResult`.

    Raises:
      ValueError: if the first byte is unrecognised.
    """
    decoded = data.decode()

    if decoded.startswith('E'):
        may_jump = decoded[1] == '1'
        on_ground = decoded[2] == '1'
        level_scene, check_sum_got = decode(decoded[3 : 3 + _ENCODED_SCENE_LEN])
        try:
            check_sum_recv = int(decoded[3 + _ENCODED_SCENE_LEN :])
        except ValueError:
            check_sum_recv = None
        if check_sum_recv is not None and check_sum_got != check_sum_recv:
            logger.warning(
                'E-format checksum mismatch: got %d != recv %d',
                check_sum_got,
                check_sum_recv,
            )
        return Observation(
            may_jump=may_jump,
            on_ground=on_ground,
            level_scene=level_scene,
        )

    tokens = decoded.split(' ')
    head = tokens[0]

    if head == 'FIT':
        return FitnessResult(
            status=int(tokens[1]),
            distance=float(tokens[2]),
            time_left=int(tokens[3]),
            mario_mode=int(tokens[4]),
            coins=int(tokens[5]),
        )

    if head == 'O':
        may_jump = tokens[1] == 'true'
        on_ground = tokens[2] == 'true'
        level_scene = np.empty(_LEVEL_SHAPE, dtype=int)
        k = 0
        for i in range(_LEVEL_SHAPE[0]):
            for j in range(_LEVEL_SHAPE[1]):
                level_scene[i, j] = int(tokens[k + 3])
                k += 1
        k += 3
        float_x = _parse_float_token(tokens[k])
        float_y = _parse_float_token(tokens[k + 1])
        mario_floats = (float_x, float_y)
        k += 2
        enemies_floats = [float(tok) for tok in tokens[k:] if tok]
        return Observation(
            may_jump=may_jump,
            on_ground=on_ground,
            level_scene=level_scene,
            mario_floats=mario_floats,
            enemies_floats=enemies_floats,
        )

    raise ValueError(f'Wrong format or corrupted observation: first byte={head!r}')


def _parse_float_token(token: str) -> float:
    """Parse a float token that may have trailing garbage (e.g., a delimiter)."""
    try:
        return float(token)
    except ValueError:
        return float(token[:-2])


# Backwards-compatible alias so notebooks importing the old camelCase name keep working.
extractObservation = extract_observation
