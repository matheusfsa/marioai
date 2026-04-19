"""Pixel-observation variant of :class:`ShapedMarioEnv`.

The observation is the rendered window frame captured via
:class:`marioai.capture.GameWindowCapture` instead of the symbolic 22×22 grid.
The TCP state is still consumed (it is the source of truth for episode
termination, distance, coins and status used by reward shaping) — only the
*observation* the policy sees changes.

This env is the entry point for the ``DQN pixels + CNN`` agent of the
competition (``competition/agents/dqn_pixels``). Wrap it with
``gym.wrappers.FrameStack(env, num_stack=4)`` to feed an Atari-style 4×84×84
tensor to a ``CnnPolicy``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from gym import spaces

from ..core.utils import FitnessResult, Observation
from .shaped_environment import ShapedMarioEnv

if TYPE_CHECKING:
    from ..capture import GameWindowCapture

__all__ = ['ShapedPixelMarioEnv']

logger = logging.getLogger(__name__)

_MAX_MISSED_FRAMES_WARN = 5


class ShapedPixelMarioEnv(ShapedMarioEnv):
    """RGB/grayscale pixel observations + shaped reward.

    The ``capture`` instance must be configured with ``grayscale=True`` and
    a non-None ``resize``; the resulting observation has shape
    ``(height, width)`` and dtype ``uint8`` — the format expected by SB3's
    ``CnnPolicy`` after ``VecTransposeImage`` adds the channel dim.
    """

    def __init__(
        self,
        capture: GameWindowCapture,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not capture.grayscale or capture.resize is None:
            raise ValueError(
                'ShapedPixelMarioEnv requires the capture to be grayscale=True with a non-None resize, '
                f'got grayscale={capture.grayscale}, resize={capture.resize}',
            )
        self._capture = capture
        width, height = capture.resize
        self.observation_space = spaces.Box(low=0, high=255, shape=(height, width), dtype=np.uint8)
        self._last_obs: np.ndarray | None = None
        self._missed_frames = 0
        self._capture.start()

    # ------------------------------------------------------------------
    # gym overrides
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        # Reset reward-shaping accumulators and the cached frame so transient
        # capture failures across episode boundaries don't leak the previous
        # episode's last frame into the new one.
        self._last_obs = None
        self._missed_frames = 0
        return super().reset()

    def _build_observation(self, sense: Observation | FitnessResult) -> np.ndarray:
        if isinstance(sense, FitnessResult):
            self.finished = True
            # Return the last good frame (or zeros) so the shape contract is
            # preserved for SB3's terminal-step bookkeeping.
            return self._last_obs if self._last_obs is not None else np.zeros(self.observation_space.shape, dtype=np.uint8)

        frame = self._capture.capture_frame()
        if frame is None:
            self._missed_frames += 1
            if self._missed_frames == _MAX_MISSED_FRAMES_WARN:
                logger.warning('[pixel-env] %d consecutive frames missed; check the capture window', self._missed_frames)
            if self._last_obs is None:
                # Cold start: SB3 needs *something*. Zeros are fine — the next
                # frame will overwrite this and the policy hasn't acted yet.
                return np.zeros(self.observation_space.shape, dtype=np.uint8)
            return self._last_obs

        self._missed_frames = 0
        self._last_obs = frame
        return frame

    def close(self) -> None:
        try:
            self._capture.stop()
        finally:
            super().close()
