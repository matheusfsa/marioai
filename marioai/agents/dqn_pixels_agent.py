"""Wrapper that exposes a trained SB3 DQN (CnnPolicy) as an :class:`Agent`.

This agent consumes the pixel frames captured by
:class:`marioai.capture.GameWindowCapture` (delivered through
:meth:`Agent.observe_frame`), maintains a 4-frame Atari-style stack, and
queries the loaded ``DQN`` model deterministically each step.

It is the inference-side counterpart of
``competition/agents/dqn_pixels/train.py`` — train with that script, save the
``.zip``, then load it here for evaluation against the competition phases.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from ..core.agent import Agent
from ..gym.environment import ACTIONS

__all__ = ['DqnPixelsAgent']

logger = logging.getLogger(__name__)

_FRAME_STACK = 4
_DEFAULT_OBS_SHAPE = (84, 84)


class DqnPixelsAgent(Agent):
    """Inference adapter for a CNN-based DQN trained on pixel observations."""

    def __init__(
        self,
        model_path: str,
        obs_shape: tuple[int, int] = _DEFAULT_OBS_SHAPE,
        deterministic: bool = True,
    ) -> None:
        super().__init__()
        # Lazy import: stable_baselines3 is heavy and only needed when this
        # agent is actually instantiated.
        from stable_baselines3 import DQN

        self.model = DQN.load(model_path)
        self.obs_shape = obs_shape
        self._deterministic = deterministic
        self._frames: deque[np.ndarray] = deque(maxlen=_FRAME_STACK)
        self._cv2 = self._maybe_import_cv2()
        self._last_action: list[int] = [0, 0, 0, 0, 0]

    @staticmethod
    def _maybe_import_cv2() -> Any | None:
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError:
            return None
        return cv2

    # ------------------------------------------------------------------
    # Agent overrides
    # ------------------------------------------------------------------
    def reset(self) -> None:
        super().reset()
        self._frames.clear()
        self._last_action = [0, 0, 0, 0, 0]

    def observe_frame(self, frame: np.ndarray | None) -> None:
        """Receive the latest captured frame, preprocess, and push to the stack."""
        if frame is None:
            # Capture failed transiently — keep the existing stack so the
            # policy still has 4 frames to look at.
            return
        processed = self._preprocess(frame)
        self._frames.append(processed)

    def act(self) -> list[int]:
        if not self._frames:
            # No frame seen yet — stand still until the first observation arrives.
            return [0, 0, 0, 0, 0]
        # Pad the stack by replicating the oldest frame until we have 4.
        while len(self._frames) < _FRAME_STACK:
            self._frames.appendleft(self._frames[0])
        stacked = np.stack(list(self._frames), axis=0)  # (4, H, W) uint8
        action_idx, _ = self.model.predict(stacked, deterministic=self._deterministic)
        action = ACTIONS[int(action_idx)]
        self._last_action = action
        return action

    # ------------------------------------------------------------------
    # Optional Protocol method — the future CompetitionRunner is allowed to
    # call this on any agent that exposes it.
    # ------------------------------------------------------------------
    def set_deterministic(self) -> None:
        self._deterministic = True

    # ------------------------------------------------------------------
    # preprocessing
    # ------------------------------------------------------------------
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Coerce a captured frame to ``self.obs_shape`` uint8 grayscale."""
        target_h, target_w = self.obs_shape
        # Already in the right shape — skip work (the common case when the
        # GameWindowCapture is configured with grayscale=True, resize=(W,H)).
        if frame.ndim == 2 and frame.shape == (target_h, target_w) and frame.dtype == np.uint8:
            return frame
        if frame.ndim == 3 and frame.shape[2] == 3:
            if self._cv2 is not None:
                frame = self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2GRAY)
            else:
                frame = (frame.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)).astype(np.uint8)
        if frame.shape != (target_h, target_w):
            if self._cv2 is None:
                raise RuntimeError(
                    f"cannot resize frame to {self.obs_shape} without opencv-python; install with: pip install 'marioai[capture]'",
                )
            frame = self._cv2.resize(frame, (target_w, target_h), interpolation=self._cv2.INTER_AREA)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        return frame
