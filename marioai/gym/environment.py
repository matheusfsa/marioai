from __future__ import annotations

from typing import Any

import gym
import numpy as np
from gym import spaces

from ..core import Environment
from ..core.utils import FitnessResult, Observation

__all__ = ['MarioEnv']

ACTIONS: list[list[int]] = [
    # [backward, forward, crouch, jump, speed/bombs]
    [0, 0, 0, 0, 0],  # do nothing
    [0, 0, 0, 1, 0],  # jump
    [0, 0, 0, 0, 1],  # bombs
    [0, 0, 1, 0, 0],  # crouch
    [0, 0, 1, 0, 1],  # crouch and bombs
    [0, 0, 0, 1, 1],  # jump and bombs/speed
    [0, 1, 0, 0, 0],  # move forward
    [0, 1, 0, 0, 1],  # move forward and bombs/speed
    [0, 1, 0, 1, 0],  # jump forward
    [0, 1, 0, 1, 1],  # jump forward and bombs/speed
    [1, 0, 0, 0, 0],  # move backward
    [1, 0, 0, 0, 1],  # move backward and bombs/speed
    [1, 0, 0, 1, 0],  # jump backward
    [1, 0, 0, 1, 1],  # jump backward and bombs/speed
]

LEVEL_SHAPE = (22, 22)
PLAYER_POSITION = 11


class MarioEnv(gym.Env):
    """Gym wrapper around :class:`marioai.core.Environment`.

    The observation space is the 22x22 ``level_scene`` grid, with a few
    values remapped so they fit in ``Box(low=0, high=26)``. The action
    space is :data:`Discrete(14)` over :data:`ACTIONS`.
    """

    def __init__(
        self,
        level_difficulty: int = 0,
        level_type: int = 0,
        creatures_enabled: bool = True,
        mario_mode: int = 2,
        level_seed: int = 1,
        time_limit: int = 100,
        max_fps: int = 24,
        visualization: bool = True,
        fitness_values: int = 5,
    ) -> None:
        self._env = Environment()
        self.max_fps = max_fps
        self.observation_space = spaces.Box(low=0, high=26, shape=LEVEL_SHAPE)
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.mario_pos = 0
        self.finished = False
        self.last_sense: Observation | FitnessResult | None = None

        self._env.level_difficulty = level_difficulty
        self._env.level_type = level_type
        self._env.creatures_enabled = creatures_enabled
        self._env.init_mario_mode = mario_mode
        self._env.level_seed = level_seed
        self._env.time_limit = time_limit
        self._env.visualization = visualization
        self._env.fitness_values = fitness_values

    def _get_info(self, sense: Observation | FitnessResult) -> dict[str, Any]:
        if isinstance(sense, Observation):
            distance = sense.mario_floats[0] if sense.mario_floats else 0.0
        else:
            distance = sense.distance
        return {'distance': distance}

    def _build_observation(self, sense: Observation | FitnessResult) -> np.ndarray:
        """Turn ``sense`` into the 22x22 array expected by the observation space."""
        if isinstance(sense, Observation):
            scene = sense.level_scene.copy()
            scene[scene == 25] = 22
            scene[scene == -11] = 23
            scene[scene == -10] = 24
            scene[scene == 42] = 25
            scene[PLAYER_POSITION, PLAYER_POSITION] = 26
            return scene
        self.finished = True
        return np.zeros(LEVEL_SHAPE)

    def seed(self, seed: int | None = None) -> list[int | None]:
        """Shim for ``shimmy.openai_gym_compatibility`` — SB3 calls this on reset.

        Level randomness is controlled server-side by ``level_seed``, so this
        is a pass-through that just returns the seed in the list form classic
        gym expected.
        """
        return [seed]

    def reset(self) -> np.ndarray:
        self._env.reset()
        sense = self._env.get_sensors()
        self.last_sense = sense
        self.finished = isinstance(sense, FitnessResult)
        return self._build_observation(sense)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        self._env.perform_action(ACTIONS[action])
        sense = self._env.get_sensors()
        self.last_sense = sense
        observation = self._build_observation(sense)
        info = self._get_info(sense)

        if isinstance(sense, FitnessResult):
            reward_data = {
                'status': sense.status,
                'distance': sense.distance,
                'timeLeft': sense.time_left,
                'marioMode': sense.mario_mode,
                'coins': sense.coins,
            }
            reward = self.compute_reward(reward_data)
        else:
            reward = self.compute_reward(info)

        return observation, reward, self.finished, info

    def compute_reward(self, reward_data: dict[str, Any]) -> float:
        if 'distance' in reward_data and reward_data['distance'] is not None:
            return float(reward_data['distance'])
        return 0.0

    def close(self) -> None:
        self._env.disconnect()
