from __future__ import annotations

from typing import Any

import numpy as np

from . import sensing
from .environment import Environment
from .utils import FitnessResult, Observation

__all__ = ['Task']


DEFAULT_REWARD: dict[str, float] = {
    'status': 0,
    'distance': 0,
    'timeLeft': 0,
    'marioMode': 0,
    'coins': 0,
}


class Task:
    """Bridges the :class:`Environment` and an :class:`Agent`.

    The task decides how to turn raw observations into a state dict the
    agent can consume, how to compute rewards, and how to filter the
    agent's action pool based on the current state.

    Attributes:
      env: the :class:`Environment` instance this task drives.
      finished: whether the current episode is over.
      reward: the latest reward dict.
      status: the episode status from the last fitness message.
      cum_reward: accumulated reward since the start of the episode.
      samples: number of actions sent during the current episode.
    """

    def __init__(
        self,
        window_size: int = 4,
        max_dist: int = 2,
        player_pos: int = 11,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.env = Environment(*args, **kwargs)
        self.window_size = window_size
        self.max_dist = max_dist
        self.player_pos = player_pos
        self.finished = False
        self.state: dict[str, Any] | None = None
        self.status = 0
        self.cum_reward: float = 0
        self.samples = 0
        self.reward: dict[str, float] = dict(DEFAULT_REWARD)
        self._action_pool = np.array(
            [
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
        )
        self.objects = sensing.DEFAULT_OBJECTS

    def reset(self) -> None:
        """Reset the environment and the episode counters."""
        self.env.reset()
        self.cum_reward = 0
        self.samples = 0
        self.finished = False
        self.status = 0
        self.reward = dict(DEFAULT_REWARD)

    def filter_actions(self) -> np.ndarray:
        """Return the action pool with invalid actions removed.

        Drops jump actions when the current state indicates Mario cannot jump.
        """
        action_pool = np.copy(self._action_pool)
        if self.state is not None and not self.state.get('can_jump', True):
            action_pool = action_pool[action_pool[:, 3] == 0]
        return action_pool

    def disconnect(self) -> None:
        self.env.disconnect()

    def get_sensors(self) -> dict[str, Any]:
        """Fetch the next observation and turn it into the agent-facing state dict."""
        sense = self.env.get_sensors()
        self.state = self.build_state(sense)

        if isinstance(sense, FitnessResult):
            reward_data = {
                'status': sense.status,
                'distance': sense.distance,
                'timeLeft': sense.time_left,
                'marioMode': sense.mario_mode,
                'coins': sense.coins,
            }
            self.reward = self.compute_reward(reward_data)
            self.status = sense.status
            self.finished = True

        return self.state

    def compute_reward(self, reward_data: dict[str, float]) -> dict[str, float]:
        """Default implementation just returns the fitness dict verbatim.

        Subclasses override this to produce a scalar reward signal.
        """
        return reward_data

    def perform_action(self, action: list[int]) -> None:
        """Forward the action to the environment and bookkeeping counters."""
        if self.finished:
            return
        self.env.perform_action(action)
        self.cum_reward += self.reward.get('distance', 0) or 0
        self.samples += 1

    def build_state(self, sense: Observation | FitnessResult) -> dict[str, Any]:
        """Turn an observation/fitness payload into the state dict consumed by agents."""
        state: dict[str, Any] = {}

        if isinstance(sense, Observation):
            state['episode_over'] = False
            state['can_jump'] = sense.may_jump
            state['on_ground'] = sense.on_ground
            state['mario_floats'] = sense.mario_floats
            state['enemies_floats'] = sense.enemies_floats
            state['level_scene'] = sense.level_scene
        else:
            state['episode_over'] = True
            state['can_jump'] = None
            state['on_ground'] = None
            state['mario_floats'] = None
            state['enemies_floats'] = None
            state['level_scene'] = None

        level_scene = state['level_scene']

        if not state['episode_over']:
            ground_pos = sensing.get_ground(
                level_scene,
                state['on_ground'],
                window_size=self.window_size,
                player_pos=self.player_pos,
            )
        else:
            ground_pos = None

        for o_name, o_values in self.objects.items():
            for dist in range(1, self.max_dist + 1):
                if state['episode_over']:
                    state[f'{o_name}_{dist}'] = None
                else:
                    state[f'{o_name}_{dist}'] = sensing.is_near(level_scene, o_values, dist, player_pos=self.player_pos)

        for dist in range(1, self.max_dist + 1):
            if state['episode_over']:
                state[f'has_role_near_{dist}'] = None
            else:
                state[f'has_role_near_{dist}'] = sensing.has_role_near(level_scene, ground_pos, dist, player_pos=self.player_pos)

        return state
