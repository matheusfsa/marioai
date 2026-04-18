from __future__ import annotations

from typing import Any

import numpy as np
from tqdm import tqdm

from marioai.agents.base_agent import BaseAgent
from marioai.core import Runner, Task

__all__ = ['MonteCarloAgent']


class State:
    """Hashable wrapper around a dict so it can be used as a ``_Q`` key."""

    def __init__(self, **kwargs: Any) -> None:
        self.state_attrs: list[str] = []
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.state_attrs.append(key)

    def __repr__(self) -> str:
        parts = [f'{attr}={getattr(self, attr)}' for attr in self.state_attrs]
        return 'State(' + ', '.join(parts) + ')'

    def __hash__(self) -> int:
        values = []
        for attr in self.state_attrs:
            value = getattr(self, attr)
            if isinstance(value, np.ndarray):
                values.append(value.tobytes())
            elif isinstance(value, list):
                values.append(tuple(value))
            else:
                values.append(value)
        return hash(tuple(values))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return NotImplemented
        if self.state_attrs != other.state_attrs:
            return False
        for attr in self.state_attrs:
            a = getattr(self, attr)
            b = getattr(other, attr)
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                if np.any(a != b):
                    return False
            elif a != b:
                return False
        return True


class MonteCarloAgent(BaseAgent):
    """Tabular Monte Carlo control with ε-greedy exploration."""

    def __init__(
        self,
        n_samples: int,
        discount: float,
        min_epsilon: float = 0.3,
        reward_threshold: float = 0,
        reward_increment: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.discount = discount
        self.in_fit = False
        self.min_epsilon = min_epsilon
        self.reward_threshold = reward_threshold
        self.reward_increment = reward_increment
        self.epsilon: float = 1.0
        self.policy_kind = 'greedy'
        self._Q: dict[State, np.ndarray] = {}
        self._N: dict[State, np.ndarray] = {}
        self.actions_idx: list[int] = []
        self.fit_rewards: list[float] = []
        self.actual_x: float = 0
        self._action_pool: np.ndarray | None = None

    def reset(self) -> None:
        self.actual_x = 0
        self.states = []
        self.rewards = []
        self.actions = []
        self.actions_idx = []

    def compute_reward(self, reward_data: dict[str, float]) -> float:
        if reward_data.get('status') == 1:
            return reward_data['distance'] * 2
        if 'distance' in reward_data:
            return reward_data['distance'] * 0.1
        if self.mario_floats is None:
            return 0
        dist = self.mario_floats[0] - self.actual_x
        self.actual_x = self.mario_floats[0]
        return dist * 0.01

    def filter_actions(self) -> np.ndarray:
        """Return the action pool with jump actions removed when Mario can't jump."""
        assert self._action_pool is not None, 'fit() must be called before filter_actions'
        action_pool = np.copy(self._action_pool)
        if self.state is not None and not self.state.get('can_jump', True):
            action_pool = action_pool[action_pool[:, 3] == 0]
        return action_pool

    def policy(self, state: State, n_actions: int) -> int:
        """Pick an action index for ``state``."""
        if self.policy_kind == 'random':
            action_idx = np.random.randint(n_actions)
        elif self.policy_kind == 'greedy':
            action_idx = int(self._Q[state].argmax())
        elif self.policy_kind == 'e_greedy':
            if np.random.random() > self.epsilon:
                action_idx = int(self._Q[state].argmax())
            else:
                action_idx = np.random.randint(n_actions)
        else:
            raise ValueError(f'Unknown policy_kind: {self.policy_kind!r}')
        self.actions_idx.append(action_idx)
        return action_idx

    def act(self) -> list[int]:
        action_pool = self.filter_actions()
        state = State(**self.state)
        if state not in self._Q:
            self._Q[state] = np.zeros(action_pool.shape[0])
        action_idx = self.policy(state, action_pool.shape[0])
        action = action_pool[action_idx].tolist()
        self.actions.append(action)
        return action

    def fit(self, task: Task, **runner_kwargs: Any) -> MonteCarloAgent:
        """Train for ``self.n_samples`` episodes and return self."""
        self.in_fit = True
        self._action_pool = task._action_pool
        self.policy_kind = 'e_greedy'
        runner = Runner(self, task, **runner_kwargs)
        self.epsilon = 1.0
        pbar = tqdm(range(self.n_samples), total=self.n_samples)
        for _ in pbar:
            runner.run()
            self.fit_rewards.append(self._step())
            if self.epsilon > self.min_epsilon and self.fit_rewards[-1] >= self.reward_threshold:
                epsilon_delta = 1 / self.n_samples
                self.epsilon = self.epsilon - epsilon_delta
                self.reward_threshold = self.reward_threshold + self.reward_increment
            pbar.set_description(f'Last Reward {self.fit_rewards[-1]:.2f} Epsilon: {self.epsilon:.3f} Reward Th: {self.reward_threshold: .3f}')
            pbar.refresh()

        runner.close()
        self.in_fit = False
        self.policy_kind = 'greedy'
        return self

    def _step(self) -> float:
        rewards = np.array([self.compute_reward(r) for r in self.rewards])
        for i, (state_dict, action) in enumerate(zip(self.states, self.actions_idx, strict=False)):
            state = State(**state_dict)
            if state not in self._N:
                self._N[state] = np.zeros(self._Q[state].shape[0])
            self._N[state][action] += 1

            future_rewards = rewards[i:]
            discounts = np.array([self.discount**k for k in range(future_rewards.shape[0])])
            g_t = float(np.dot(discounts, future_rewards)) / self.n_samples

            q_value = self._Q[state][action]
            self._Q[state][action] = q_value + (1 / self._N[state][action]) * (g_t - q_value)

        return float(rewards.sum())
