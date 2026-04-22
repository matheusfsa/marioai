"""Off-policy tabular Q-learning agent.

Online TD(0) updates are applied between frames: after the agent sees the new
state ``s'`` it closes out the previous transition ``(s, a, r, s')`` with
``Q(s, a) ← Q(s, a) + α [r + γ · max_a' Q(s', a')  − Q(s, a)]``.

Updates are driven by the usual `Experiment` loop (``sense`` → ``act`` →
``give_rewards``) so no changes to the Runner are required.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from marioai.agents.base_agent import BaseAgent
from marioai.agents.utils import TABULAR_STATE_KEYS, State, decay_epsilon
from marioai.core import Runner, Task

__all__ = ['QLearningAgent']

FORWARD_ACTION = (0, 1, 0, 0, 0)
FORWARD_BIAS = 1e-3


class QLearningAgent(BaseAgent):
    """Tabular off-policy Q-learning with ε-greedy behaviour policy."""

    def __init__(
        self,
        n_episodes: int = 3000,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        decay_fraction: float = 0.8,
    ) -> None:
        super().__init__()
        self.n_episodes = n_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_fraction = decay_fraction
        self.epsilon: float = epsilon_start
        self.in_fit = False
        self.policy_kind = 'greedy'
        self._Q: dict[State, np.ndarray] = {}
        self._action_pool: np.ndarray | None = None
        self.fit_rewards: list[float] = []
        self.actual_x: float = 0

        self._prev_state: State | None = None
        self._prev_action: int | None = None
        self._prev_reward: float | None = None
        self._episode_reward: float = 0.0

    def reset(self) -> None:
        self.actual_x = 0
        self.states = []
        self.rewards = []
        self.actions = []
        self._prev_state = None
        self._prev_action = None
        self._prev_reward = None
        self._episode_reward = 0.0

    def set_deterministic(self) -> None:
        self.policy_kind = 'greedy'
        self.epsilon = 0.0

    def compute_reward(self, reward_data: dict[str, float]) -> float:
        """Dense shaping with strong terminal signal (see 02-modelagem.md)."""
        status = reward_data.get('status', 0)
        if status == 1:
            return 100.0 + 10.0 * reward_data.get('coins', 0)
        if status == 2:
            return -50.0 + 10.0 * reward_data.get('coins', 0)
        if self.mario_floats is None:
            return 0.0
        dx = self.mario_floats[0] - self.actual_x
        self.actual_x = self.mario_floats[0]
        return dx * 0.1

    def filter_actions(self) -> np.ndarray:
        assert self._action_pool is not None, 'fit() must be called before filter_actions'
        action_pool = np.copy(self._action_pool)
        if self.state is not None and not self.state.get('can_jump', True):
            action_pool = action_pool[action_pool[:, 3] == 0]
        return action_pool

    @staticmethod
    def _state_key(obs: dict[str, Any]) -> State:
        return State(**{k: bool(obs.get(k)) for k in TABULAR_STATE_KEYS})

    def _init_q(self, action_pool: np.ndarray) -> np.ndarray:
        q = np.zeros(action_pool.shape[0])
        for idx, row in enumerate(action_pool):
            if tuple(int(x) for x in row) == FORWARD_ACTION:
                q[idx] = FORWARD_BIAS
                break
        return q

    def _ensure_q(self, state: State, action_pool: np.ndarray) -> np.ndarray:
        if state not in self._Q:
            self._Q[state] = self._init_q(action_pool)
        return self._Q[state]

    def policy(self, q_values: np.ndarray) -> int:
        if self.policy_kind == 'greedy':
            return int(q_values.argmax())
        if self.policy_kind == 'e_greedy':
            if np.random.random() < self.epsilon:
                return int(np.random.randint(q_values.shape[0]))
            return int(q_values.argmax())
        if self.policy_kind == 'random':
            return int(np.random.randint(q_values.shape[0]))
        raise ValueError(f'Unknown policy_kind: {self.policy_kind!r}')

    def _td_update(self, next_state: State | None, terminal: bool) -> None:
        """Apply Q-learning TD(0) update on the pending (s, a, r) transition."""
        if self._prev_state is None or self._prev_action is None or self._prev_reward is None:
            return
        q_sa = self._Q[self._prev_state][self._prev_action]
        if terminal or next_state is None:
            target = self._prev_reward
        else:
            target = self._prev_reward + self.gamma * float(self._Q[next_state].max())
        self._Q[self._prev_state][self._prev_action] = q_sa + self.alpha * (target - q_sa)

    def act(self) -> list[int]:
        action_pool = self.filter_actions()
        state = self._state_key(self.state)
        q_values = self._ensure_q(state, action_pool)

        if self.in_fit:
            self._td_update(state, terminal=False)

        action_idx = self.policy(q_values)
        action = action_pool[action_idx].tolist()
        self.actions.append(action)

        self._prev_state = state
        self._prev_action = action_idx
        self._prev_reward = None
        return action

    def give_rewards(self, reward: dict[str, float], cum_reward: float) -> None:
        super().give_rewards(reward, cum_reward)
        if not self.in_fit:
            return
        r = self.compute_reward(reward)
        self._episode_reward += r
        self._prev_reward = r
        terminal = reward.get('status', 0) in (1, 2)
        if terminal:
            self._td_update(next_state=None, terminal=True)
            self.fit_rewards.append(self._episode_reward)
            self._prev_state = None
            self._prev_action = None
            self._prev_reward = None

    def fit(self, task: Task, **runner_kwargs: Any) -> QLearningAgent:
        self.in_fit = True
        self._action_pool = task._action_pool
        self.policy_kind = 'e_greedy'
        runner = Runner(self, task, **runner_kwargs)

        decay_total = max(int(self.n_episodes * self.decay_fraction), 1)
        pbar = tqdm(range(self.n_episodes), total=self.n_episodes)
        for ep in pbar:
            self.epsilon = decay_epsilon(
                'linear',
                min(ep, decay_total - 1),
                decay_total,
                start=self.epsilon_start,
                end=self.epsilon_end,
            )
            runner.run()
            last = self.fit_rewards[-1] if self.fit_rewards else 0.0
            pbar.set_description(f'Reward {last:.2f} ε={self.epsilon:.3f} |Q|={len(self._Q)}')
            pbar.refresh()

        runner.close()
        self.in_fit = False
        self.set_deterministic()
        return self

    def save(self, path: str | Path) -> None:
        payload = {
            'Q': self._Q,
            'action_pool': self._action_pool,
            'n_episodes': self.n_episodes,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'decay_fraction': self.decay_fraction,
        }
        with open(path, 'wb') as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> QLearningAgent:
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        agent = cls(
            n_episodes=payload['n_episodes'],
            alpha=payload['alpha'],
            gamma=payload['gamma'],
            epsilon_start=payload['epsilon_start'],
            epsilon_end=payload['epsilon_end'],
            decay_fraction=payload['decay_fraction'],
        )
        agent._Q = payload['Q']
        agent._action_pool = payload['action_pool']
        agent.set_deterministic()
        return agent
