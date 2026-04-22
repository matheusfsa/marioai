"""On-policy ε-greedy Monte Carlo agent (every-visit, γ=1).

Compared to :class:`MonteCarloAgent`, this variant:

- Uses an undiscounted return (``γ = 1``), so it learns the raw sum of shaped
  rewards instead of a discounted one (Etapa 3.2 of the competition roadmap).
- Decays ``ε`` linearly from 1.0 to a floor over a fraction of the training
  schedule — not tied to a reward threshold.
- Biases unseen states toward FORWARD (action index 6 of
  :attr:`Task._action_pool`) so greedy evaluation doesn't fall back to
  "do nothing" when it sees a state it never visited during training.
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

__all__ = ['EpsilonGreedyAgent']

FORWARD_ACTION = (0, 1, 0, 0, 0)
FORWARD_BIAS = 1e-3


class EpsilonGreedyAgent(BaseAgent):
    """Tabular ε-greedy agent with first-visit MC updates and γ=1."""

    def __init__(
        self,
        n_episodes: int = 3000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        decay_fraction: float = 0.8,
        first_visit: bool = True,
    ) -> None:
        super().__init__()
        self.n_episodes = n_episodes
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_fraction = decay_fraction
        self.first_visit = first_visit
        self.epsilon: float = epsilon_start
        self.in_fit = False
        self.policy_kind = 'greedy'
        self._Q: dict[State, np.ndarray] = {}
        self._N: dict[State, np.ndarray] = {}
        self.actions_idx: list[int] = []
        self.fit_rewards: list[float] = []
        self.actual_x: float = 0
        self._action_pool: np.ndarray | None = None
        self._forward_idx: int | None = None

    def reset(self) -> None:
        self.actual_x = 0
        self.states = []
        self.rewards = []
        self.actions = []
        self.actions_idx = []

    def set_deterministic(self) -> None:
        """Force greedy action selection for evaluation."""
        self.policy_kind = 'greedy'
        self.epsilon = 0.0

    def compute_reward(self, reward_data: dict[str, float]) -> float:
        """Terminal-aware shaping (see competition/agents/epsilon_greedy/02-modelagem.md)."""
        status = reward_data.get('status', 0)
        if status == 1:
            return 50.0 + 2.0 * reward_data.get('coins', 0)
        if status == 2:
            return -20.0 + 2.0 * reward_data.get('coins', 0)
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

    def policy(self, state: State, n_actions: int) -> int:
        if self.policy_kind == 'greedy':
            action_idx = int(self._Q[state].argmax())
        elif self.policy_kind == 'e_greedy':
            if np.random.random() > self.epsilon:
                action_idx = int(self._Q[state].argmax())
            else:
                action_idx = int(np.random.randint(n_actions))
        elif self.policy_kind == 'random':
            action_idx = int(np.random.randint(n_actions))
        else:
            raise ValueError(f'Unknown policy_kind: {self.policy_kind!r}')
        self.actions_idx.append(action_idx)
        return action_idx

    @staticmethod
    def _state_key(obs: dict[str, Any]) -> State:
        return State(**{k: bool(obs.get(k)) for k in TABULAR_STATE_KEYS})

    def _init_q(self, action_pool: np.ndarray) -> np.ndarray:
        """New state → zeros with a small FORWARD bias so unseen states walk."""
        q = np.zeros(action_pool.shape[0])
        for idx, row in enumerate(action_pool):
            if tuple(int(x) for x in row) == FORWARD_ACTION:
                q[idx] = FORWARD_BIAS
                break
        return q

    def act(self) -> list[int]:
        action_pool = self.filter_actions()
        state = self._state_key(self.state)
        if state not in self._Q:
            self._Q[state] = self._init_q(action_pool)
        action_idx = self.policy(state, action_pool.shape[0])
        action = action_pool[action_idx].tolist()
        self.actions.append(action)
        return action

    def fit(self, task: Task, **runner_kwargs: Any) -> EpsilonGreedyAgent:
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
            self.fit_rewards.append(self._step())
            pbar.set_description(f'Reward {self.fit_rewards[-1]:.2f} ε={self.epsilon:.3f}')
            pbar.refresh()

        runner.close()
        self.in_fit = False
        self.set_deterministic()
        return self

    def save(self, path: str | Path) -> None:
        payload = {
            'Q': self._Q,
            'N': self._N,
            'action_pool': self._action_pool,
            'n_episodes': self.n_episodes,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'decay_fraction': self.decay_fraction,
            'first_visit': self.first_visit,
        }
        with open(path, 'wb') as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> EpsilonGreedyAgent:
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        agent = cls(
            n_episodes=payload['n_episodes'],
            epsilon_start=payload['epsilon_start'],
            epsilon_end=payload['epsilon_end'],
            decay_fraction=payload['decay_fraction'],
            first_visit=payload.get('first_visit', True),
        )
        agent._Q = payload['Q']
        agent._N = payload['N']
        agent._action_pool = payload['action_pool']
        agent.set_deterministic()
        return agent

    def _step(self) -> float:
        """Every/first-visit MC update with γ=1 (undiscounted return)."""
        rewards = np.array([self.compute_reward(r) for r in self.rewards], dtype=float)
        returns = np.cumsum(rewards[::-1])[::-1]  # G_t = Σ_{k≥t} r_k (γ=1)

        seen: set[tuple[State, int]] = set()
        for i, (state_dict, action) in enumerate(zip(self.states, self.actions_idx, strict=False)):
            state = self._state_key(state_dict)
            if self.first_visit and (state, action) in seen:
                continue
            seen.add((state, action))

            if state not in self._N:
                self._N[state] = np.zeros(self._Q[state].shape[0])
            self._N[state][action] += 1

            g_t = float(returns[i])
            q_value = self._Q[state][action]
            self._Q[state][action] = q_value + (g_t - q_value) / self._N[state][action]

        return float(rewards.sum())
