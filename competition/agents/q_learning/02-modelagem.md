# Q-Learning — Modelagem

## 1. Representação do estado

Mesma tupla de 8 booleanas dos demais tabulares:

```python
state = (
    bool(obs['enemy_1']),
    bool(obs['enemy_2']),
    bool(obs['hard_1']),
    bool(obs['soft_1']),
    bool(obs['has_role_near_1']),
    bool(obs['has_role_near_2']),
    bool(obs['can_jump']),
    bool(obs['on_ground']),
)
```

Wrapping via `State` (`marioai/agents/monte_carlo_agent.py:14-52`).

## 2. Espaço de ações

**14 combinações** do `Task._action_pool` com filtro via `Task.filter_actions()` quando `can_jump=False`.

## 3. Função de recompensa

Shaping denso com ênfase em progresso e vitória:

```python
def compute_reward(reward_data):
    # fim de episódio
    if reward_data.get('status') is not None and 'timeLeft' in reward_data:
        r = 0.0
        r += 100.0 if reward_data['status'] == 1 else 0.0  # venceu
        r -= 50.0 if reward_data['status'] == 2 else 0.0   # morreu
        r += 10.0 * reward_data.get('coins', 0)
        return r
    # passo normal
    if self.mario_floats is None:
        return 0.0
    dx = self.mario_floats[0] - self.prev_x
    self.prev_x = self.mario_floats[0]
    return dx * 0.1
```

Idêntico ao do SARSA — permite comparação direta on-policy vs. off-policy.

## 4. Hiperparâmetros e regime de treino

| Parâmetro | Valor |
|---|---|
| Nº de episódios | 3000 |
| `α` (learning rate) | 0.1 |
| `γ` (discount) | 0.95 |
| `ε` inicial | 1.0 |
| `ε` mínimo | 0.05 |
| Schedule de `ε` | Linear, atinge `ε_min` em 80% dos episódios |
| Duração estimada | ~1h em CPU |

Diversificar seeds de treino (≠ das 5 fases).

## 5. Protocolo de avaliação

- `self.epsilon = 0.0` e `self.policy_kind = 'greedy'` antes das 5 fases.
- Inicializar `Q` novo com bias positivo em `FORWARD` para lidar com estados não vistos no treino.

## 6. Integração com o repo

- **Estender**: `marioai.agents.base_agent.BaseAgent`.
- **Reutilizar**:
  - Classe `State` de `marioai/agents/monte_carlo_agent.py:14-52`.
  - `Task._action_pool` e `Task.filter_actions`.
- **Implementar loop off-policy**:

```python
def fit(self, task, n_episodes=3000, **runner_kwargs):
    runner = Runner(self, task, **runner_kwargs)
    for ep in range(n_episodes):
        task.reset()
        s = hashable_state(task.state)
        while not task.finished:
            a = self.policy(s)          # ε-greedy
            task.perform_action(self._action_pool[a])
            task.get_sensors()
            r = self.compute_reward(task.reward)
            s_prime = hashable_state(task.state)
            max_q_prime = max(self.Q.get((s_prime, a_), 0.0)
                              for a_ in range(len(self._action_pool)))
            self.Q[(s, a)] += self.alpha * (
                r + self.gamma * max_q_prime - self.Q.get((s, a), 0.0)
            )
            s = s_prime
        self.decay_epsilon()
```

Note a diferença crucial do SARSA: o target é `r + γ · max_{a′} Q(s′, a′)`, não `r + γ · Q(s′, a′_amostrado)`.

- **Arquivo sugerido**: `marioai/agents/q_learning_agent.py`, classe `QLearningAgent(BaseAgent)`.
