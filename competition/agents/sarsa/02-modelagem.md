# SARSA — Modelagem

## 1. Representação do estado

Mesma tupla de 8 booleanas (consistente com ε-greedy, Monte Carlo e Q-learning):

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

Usar `State` de `marioai/agents/monte_carlo_agent.py:14-52` para hashabilidade.

## 2. Espaço de ações

**14 combinações** do `Task._action_pool`, filtradas via `Task.filter_actions()` quando `can_jump=False`.

## 3. Função de recompensa

Shaping mais denso que o MC (TD se beneficia de sinal por passo):

```python
def compute_reward(reward_data):
    # no fim do episódio (status presente)
    if reward_data.get('status') is not None and 'timeLeft' in reward_data:
        r = 0.0
        r += 100.0 if reward_data['status'] == 1 else 0.0   # venceu
        r -= 50.0 if reward_data['status'] == 2 else 0.0    # morreu
        r += 10.0 * reward_data.get('coins', 0)
        return r
    # passo normal: progresso em x
    if self.mario_floats is None:
        return 0.0
    dx = self.mario_floats[0] - self.prev_x
    self.prev_x = self.mario_floats[0]
    return dx * 0.1
```

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

Variar dificuldade/seeds de treino (nunca usar as seeds das fases 1001/2042/2077/3013/3099).

## 5. Protocolo de avaliação

- `self.epsilon = 0.0` e `self.policy_kind = 'greedy'` antes das 5 fases.
- Para estados novos (nunca vistos), inicializar `Q` com bias em `FORWARD` (ação 6) em vez de zeros puros — evita travamento por `argmax([0, ..., 0]) = 0 = do nothing` em cenários não treinados.

## 6. Integração com o repo

- **Estender**: `marioai.agents.base_agent.BaseAgent`.
- **Reutilizar**:
  - Classe `State` de `marioai/agents/monte_carlo_agent.py:14-52`.
  - Estrutura de `filter_actions()` override se quiser (`monte_carlo_agent.py:100-106`).
- **Implementar o loop on-policy** (diferente do `fit()` do MC, que é off-line batch):

```python
def fit(self, task, n_episodes=3000, **runner_kwargs):
    runner = Runner(self, task, **runner_kwargs)
    for ep in range(n_episodes):
        task.reset()
        s = hashable_state(task.state)
        a = self.policy(s)         # ε-greedy
        while not task.finished:
            task.perform_action(self._action_pool[a])
            task.get_sensors()
            r = self.compute_reward(task.reward)
            s_prime = hashable_state(task.state)
            a_prime = self.policy(s_prime)
            target = r + self.gamma * self.Q.get((s_prime, a_prime), 0.0)
            self.Q[(s, a)] += self.alpha * (target - self.Q.get((s, a), 0.0))
            s, a = s_prime, a_prime
        self.decay_epsilon()
```

- **Arquivo sugerido**: `marioai/agents/sarsa_agent.py`, classe `SARSAAgent(BaseAgent)`.
- **Atenção**: `Runner.run()` encapsula um episódio completo com `act()`/`perform_action()`/`get_sensors()` — ou escrever um loop custom (como acima) ou usar hooks internos de `BaseAgent` em `give_rewards()` para fazer o update TD.
