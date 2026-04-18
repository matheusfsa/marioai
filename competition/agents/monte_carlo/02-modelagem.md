# Monte Carlo Control — Modelagem

## 1. Representação do estado

Mesma tupla de 8 booleanas do ε-greedy (para permitir comparação direta):

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

Wrapping via `State` hashable de `marioai/agents/monte_carlo_agent.py:14-52` (já aceita numpy arrays e listas).

## 2. Espaço de ações

Todas as **14 combinações** do `Task._action_pool`, filtradas via `Task.filter_actions()` quando `can_jump=False`.

## 3. Função de recompensa

Usa o shaping **já implementado** em `marioai/agents/monte_carlo_agent.py:89-98`:

```python
def compute_reward(self, reward_data):
    if reward_data.get('status') == 1:
        return reward_data['distance'] * 2   # bônus grande ao vencer
    if 'distance' in reward_data:
        return reward_data['distance'] * 0.1  # reward residual no FIT
    if self.mario_floats is None:
        return 0
    dist = self.mario_floats[0] - self.actual_x
    self.actual_x = self.mario_floats[0]
    return dist * 0.01  # Δx por passo
```

## 4. Hiperparâmetros e regime de treino

| Parâmetro | Valor |
|---|---|
| `n_samples` (episódios) | 2000 |
| `discount` (γ) | 0.9 |
| `min_epsilon` | 0.1 |
| `reward_threshold` inicial | 0.0 |
| `reward_increment` | 0.5 |
| `ε`-schedule | Adaptativo: decai `1/n_samples` toda vez que `reward ≥ reward_threshold`, e aumenta `reward_threshold` em `reward_increment` (ver `monte_carlo_agent.py:144-149`) |
| Duração estimada | ~3000 passos por episódio × 2000 episódios ≈ 1h em CPU |

## 5. Protocolo de avaliação

- Ao fim do `fit()`, o agente já faz `self.policy_kind = 'greedy'` (`monte_carlo_agent.py:154`) — pronto para avaliação.
- Para reset adicional entre fases, chamar `agent.reset()` antes de cada `Runner.run()`.

## 6. Integração com o repo

**Está já implementado**: `marioai/agents/monte_carlo_agent.py:55-173`. Para a competição, basta instanciar com os hiperparâmetros acima:

```python
from marioai.agents import MonteCarloAgent
from marioai.core import Task

agent = MonteCarloAgent(
    n_samples=2000,
    discount=0.9,
    min_epsilon=0.1,
    reward_threshold=0.0,
    reward_increment=0.5,
)
task = Task()
agent.fit(
    task=task,
    level_difficulty=3,   # diferente das fases de teste
    level_seed=42,        # diferente das fases de teste
    level_type=0,
    mario_mode=2,
    time_limit=60,
)
# agent.policy_kind é 'greedy' automaticamente após fit
```

**Pontos de atenção**:

- O `fit()` roda o `Runner` internamente (`monte_carlo_agent.py:139`). Isso abre um servidor Java persistente enquanto dura o treino — a avaliação nas 5 fases **precisa reconectar** (novo `Task()`).
- Diversificar as seeds de treino para melhorar generalização — treinar em (difficulty ∈ {0, 3, 7}, type ∈ {0, 1}, seeds aleatórias ≠ das fases).
- O compute_reward atual só retorna shaping positivo — considerar adicionar penalidade por morte (`status == 2` → `-20`) para acelerar aprendizado de evitar inimigos.
