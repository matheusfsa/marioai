# ε-greedy com features — Modelagem

## 1. Representação do estado

Tupla de **8 booleanas** — 256 estados no pior caso:

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

Todas essas chaves são produzidas por `Task.build_state()` (`marioai/core/task.py:165-176`) a partir de `sensing.is_near` e `sensing.has_role_near`.

Para hash eficiente, pode-se reutilizar a classe `State` de `marioai/agents/monte_carlo_agent.py:14-52` (aceita numpy arrays, tuplas, e tipos primitivos).

## 2. Espaço de ações

Todas as **14 combinações** do `Task._action_pool` (`marioai/core/task.py:57-75`), filtradas via `Task.filter_actions()` quando `can_jump=False` (remove as 7 ações com `jump=1`).

## 3. Função de recompensa

Shaping **terminal-aware**:

```python
def compute_reward(reward_data):
    r = 0.0
    if 'distance' in reward_data and reward_data.get('status') is not None:
        # mensagem FIT (fim de episódio)
        r += 50.0 if reward_data['status'] == 1 else 0.0  # venceu
        r -= 20.0 if reward_data['status'] == 2 else 0.0  # morreu
        r += 2.0 * reward_data.get('coins', 0)
        return r
    # passo normal: usa Δx
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
| `ε` inicial | 1.0 |
| `ε` mínimo | 0.05 |
| Schedule de `ε` | Linear, atinge `ε_min` em 80% dos episódios (`Δε = (1 − 0.05) / 2400`) |
| `γ` | **1.0** (sem desconto, diferencia do MC) |
| Atualização | First-visit ou every-visit — ambas aceitas |
| Variância de treino | Níveis com `level_difficulty ∈ {0..10}` e `level_type ∈ {0, 1}`, seeds ≠ das 5 fases de teste |
| Duração esperada | ~1 episódio/s → ~50 min de treino |

## 5. Protocolo de avaliação

- Fixar `ε = 0` antes das 5 fases (força greedy).
- Para estados não vistos durante treino (Q ainda zerado), a política cai no `argmax` do vetor zero → ação 0 (`do nothing`). Para evitar isso, inicializar `Q` novo como `[0, ..., 0]` mas com um bias positivo em `FORWARD` (ação 6) — força progressão como fallback razoável em estados desconhecidos.

## 6. Integração com o repo

- **Estender**: `marioai.agents.base_agent.BaseAgent` (herda de `Agent` e já registra `self.states`, `self.actions`, `self.rewards` a cada passo).
- **Reutilizar**:
  - Classe `State` de `marioai/agents/monte_carlo_agent.py:14-52`.
  - `Task._action_pool` e `Task.filter_actions` (via `self.task` na classe).
  - Estrutura do loop `fit()` de `monte_carlo_agent.py:134-155` como template (apenas trocar `γ` por 1 e remover desconto).
- **Implementar**:
  - Método `policy(state)` com ε-greedy sobre `Q[state]`.
  - Método `fit(task, n_episodes=3000)` que faz o loop de treino e decai `ε`.
  - Método `set_deterministic()` que fixa `ε = 0`.
- **Arquivo sugerido**: `marioai/agents/epsilon_greedy_agent.py`, classe `EpsilonGreedyAgent(BaseAgent)`.
