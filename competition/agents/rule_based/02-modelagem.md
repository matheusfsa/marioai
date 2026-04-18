# Rule-based — Modelagem

## 1. Representação do estado

Usa **6 features booleanas** extraídas pelo `Task.build_state()` (`marioai/core/task.py:134-178`) — não precisa olhar o `level_scene` bruto:

| Feature | Origem | Significado |
|---|---|---|
| `enemy_1` | `sensing.is_near(level_scene, [2..15], dist=1)` | Há inimigo/power-up na coluna +1 à frente |
| `enemy_2` | `sensing.is_near(..., dist=2)` | Há inimigo na coluna +2 à frente |
| `hard_1` | `sensing.is_near(..., [20, -10], dist=1)` | Há cano/muro duro imediatamente à frente |
| `has_role_near_1` | `sensing.has_role_near(..., dist=1)` | Coluna +1 é um buraco (chão livre abaixo) |
| `has_role_near_2` | `sensing.has_role_near(..., dist=2)` | Coluna +2 é um buraco |
| `can_jump` | campo do `Observation` | Mario pode pular neste frame |

Todas as features são recalculadas automaticamente a cada chamada de `Task.get_sensors()`.

## 2. Espaço de ações

Subconjunto de 5 combinações (das 14 em `Task._action_pool`, `marioai/core/task.py:57-75`):

| Nome | Vetor `[back, fwd, crouch, jump, speed]` |
|---|---|
| `FORWARD` | `[0, 1, 0, 0, 0]` |
| `FORWARD_JUMP` | `[0, 1, 0, 1, 0]` |
| `FORWARD_JUMP_SPEED` | `[0, 1, 0, 1, 1]` |
| `JUMP` (no lugar) | `[0, 0, 0, 1, 0]` |
| `BACKWARD` | `[1, 0, 0, 0, 0]` |

Não precisa chamar `Task.filter_actions()` porque as regras já condicionam o pulo em `can_jump=True`.

## 3. Função de recompensa

**N/A** — agente não treina. O `compute_reward` padrão de `Task` (passthrough do dict de fitness) pode ser mantido; as métricas só são lidas no fim do episódio para o placar.

## 4. Hiperparâmetros e regime de treino

Não há treino. A única "configuração" é a prioridade das regras, codificada diretamente em `act()`:

```python
def act(self):
    # prioridades (de cima para baixo)
    if self.state['enemy_1'] and self.can_jump:
        return FORWARD_JUMP_SPEED
    if self.state['has_role_near_1'] and self.can_jump:
        return FORWARD_JUMP_SPEED  # pula buraco correndo (mais alcance)
    if self.state['hard_1'] and self.can_jump:
        return FORWARD_JUMP
    if self.state['hard_1'] and not self.can_jump:
        return BACKWARD  # recua um pouco até poder pular de novo
    if self.state['enemy_2'] and self.can_jump:
        return FORWARD_JUMP  # pula preventivo
    return FORWARD
```

## 5. Protocolo de avaliação

Agente já é determinístico — nenhum flag extra é necessário. Cada execução das 5 fases produz o mesmo resultado dado a mesma seed.

## 6. Integração com o repo

- **Estender**: `marioai.core.Agent` ou `marioai.agents.exploratory_agent.ExploratoryAgent` (já calcula as features proximais em `_build_state()`, `marioai/agents/exploratory_agent.py:31-57`).
- **Reutilizar**:
  - `marioai.core.sensing.is_near`, `has_role_near` — para calcular features se preferir extender `Agent` direto.
  - `marioai.agents.exploratory_agent.ExploratoryAgent._build_state` — já chama `is_near`/`has_role_near` e popula `self.state`.
- **Override**: apenas `act()` — substitui o `[0, 1, 0, random, random]` do `ExploratoryAgent`.
- **Arquivo sugerido (implementação futura)**: `marioai/agents/rule_based_agent.py`, classe `RuleBasedAgent(ExploratoryAgent)`.
