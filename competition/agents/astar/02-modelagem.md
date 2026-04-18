# A* — Modelagem

## 1. Representação do estado

Usa o **`level_scene` cru** (22×22) e a posição de Mario em coordenadas absolutas — não discretiza em features booleanas como os outros agentes.

| Chave | Origem | Uso |
|---|---|---|
| `level_scene` | `Observation.level_scene` | Grafo local 22×22, Mario em [11, 11] |
| `mario_floats` | `Observation.mario_floats` | (x, y) absoluto — só para saber onde o agente está |
| `can_jump` | `Observation.may_jump` | Permite ou não incluir arestas de pulo |

### Mapeamento de tiles para o grafo

A classificação dos valores (ver `marioai/agents/utils/objects.py` e `marioai/core/sensing.py`):

| Valor no grid | Classe para A* | Efeito |
|---|---|---|
| `0` | **andável** | aresta normal (custo 1) |
| `-11` (soft) | **andável pulando** | só alcançável por pulo (custo 2) |
| `-10`, `20` (hard) | **bloqueada** | não insere nó |
| `16`, `21` (brick) | **bloqueada por baixo / andável por cima** | o topo do bloco é andável |
| `2..15` (enemy/power-up) | **andável com penalidade** | custo +100 para inimigos; +0 para power-ups (pode tratar separado) |
| `25` (projétil Mario) | **ignorado** (é próprio) | não afeta custo |
| `42` (undefined) | **bloqueada por segurança** | não expande |

## 2. Espaço de ações

A cada replanejamento, o caminho é uma sequência de células. O agente converte o **próximo passo** do caminho em macro-ação:

| Próximo delta `(Δcol, Δrow)` | Ação |
|---|---|
| `(+1, 0)` | `FORWARD` = `[0, 1, 0, 0, 0]` |
| `(+1, -1)` a `(+3, -3)` | `FORWARD_JUMP` = `[0, 1, 0, 1, 0]` |
| `(+1, -1)` com `can_jump=False` e próximo nó obrigando salto alto | `FORWARD_JUMP_SPEED` = `[0, 1, 0, 1, 1]` |
| `(-1, 0)` | `BACKWARD` = `[1, 0, 0, 0, 0]` |
| target acima e `can_jump=True` | `JUMP` = `[0, 0, 0, 1, 0]` |

Replaneja a cada 12 frames ou quando Mario se desvia da célula prevista em `level_scene[11, 11]` por mais de 1 unidade.

## 3. Função de recompensa

**N/A** — sem treino. O passthrough default em `Task.compute_reward` já serve.

## 4. Hiperparâmetros

| Parâmetro | Valor |
|---|---|
| Heurística `h(n)` | Manhattan até `(11, 21)` — borda direita da janela visível |
| Custo andar | 1 |
| Custo pular | 2 |
| Penalidade por inimigo na célula | +100 |
| Frequência de replanejamento | a cada 12 frames ou ao desviar do plano |
| Alcance máximo do pulo | 3 colunas horizontais × 3 linhas verticais |

Observação: como o campo de visão é limitado (janela 22×22), o "objetivo" nunca é a bandeira — é sempre a **borda direita visível**. O agente essencialmente faz um "seguir a margem direita", replanejando conforme novos tiles entram no campo de visão.

## 5. Protocolo de avaliação

A* é 100% determinístico dado o estado — não precisa mudar nada para virar greedy. Basta rodar `Runner(agent, task, **phase_config)` para cada fase.

## 6. Integração com o repo

- **Estender**: `marioai.core.Agent` direto. Não precisa de `MarioEnv` (não é RL) nem de `BaseAgent` (não grava trajetória).
- **Reutilizar**:
  - `self.level_scene` e `self.mario_floats` populados por `Agent.sense()` a partir do state-dict.
  - `marioai.agents.utils.objects.OBJECTS` para classificar tiles sem hardcode mágico.
- **Implementar**:
  - Função `plan(level_scene, start=(11,11), goal=(11,21))` com A* clássico (fila de prioridade `heapq`).
  - Cache do último plano; invalida em discrepância ou ao passar de 12 frames.
- **Arquivo sugerido (implementação futura)**: `marioai/agents/astar_agent.py`, classe `AStarAgent(Agent)`.
