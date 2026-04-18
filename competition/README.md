# Competição de Agentes MarioAI

Competição entre **8 agentes** que implementam técnicas distintas de tomada de decisão — do rule-based clássico a RL profundo — avaliados num mesmo conjunto de 5 fases reprodutíveis.

O foco desta pasta é **documentação e modelagem**. Cada agente tem dois markdowns:

- `01-teoria.md` — descrição didática e independente do repo.
- `02-modelagem.md` — como o agente é modelado no ambiente MarioAI (estado, ações, recompensa, treino, avaliação, integração).

A implementação dos agentes e do runner da competição será feita em PRs futuros.

## Índice dos agentes

| # | Técnica | Paradigma | Pasta |
|---|---|---|---|
| 1 | Rule-based | Heurística (sem treino) | [`agents/rule_based/`](agents/rule_based/) |
| 2 | A* | Planejamento clássico | [`agents/astar/`](agents/astar/) |
| 3 | ε-greedy com features | RL tabular simples | [`agents/epsilon_greedy/`](agents/epsilon_greedy/) |
| 4 | Monte Carlo control | RL tabular | [`agents/monte_carlo/`](agents/monte_carlo/) |
| 5 | SARSA | RL tabular on-policy | [`agents/sarsa/`](agents/sarsa/) |
| 6 | Q-learning | RL tabular off-policy | [`agents/q_learning/`](agents/q_learning/) |
| 7 | DQN | RL profundo (valor) | [`agents/dqn/`](agents/dqn/) |
| 8 | PPO | RL profundo (policy) | [`agents/ppo/`](agents/ppo/) |

## Regras da competição

### Treinamento

- Cada agente pode treinar livremente: número de episódios, recursos computacionais e hiperparâmetros ficam a critério do implementador.
- **É proibido treinar com as seeds exatas das 5 fases de avaliação** (ver tabela abaixo). O objetivo é medir generalização — o agente não deve ter visto o nível exato durante o treino. Qualquer combinação `(level_difficulty, level_type, level_seed)` diferente das 5 fases é permitida.

### Avaliação

- Cada agente roda **1 episódio por fase** (5 episódios no total) com os parâmetros fixados na tabela.
- Agentes estocásticos entram em modo determinístico:
  - Tabulares: `policy_kind = 'greedy'` (ou `epsilon = 0`).
  - `stable_baselines3`: `model.predict(obs, deterministic=True)`.
  - A* e rule-based já são determinísticos.
- O runner da competição é responsável por garantir que `Environment.level_seed`, `level_difficulty`, `level_type`, `init_mario_mode` e `time_limit` sigam exatamente a tabela antes de cada `reset()`.

## As 5 fases de avaliação

| # | Nível     | `level_difficulty` | `level_type`    | `level_seed` | `mario_mode` | `time_limit` (s) |
|---|-----------|-------------------:|-----------------|-------------:|-------------:|-----------------:|
| 1 | Fácil     | 0                  | 0 (overground)  | 1001         | 2 (fire)     | 60               |
| 2 | Média-A   | 5                  | 0 (overground)  | 2042         | 2 (fire)     | 80               |
| 3 | Média-B   | 8                  | 1 (underground) | 2077         | 2 (fire)     | 100              |
| 4 | Difícil-A | 15                 | 2 (castle)      | 3013         | 1 (large)    | 120              |
| 5 | Difícil-B | 20                 | 3 (random)      | 3099         | 0 (small)    | 120              |

A progressão de dificuldade vem de três eixos que o servidor Java expõe:

1. **`level_difficulty`**: aumenta densidade de obstáculos/inimigos na geração procedural (`ch.idsia.mario.engine.LevelGenerator`).
2. **`level_type`**: muda o bioma — `0` overground (simples), `1` underground (confinado), `2` castle (arena tipo Bowser), `3` random (mistura imprevisível).
3. **`mario_mode`**: poder inicial do Mario — `2` fire (atira bolas-de-fogo), `1` large (aguenta um hit), `0` small (um hit = morte).

A combinação garante que as fases difíceis exijam generalização: um agente que só andou em overground facilmente sofrerá num castle com Mario small.

## Critério de vitória

### Por fase
Vence a fase **todo agente** que finalizar o episódio com `FitnessResult.status == 1` (bandeira alcançada). Múltiplos agentes podem vencer a mesma fase.

### Placar global
Soma de fases vencidas (0 a 5).

### Desempate global (em ordem)
1. **Maior `time_left` médio** nas fases vencidas. O `time_left` vem do `FitnessResult` e representa segundos de jogo restantes — quem chegou mais rápido na bandeira tem mais tempo sobrando.
2. **Maior `distance` somada** em todas as 5 fases. Útil se dois agentes empatarem também no tempo (raro).

Exemplo: se os agentes A e B vencem ambos 3 fases, mas A tem `time_left` médio de 45s e B tem 38s, A vence o desempate.

## Métricas registradas por corrida

| Métrica | Origem | Papel no placar |
|---|---|---|
| `status` | `FitnessResult.status` | **Define vitória da fase** (==1) |
| `distance` | `FitnessResult.distance` | 2º desempate global |
| `time_left` | `FitnessResult.time_left` | 1º desempate global |
| `coins` | `FitnessResult.coins` | Informativo |
| `mario_mode` final | `FitnessResult.mario_mode` | Informativo (se power-up foi preservado) |
| Wall-clock | `time.time()` na implementação | Informativo (diagnóstico, não placar) |

## Protocolo de execução

Pseudocódigo do runner de avaliação (a ser implementado em etapa futura):

```python
from marioai.core import Task, Runner

PHASES = [
    dict(level_difficulty=0,  level_type=0, level_seed=1001, mario_mode=2, time_limit=60),
    dict(level_difficulty=5,  level_type=0, level_seed=2042, mario_mode=2, time_limit=80),
    dict(level_difficulty=8,  level_type=1, level_seed=2077, mario_mode=2, time_limit=100),
    dict(level_difficulty=15, level_type=2, level_seed=3013, mario_mode=1, time_limit=120),
    dict(level_difficulty=20, level_type=3, level_seed=3099, mario_mode=0, time_limit=120),
]

def evaluate(agent):
    agent.set_deterministic()  # greedy policy, epsilon=0, etc.
    task = Task()
    results = []
    for phase in PHASES:
        runner = Runner(agent, task, **phase)
        runner.run()
        results.append({
            'status': task.status,
            'distance': task.reward.get('distance', 0),
            'time_left': task.reward.get('timeLeft', 0),
            'coins': task.reward.get('coins', 0),
            'mario_mode': task.reward.get('marioMode', 0),
        })
    task.disconnect()
    return results
```

A agregação final classifica os 8 agentes pelas regras da seção "Critério de vitória".

## Referências

- [`docs/01-servidor-java.md`](../docs/01-servidor-java.md) — protocolo do servidor Java, flags de `reset`, formato das mensagens `O`/`E`/`FIT`.
- [`docs/02-cliente-python.md`](../docs/02-cliente-python.md) — arquitetura do cliente Python (Environment → Task → Agent → Runner).
- [`README.md`](../README.md) — tabela de valores do `level_scene` e formato dos sinais de controle.
