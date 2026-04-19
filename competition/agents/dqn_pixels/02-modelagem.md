# DQN pixels + CNN — Modelagem

## 1. Representação do estado

**Entrada da política**: frames capturados da janela do servidor Java,
pré-processados no pipeline Atari-clássico:

1. `GameWindowCapture(window_title='Mario', grayscale=True, resize=(84, 84))`
   — captura via `mss`, grayscale + resize via OpenCV (`INTER_AREA`).
2. `ShapedPixelMarioEnv` — override de `_build_observation` devolve o frame
   capturado em vez do grid 22×22; o TCP continua sendo a fonte de verdade
   para episódio terminado / reward.
3. `gym.wrappers.FrameStack(env, num_stack=4)` — tensor final `(4, 84, 84) uint8`.

Código de setup:

```python
from gym.wrappers import FrameStack
from marioai.capture import GameWindowCapture
from marioai.gym import ShapedPixelMarioEnv

capture = GameWindowCapture('Mario', grayscale=True, resize=(84, 84))
env = ShapedPixelMarioEnv(capture=capture,
                          level_difficulty=3, level_type=0, level_seed=42,
                          mario_mode=2, time_limit=100, max_fps=720)
env = FrameStack(env, num_stack=4)
```

## 2. Espaço de ações

`Discrete(14)` — exatamente as 14 combinações de `marioai/gym/environment.py:14-30`
(idênticas ao DQN simbólico). Nenhum filtro dinâmico por `can_jump`: a rede
aprende por si a penalizar o botão de pulo quando não pode pular.

## 3. Função de recompensa

`ShapedMarioEnv.compute_reward` (herdado por `ShapedPixelMarioEnv`) — fórmula
idêntica à do DQN simbólico:

```
r = (distance_t − distance_{t-1})
  + (coins_t − coins_{t-1}) × 10
  + 100 se status == 1 (flag)
  − 50  se status == 2 (morte)
```

Ver `marioai/gym/shaped_environment.py`.

## 4. Hiperparâmetros e regime de treino

Defaults do script `train.py` (todos sobrescritíveis via flag):

| Parâmetro                  | Default | Flag |
|---|---:|---|
| `total_timesteps`          | 200 000 | `--total-timesteps` |
| `learning_rate`            | 1e-4    | `--learning-rate` |
| `buffer_size` (replay)     | 50 000  | `--buffer-size` |
| `learning_starts`          | 5 000   | `--learning-starts` |
| `batch_size`               | 32      | `--batch-size` |
| `γ` (discount)             | 0.99    | `--gamma` |
| `exploration_initial_eps`  | 1.0     | — |
| `exploration_final_eps`    | 0.05    | `--exploration-final-eps` |
| `exploration_fraction`     | 0.5     | `--exploration-fraction` |
| `target_update_interval`   | 1 000   | `--target-update-interval` |
| `train_freq`               | 4       | — |
| Política                   | `CnnPolicy` (NatureCNN) | — |
| Frame-stack                | 4       | — |
| Seed                       | 42      | `--seed` |

**Tempo estimado**: ~3-5h em CPU para 200k steps com `max_fps=720` no
servidor. Aumentar `total_timesteps` para ≥ 1M reproduz melhor os resultados
de Atari mas exige dezenas de horas wall-clock (sem paralelização
possível — porta 4242 fixa).

**Nível de treino**: `level_difficulty=3, level_type=0, level_seed=42`.
Diferente das 5 seeds de avaliação (1001, 2042, 2077, 3013, 3099), como
exige a regra de generalização do `competition/README.md`.

## 5. Protocolo de avaliação

```python
from marioai.core import Runner, Task
from marioai.agents import DqnPixelsAgent
from marioai.capture import GameWindowCapture

capture = GameWindowCapture('Mario', grayscale=True, resize=(84, 84))
agent   = DqnPixelsAgent('competition/agents/dqn_pixels/dqn_pixels.zip')

task = Task()
runner = Runner(agent, task, capture=capture,
                level_difficulty=0, level_type=0, level_seed=1001,
                mario_mode=2, time_limit=60)
runner.run()
runner.close()
```

`DqnPixelsAgent` encapsula o `DQN.load(...)` e implementa a interface
`Agent`:

- `observe_frame(frame)` — pré-processa (se preciso) e empilha no deque de 4.
- `act()` — empilha o deque em `(4, 84, 84)`, chama
  `model.predict(..., deterministic=True)` e devolve `ACTIONS[idx]`.
- `set_deterministic()` — no-op (já é deterministic por padrão).

Quando a Etapa 1 do roadmap (`CompetitionRunner`) estiver pronta, o agente
plugará exatamente da mesma forma: o runner instancia a captura, passa via
kwarg, e roda as 5 fases sem mudança no agente.

## 6. Integração com o repo

- **Estende**: `ShapedMarioEnv` (reward shaping) e `gym.Wrapper` via `FrameStack`.
- **Reutiliza**:
  - `marioai.gym.MarioEnv` — tradução core→gym, mapeamento de ações.
  - `marioai.capture.GameWindowCapture` — captura da janela.
  - `marioai.core.Experiment.observe_frame` — pipeline frame→agent fora do
    mundo gym (útil para o `CompetitionRunner`).
- **Implementa**:
  - `marioai/gym/pixel_environment.py::ShapedPixelMarioEnv` — observation = frame.
  - `marioai/agents/dqn_pixels_agent.py::DqnPixelsAgent` — wrapper do DQN
    carregado, para interface `Agent`.
  - `competition/agents/dqn_pixels/train.py` — script de treino CLI.
- **Limitações conhecidas**:
  - 1 env simultâneo (servidor Java com porta 4242 fixa).
  - A captura é sensível à visibilidade da janela — em Windows pode-se usar
    `--capture-backend win32` para tolerar janelas cobertas/minimizadas
    (ver [`docs/04-captura-janela.md`](../../../docs/04-captura-janela.md)).
  - Treino é mais lento que o DQN simbólico (rede maior + overhead de captura).

## 7. Como treinar

```
python -m competition.agents.dqn_pixels.train \
    --total-timesteps 200000 \
    --save-path competition/agents/dqn_pixels/dqn_pixels.zip
```

O script:

1. Constrói a pilha `GameWindowCapture → ShapedPixelMarioEnv → FrameStack`.
2. Instancia `DQN('CnnPolicy', env, **hp)`.
3. Chama `.learn(total_timesteps, log_interval=10)`.
4. Salva o modelo no caminho indicado.
