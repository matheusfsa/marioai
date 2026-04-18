# MarioAI

Python client for the [Togelius Mario AI Competition 2009](http://julian.togelius.com/mariocompetition2009/)
server. The server is Java (shipped as `.class` files under `marioai/core/server/`) and
the client talks to it over TCP.

Documentação detalhada em português:

- [`docs/01-servidor-java.md`](docs/01-servidor-java.md) — visão geral do servidor Java e do protocolo TCP
- [`docs/02-cliente-python.md`](docs/02-cliente-python.md) — arquitetura do cliente Python camada por camada
- [`docs/03-melhorias-cliente-python.md`](docs/03-melhorias-cliente-python.md) — diagnóstico e melhorias propostas

## Requirements

- Python ≥ 3.10
- Java (JDK or JRE) available on `PATH` — the client spawns the server as a subprocess

## Installation

```bash
pip install -e .
# dev extras (ruff, pytest):
pip install -e ".[dev]"
```

## CLI

The package installs a `marioai` console script with three commands:

```bash
marioai random --level_difficulty 0 --time_limit 60   # uniformly random actions
marioai mc --level_difficulty 0 --time_limit 60       # tabular Monte Carlo
marioai dqn --total_timesteps 100000                  # DQN via stable-baselines3
```

Common flags: `--level_difficulty/-ld`, `--mario_mode/-mm`, `--time_limit/-tl`,
`--max_fps/-fps`.

## Writing a custom agent

Subclass `marioai.core.Agent` (or `marioai.agents.BaseAgent` for a recorder that
stores states/actions/rewards) and implement `act`:

```python
from marioai.core import Agent, Runner, Task

class MyAgent(Agent):
    def sense(self, state):
        super().sense(state)

    def act(self):
        # [backward, forward, crouch, jump, speed/bombs]
        return [0, 1, 0, 0, 0]

    def give_rewards(self, reward, cum_reward):
        pass

task = Task()
Runner(MyAgent(), task).run()
```

## State schema

`Task.build_state` returns a dict with these keys:

| Key | Type | Meaning |
| --- | --- | --- |
| `episode_over` | `bool` | `True` once the server sends a `FIT` message |
| `can_jump` | `bool \| None` | whether Mario may jump this frame |
| `on_ground` | `bool \| None` | whether Mario is touching the ground |
| `mario_floats` | `tuple[float, float] \| None` | Mario's `(x, y)` |
| `enemies_floats` | `list[float] \| None` | flat list of enemy coordinates |
| `level_scene` | `np.ndarray \| None` | 22×22 grid of tiles/entities around Mario (center at `[11, 11]`) |
| `soft_<d>`, `hard_<d>`, `enemy_<d>`, `brick_<d>`, `projetil_<d>` | `bool \| None` | whether an object of the category is within `d` tiles |
| `has_role_near_<d>` | `bool \| None` | ground proximity feature |

### `level_scene` values

| Value | Meaning |
| --- | --- |
| -11 | soft obstacle, can jump through |
| -10 | hard obstacle, cannot pass through |
| 0 | empty |
| 1 | Mario |
| 2–10, 12, 13 | enemies (goomba, koopa, bullet bill, spiky, flower, shell …) |
| 14 | mushroom |
| 15 | fire flower |
| 16 | brick (simple / with coin / with mushroom / with flower) |
| 20 | enemy obstacle (e.g. flower pot, cannon part) |
| 21 | question brick |
| 25 | Mario's projectile |
| 42 | undefined |

## Control signals

Each position of the action list is a button; use `1` to press, `0` to release:

    [backward, forward, crouch, jump, speed/bombs]

Examples:

```python
[0, 1, 0, 0, 0]  # walk right
[1, 0, 0, 1, 0]  # jump left
```

## Development

```bash
make lint      # ruff check + ruff format --check
make format    # ruff format
make test      # pytest
```

CI runs the same three steps on push/PR (see `.github/workflows/test.yaml`).
