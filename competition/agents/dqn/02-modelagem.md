# DQN — Modelagem

## 1. Representação do estado

Usa o **grid 22×22** do `MarioEnv` (`marioai/gym/environment.py:36-98`), que já remapeia os valores para caber em `Box(0, 26)`:

| Valor original | Valor remapeado |
|---|---|
| 25 (projétil Mario) | 22 |
| −11 (soft obstacle) | 23 |
| −10 (hard obstacle) | 24 |
| 42 (undefined) | 25 |
| posição do Mario | 26 (força a presença no centro) |

Para dar noção de movimento, **empilhar últimos 4 frames** via wrapper Gym `FrameStack(env, n=4)` → tensor `(4, 22, 22)`.

## 2. Espaço de ações

`Discrete(14)` — exatamente as 14 combinações definidas em `marioai/gym/environment.py:14-30`, idênticas às do `Task._action_pool`.

Não há filtro dinâmico por `can_jump` neste agente; o DQN precisa aprender a ignorar `jump` quando não pode pular, penalizado implicitamente pelo ambiente (ação de pulo inválido ≡ ação default em frames sem privilégio de pulo).

## 3. Função de recompensa

Override de `MarioEnv.compute_reward` (`marioai/gym/environment.py:121-124`):

```python
class ShapedMarioEnv(MarioEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prev_distance = 0.0
        self._prev_coins = 0

    def compute_reward(self, reward_data):
        distance = reward_data.get('distance', 0.0) or 0.0
        coins    = reward_data.get('coins', 0) or 0
        status   = reward_data.get('status')
        # Δx por passo + ganho de moeda + terminal
        r = (distance - self._prev_distance) * 1.0
        r += (coins - self._prev_coins) * 10.0
        if status == 1:
            r += 100.0
        elif status == 2:
            r -= 50.0
        self._prev_distance = distance
        self._prev_coins = coins
        return r

    def reset(self):
        self._prev_distance = 0.0
        self._prev_coins = 0
        return super().reset()
```

## 4. Hiperparâmetros e regime de treino

| Parâmetro | Valor |
|---|---|
| `total_timesteps` | 500 000 |
| `learning_rate` | 1e-4 |
| `buffer_size` (replay) | 100 000 |
| `learning_starts` | 10 000 |
| `batch_size` | 32 |
| `γ` | 0.99 |
| `exploration_initial_eps` | 1.0 |
| `exploration_final_eps` | 0.05 |
| `exploration_fraction` | 0.5 (ε atinge min em 50% do treino) |
| `target_update_interval` | 1 000 passos |
| Política | `CnnPolicy` (SB3 default para inputs de imagem) |
| Arquitetura custom | Conv(32, 3×3) → Conv(64, 3×3) → FC(256) → FC(14) |
| Frame-stack | 4 |

## 5. Protocolo de avaliação

```python
from stable_baselines3 import DQN

model = DQN.load('dqn_mario.zip')
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(int(action))
```

`deterministic=True` força `argmax` sobre Q-values. Sem exploração na avaliação.

## 6. Integração com o repo

- **Estender**: `gym.Wrapper` sobre `MarioEnv` para o shaping de reward e frame-stack.
- **Reutilizar**:
  - `marioai.gym.MarioEnv` — toda a tradução core→gym.
  - `marioai.cli.py` função `dqn` (linhas 45-85 aprox.) como template de integração com `stable_baselines3.DQN`.
- **Implementar**:
  - Classe `ShapedMarioEnv(MarioEnv)` com override de `compute_reward` e tracking de `prev_distance`/`prev_coins`.
  - Wrapper `FrameStack(env, n=4)` (pode-se usar `gym.wrappers.FrameStack` direto).
  - Script de treino `train_dqn.py` que instancia `DQN(policy='CnnPolicy', env, **hyperparams).learn(500_000)` e salva `dqn_mario.zip`.
- **Limitação**: o servidor Java usa porta 4242 fixa — não é possível instanciar múltiplos envs paralelos (`SubprocVecEnv`) até o servidor suportar porta configurável.
- **Arquivo sugerido**: `marioai/agents/dqn_agent.py` (wrapper + utilitário `train()`) e `marioai/gym/shaped_environment.py` (ambiente com reward shaping).
