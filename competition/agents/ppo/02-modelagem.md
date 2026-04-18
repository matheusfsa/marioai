# PPO — Modelagem

## 1. Representação do estado

Igual ao DQN: grid 22×22 do `MarioEnv` (valores já remapeados para `Box(0, 26)` — ver `marioai/gym/environment.py:80-91`) com frame-stack de 4 → tensor `(4, 22, 22)`.

A rede compartilha o trunk CNN entre policy head (softmax sobre 14 ações) e value head (escalar).

## 2. Espaço de ações

`Discrete(14)` do `MarioEnv.ACTIONS` (`marioai/gym/environment.py:14-30`).

Sem filtro por `can_jump` — PPO aprende a evitar ações de pulo inválidas implicitamente.

## 3. Função de recompensa

Idêntica ao DQN para permitir comparação direta value-based × policy-based. Override de `MarioEnv.compute_reward`:

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
        r  = (distance - self._prev_distance) * 1.0
        r += (coins - self._prev_coins) * 10.0
        if status == 1: r += 100.0
        elif status == 2: r -= 50.0
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
| `n_steps` (rollout) | 2 048 |
| `n_epochs` (SGD por rollout) | 4 |
| `batch_size` | 64 |
| `learning_rate` | 3e-4 |
| `γ` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |
| `ent_coef` | 0.01 |
| `vf_coef` | 0.5 |
| `max_grad_norm` | 0.5 |
| Política | `CnnPolicy` (SB3 default) |
| Nº de envs paralelos | **1** (limitação da porta 4242 fixa) |

## 5. Protocolo de avaliação

```python
from stable_baselines3 import PPO

model = PPO.load('ppo_mario.zip')
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(int(action))
```

`deterministic=True` força a moda da distribuição categórica (a ação com maior probabilidade). Sem amostragem.

## 6. Integração com o repo

- **Estender**: mesmo wrapper `ShapedMarioEnv` do DQN (pode ser compartilhado).
- **Reutilizar**:
  - `marioai.gym.MarioEnv`.
  - `marioai.cli.py` função `dqn` como referência de integração com stable-baselines3 (o padrão é idêntico, só muda a classe).
- **Implementar**:
  - Script `train_ppo.py` que instancia `PPO(policy='CnnPolicy', env=ShapedMarioEnv_stacked, **hyperparams).learn(500_000)`.
- **Limitações conhecidas**:
  - Single-env é subótimo para PPO — a maior parte das publicações usa 8+ workers paralelos. Em Mario, a porta TCP fixa do servidor (`marioai/core/environment.py:144-192`) impede `SubprocVecEnv`. Possíveis soluções: (a) patchar o servidor Java para aceitar `-port N`; (b) rodar várias instâncias em containers isoladas. Ambas fogem do escopo atual.
  - `ent_coef=0.01` é agressivo; em caso de colapso prematuro da política, subir para 0.05.
- **Arquivo sugerido**: `marioai/agents/ppo_agent.py` (wrapper + utilitário `train()`).
