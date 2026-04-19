# DQN com pixels + CNN — Teoria

Variante "visual" do DQN (ver [`../dqn/01-teoria.md`](../dqn/01-teoria.md) para a base
algorítmica). Aqui a **observação é a imagem renderizada da janela do jogo** —
não o grid simbólico 22×22 exposto pelo servidor via TCP — e a rede é a
**NatureCNN** de Mnih et al. 2015 (*"Human-level control through deep reinforcement learning"*, Nature 518).

## Por que pixels + CNN

O grid 22×22 é um resumo já curado do estado (server-side). Aprender com pixels:

1. **Testa generalização visual**: o agente precisa descobrir sozinho o que é
   tile sólido, inimigo, power-up, moeda, distância ao flag — informação que
   o grid entrega de graça. É o benchmark clássico dos papers.
2. **Simula imitação humana**: um agente que funciona só com o que o jogador
   humano vê na tela pode ser treinado por behavioural cloning a partir de
   vídeo, sem acesso ao estado interno.
3. **Desacopla do servidor**: se o protocolo TCP mudar ou um segundo "ambiente"
   for introduzido (emulador, fork), o agente continua funcionando desde que
   haja janela renderizada.

## Pipeline de observação

O pipeline segue o padrão Atari DQN:

```
janela do jogo (ex.: 800x600 RGB)
  ↓ GameWindowCapture(grayscale=True, resize=(84,84))
frame 84x84 uint8
  ↓ FrameStack(num_stack=4)
observação (4, 84, 84) uint8
  ↓ NatureCNN
features (512,)
  ↓ Linear → |A|=14
Q-values
```

**Grayscale**: descarta cor (perdemos distinção Mario-vermelho × fogo-laranja,
mas reduz parâmetros 3×). **84×84**: downscale para caber em memória; detalhes
finos do sprite viram texturas genéricas, o que paradoxalmente ajuda a
generalização. **FrameStack 4**: dá noção de movimento implícita (velocidade,
direção) — crucial porque um único frame estático é um POMDP.

## NatureCNN

A arquitetura padronizada por Mnih et al. 2015, usada como `CnnPolicy` default
no stable-baselines3:

```
input: (4, 84, 84) uint8, normalizado para [0, 1]
  → Conv2D(32, kernel=8, stride=4) + ReLU    # (32, 20, 20)
  → Conv2D(64, kernel=4, stride=2) + ReLU    # (64,  9,  9)
  → Conv2D(64, kernel=3, stride=1) + ReLU    # (64,  7,  7)
  → Flatten                                   # 3136
  → Linear(512) + ReLU
  → Linear(|A|=14)                            # Q-values
```

Tamanho: ~1.7M parâmetros. Roda em CPU (~5-15 FPS de inferência); uma GPU
modesta triplica isso facilmente.

## Desafios específicos vs. DQN simbólico

| Desafio | Mitigação |
|---|---|
| Sample efficiency baixa — pixels exigem ≫ transições | Shaping de reward denso (Δdist + coins + terminal); FrameStack; `exploration_fraction=0.5` |
| Captura de janela pode falhar momentaneamente | `observe_frame(None)` preserva o deque; warning após 5 frames perdidos |
| Sem `SubprocVecEnv` (porta 4242 fixa no servidor) | 1 env; documentado como limitação |
| Sincronização entre frame e estado TCP | Captura é feita *após* `get_sensors()` retornar — o servidor já processou a ação |
| Janela pode ser arrastada durante treino | `GameWindowCapture` re-lê bbox a cada 30 frames |

## Relação com o DQN simbólico da competição

Os dois agentes (`dqn` e `dqn_pixels`) dividem:

- **Espaço de ações** (14 discretas, em `marioai/gym/environment.py:14-30`).
- **Reward shaping** (`ShapedMarioEnv`, em `marioai/gym/shaped_environment.py`).
- **Protocolo de avaliação** (greedy via `deterministic=True`).

E diferem em:

- **Observação**: grid 22×22 × 4 (simbólico) vs. 84×84 × 4 grayscale (pixels).
- **Política**: `MlpPolicy` vs. `CnnPolicy` (NatureCNN).
- **Sample efficiency**: o simbólico vence fases mais cedo (menos steps), mas
  o pixels é o único dos dois que funcionaria num cenário de imitação visual.

O relatório final (`competition/RESULTS.md`, Etapa 5) compara os dois lado a
lado — é a parte mais interessante do placar.

## Referências

- Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518(7540), 529–533.
- Mnih, V., et al. (2013). *Playing Atari with deep reinforcement learning*. NeurIPS DL workshop.
- Raffin, A., et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. JMLR 22(268). `CnnPolicy` = NatureCNN default.
