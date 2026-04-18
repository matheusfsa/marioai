# Cliente Python

O cliente Python é o que dá vida aos agentes deste repo. Ele **sobe o servidor Java em subprocess**, abre um socket TCP e conversa com ele num protocolo text-based (descrito em [`01-servidor-java.md`](./01-servidor-java.md)). O cliente também expõe uma hierarquia de abstrações para escrever agentes em Python, um wrapper `gym.Env` e uma CLI.

Essa doc descreve **o que está implementado hoje** (estado atual do código). Uma análise crítica do que não está bom está em [`03-melhorias-cliente-python.md`](./03-melhorias-cliente-python.md).

---

## Estrutura do pacote

```
marioai/
├── __init__.py               # vazio
├── core/
│   ├── __init__.py           # exporta Agent, Environment, Experiment, Runner, Task
│   ├── agent.py              # classe base Agent (interface)
│   ├── environment.py        # Environment (subprocess do Java + orquestração) + TCPClient (socket cru)
│   ├── utils.py              # parsing das mensagens O/E/FIT
│   ├── task.py               # Task: bridge entre Environment e Agent
│   ├── experiment.py         # loop de episódio
│   ├── runner.py             # orquestração high-level
│   └── server/               # .class do servidor Java (não é código Python)
├── agents/
│   ├── __init__.py           # exporta BaseAgent, ExploratoryAgent, MonteCarloAgent, RandomAgent
│   ├── base_agent.py         # BaseAgent — registra states/actions/rewards
│   ├── random_agent.py       # RandomAgent — baseline
│   ├── exploratory_agent.py  # ExploratoryAgent — constrói features do level_scene
│   ├── monte_carlo_agent.py  # MonteCarloAgent — Q-table com MC control
│   └── utils/
│       └── objects.py        # OBJECTS: mapeamento id → descrição
├── config/                   # ⚠️ módulo não usado em lugar nenhum, import quebrado
└── gym/
    ├── __init__.py           # ⚠️ exporta Environment (nome errado)
    └── environment.py        # MarioEnv — wrapper gym.Env
```

No root do repo:

- `main.py` — CLI Click com `random`, `mc`, `dqn`
- `requirements.txt`, `requirements-dev.txt`, `Makefile`, `.flake8`, `.github/workflows/test.yaml`
- `server.bat` — para subir o server manualmente no Windows

---

## As camadas, de baixo pra cima

O cliente é organizado em camadas bem claras. Uma ação do agente flui do topo pro fundo; uma observação flui do fundo pro topo.

```
   ┌──────────────────────────────────────────┐
   │ Agente (RandomAgent, MonteCarloAgent...) │
   └──────────────┬───────────────────────────┘
                  │ act() / sense()
                  ▼
   ┌──────────────────────────────────────────┐
   │ Task — filtra ações, monta state dict    │
   └──────────────┬───────────────────────────┘
                  │ perform_action / get_sensors
                  ▼
   ┌──────────────────────────────────────────┐
   │ Environment — sobe JVM, orquestra TCP    │
   └──────────────┬───────────────────────────┘
                  │ send / recv bytes
                  ▼
   ┌──────────────────────────────────────────┐
   │ TCPClient — socket.socket()              │
   └──────────────┬───────────────────────────┘
                  │ bytes
                  ▼
   ┌──────────────────────────────────────────┐
   │ Servidor Java (porta 4242)               │
   └──────────────────────────────────────────┘

   Experiment _episode() gira o loop:
     state = task.get_sensors()
     agent.sense(state)
     task.perform_action(agent.act())
     agent.give_rewards(task.reward, task.cum_reward)
```

### 1. `TCPClient` (`marioai/core/environment.py:188`)

Socket cru. Tem seis métodos públicos: `connect`, `disconnect`, `recvData`, `send_data`, além de `__init__` e `__del__`. Usa buffer de 4096 bytes. Não faz framing — depende de o server entregar cada mensagem numa chamada `recv` (assumção frágil, mas funciona porque cada observação cabe em 4 KB).

### 2. `Environment` (`marioai/core/environment.py:10`)

Responsável por três coisas que poderiam estar separadas:

1. **Subir a JVM** — `_run_server` faz `subprocess.Popen` com `java ch.idsia.scenarios.MainRun -server on`, redireciona stdout/stderr para `server/tmp/*.log`.
2. **Conectar ao server** — cria um `TCPClient`, tenta até 5 vezes com 5s de intervalo.
3. **Traduzir ação ↔ observação** — `perform_action` formata a lista `[b,f,c,j,s]` em string `00000\r\n`; `get_sensors` chama `extractObservation` no buffer recebido.

Atributos de configuração (todos têm default): `level_difficulty`, `level_type`, `creatures_enabled`, `init_mario_mode`, `level_seed`, `time_limit`, `fast_tcp`, `visualization`, `custom_args`, `fitness_values`. São lidos em `reset()` para montar o comando `reset ...` que vai pro server.

> O `__init__` aceita `name`, `host`, `port` mas **não os armazena em `self`** — um bug que `_run_server` dispara ao tentar usar `self.name`, `self.host`, `self.port` (ver melhorias).

### 3. `marioai.core.utils.extractObservation` (`marioai/core/utils.py:65`)

Parser do protocolo. Decide pelo primeiro byte:

- `'E'` → chama `decode()` pra desempacotar 31 chars em grid 22×22, valida checksum, retorna `(mayJump, onGround, level_scene)`
- `'FIT'` → split por espaço, parseia 5 campos, retorna `(status, distance, timeLeft, marioMode, coins)`
- `'O'` → split por espaço, parseia flags booleanas, 484 ints do grid, 2 floats do Mario, restante como floats de inimigos. Retorna 6-tupla.
- Outro → `raise ValueError('Wrong format or corrupted observation...')`

### 4. `Task` (`marioai/core/task.py:8`)

Ponte entre o agente e o ambiente. Responsabilidades:

- **Instanciar** o `Environment` internamente (`self.env = core.Environment(...)`)
- **Transformar** a tupla crua do `extractObservation` em um dicionário `state` com chaves semânticas (`can_jump`, `on_ground`, `mario_floats`, `enemies_floats`, `level_scene`, `episode_over`) + features derivadas (`soft_1`, `soft_2`, `hard_1`, `enemy_1`, `brick_1`, `projetil_1`, `has_role_near_1`, ...)
- **Filtrar ações** — `filter_actions()` remove ações com `jump=1` quando `can_jump` é falso
- **Calcular recompensa** — `compute_reward()` é **passthrough por default**: retorna o dict cru de fitness e deixa subclasses sobrescreverem
- **Pool de ações** (`_action_pool`) — 14 combinações pré-definidas que cobrem andar, pular, correr, agachar, nas duas direções

O `build_state` também adiciona features de sensing derivadas do grid: pra cada categoria de objeto (`soft`, `hard`, `enemy`, `brick`, `projetil`) e pra cada distância de 1 a `max_dist`, marca se existe algo daquele tipo a essa distância do Mario. E `has_role_near_{dist}` marca se há piso disponível.

### 5. `Agent` (`marioai/core/agent.py:1`)

Interface simples, pensada pra ser herdada:

```python
class Agent:
    def reset(self): ...                 # chamado no início de cada episódio
    def sense(self, state): ...          # recebe o state dict do Task
    def act(self): ...                   # retorna [b,f,c,j,s]
    def give_rewards(self, r, cr): ...   # recebe reward e cumulative reward
```

Atributos expostos após `sense`: `level_scene`, `on_ground`, `can_jump`, `mario_floats`, `enemies_floats`, `episode_over`.

### 6. `Experiment` (`marioai/core/experiment.py:4`)

Loop de episódio puro. O método-chave é `_step`:

```python
state = task.get_sensors()
if task.finished or (frame % (response_delay + 1)) == 0:
    agent.sense(state)
    task.perform_action(agent.act())
    agent.give_rewards(task.reward, task.cum_reward)
else:
    task.perform_action([0, 0, 0, 0, 0])   # idle
```

O `response_delay` é um throttle: o agente decide só 1 em cada `N+1` frames; nos outros, o Mario segue parado. `do_episodes(n)` encadeia `_episode()` `n` vezes.

### 7. `Runner` (`marioai/core/runner.py:8`)

Açúcar em cima do `Experiment`. Recebe `agent` e `task`, configura o `Environment` dentro do `Task` (passa `level_difficulty`, `mario_mode`, etc.) e roda 1 episódio com `run()`. Tem `close()` que desconecta e `__del__` que chama `close()` como safety net.

### 8. `MarioEnv` (`marioai/gym/environment.py:32`)

Wrapper `gym.Env` em torno do `Environment`. Define:

- `action_space = Discrete(14)` — uma ação por entrada do pool (mesmas 14 combinações do `Task`)
- `observation_space = Box(0, 26, (22, 22))` — só o `level_scene`, achatado num único array
- `build_state` faz **remap** de valores especiais pra caber no `Box(0,26)`: `-10→24`, `-11→23`, `25→22`, `42→25`, Mario em `[11,11]→26`

`step(action)` chama `perform_action`, lê a próxima observação, calcula a recompensa e retorna a tupla `(obs, reward, done, info)` do Gym clássico.

> Esse wrapper é o que permite treinar com `stable_baselines3.DQN` (comando `dqn` da CLI).

---

## Agentes implementados

### `RandomAgent` (`marioai/agents/random_agent.py:10`)

Baseline minimalista. `act()` retorna `[0, 1, 0, random_jump, random_speed]` — sempre anda pra direita, pulo e corrida aleatórios. Nunca olha pro state.

### `BaseAgent` (`marioai/agents/base_agent.py:9`)

Versão "com memória" do random. Mesma `act()`, mas guarda listas de `states`, `actions`, `rewards`, `frames`. Útil como classe pai de agentes aprendendo.

### `ExploratoryAgent` (`marioai/agents/exploratory_agent.py:12`)

Tenta construir um state mais rico a partir do `level_scene` — mesma lógica de `Task.build_state`, duplicada aqui. A cada 24 frames, recalcula features de proximidade para `soft`, `hard`, `enemy`, `brick`, `projetil`. Apesar de construir o state, **a `act()` ainda retorna ação random** — o state é só registrado, não usado pra decidir.

### `MonteCarloAgent` (`marioai/agents/monte_carlo_agent.py:42`)

Tentativa mais séria: Monte Carlo control tabular com política ε-greedy.

- **State**: classe `State` (L10) que aceita kwargs arbitrários e implementa `__hash__`/`__eq__` pra servir como chave de dict. O state é alimentado pelo state dict que o `Task` constrói.
- **Q-table**: `self._Q: dict[State, np.ndarray]`, onde cada array tem shape `(n_ações_válidas,)`.
- **Policy**: `'random'`, `'greedy'` ou `'e_greedy'` (muda durante o treino).
- **compute_reward** (L76): não é passthrough — se ganhou, `distance * 2`; se tem `distance`, `distance * 0.1`; senão delta em x do Mario.
- **fit(task, **kwargs)**: loop de N episódios, atualiza ε e threshold, usa `tqdm`. No fim de cada episódio, `_step()` recalcula `G_t` descontado e atualiza Q.

Há dois bugs relevantes neste agente (detalhados em [`03-melhorias-cliente-python.md`](./03-melhorias-cliente-python.md)):

1. `filter_actions` (L99) cópia o pool de ações mas não filtra nada de fato.
2. `_step` (L157) usa `self.epsilon**k` como fator de desconto, quando o correto seria `self.discount**k`.

---

## CLI

`main.py` (root do repo) usa Click. Três subcomandos:

| Comando | Função |
| --- | --- |
| `python main.py random` | Roda 10.000 steps sampleando do `MarioEnv.action_space` |
| `python main.py mc` | Instancia `Task` + `MonteCarloAgent` e chama `.fit()` |
| `python main.py dqn` | Usa `stable_baselines3.DQN` sobre `MarioEnv` |

Opções comuns: `--level_difficulty`, `--mario_mode`, `--time_limit`, `--max_fps`. O comando `mc` aceita `--response_delay`; o `dqn` aceita `--total_timesteps` e `--log_interval`.

---

## Fluxo completo de um episódio (do jeito que acontece hoje)

1. Usuário roda `python main.py mc --level_difficulty 2`
2. Click parsea → chama `monte_carlo(...)` → instancia `Task` e `MonteCarloAgent`
3. `MonteCarloAgent.fit(task)` cria um `Runner(agent, task, ...)`
4. `Runner.__init__` aplica as configs no `task.env` (que é o `Environment`)
5. `Runner.run()` → `Experiment.do_episodes(1)` → `_episode()`
6. `_episode` chama `task.reset()`:
   - `Environment.reset` manda `reset -maxFPS on -ld 2 ...\r\n` pro server
   - Zera `cum_reward`, `finished`, `reward`
7. Loop `while not task.finished`:
   - `task.get_sensors()` → `env.get_sensors()` → `tcpclient.recvData()` → `extractObservation(bytes)` → tupla
   - `task.build_state(tuple)` → dict de state
   - Se veio `FIT` (len == 5), `task.finished = True`
   - `agent.sense(state)` atualiza atributos; `agent.act()` retorna `[b,f,c,j,s]`
   - `task.perform_action(action)` → `env.perform_action(action)` → `tcpclient.send_data(b"01000\r\n")`
8. Quando episódio termina, `Runner.run()` retorna
9. `MonteCarloAgent._step()` pega `self.states`, `self.actions_idx`, `self.rewards`, computa `G_t`, atualiza `_Q`
10. Próximo episódio

Ao final, `runner.close()` chama `task.disconnect()`, que chama `env.disconnect()`, que fecha o socket e mata o processo Java.

---

## Pontos do desenho que merecem leitura extra

- **Acoplamento Task↔Environment**: `Task.__init__` instancia o `Environment` diretamente com `core.Environment(*args, **kwargs)`. Não dá pra trocar o env por um mock sem hack.
- **State como dict**: tanto `Task.build_state` quanto `ExploratoryAgent._build_state` devolvem dicionários com chaves string. Sem TypedDict ou dataclass, é fácil errar uma chave.
- **`Environment` é god class**: ela faz subprocess + sockets + protocolo + config. Três responsabilidades deveriam virar três classes.
- **Duas implementações de sensing**: `Task._is_near/_has_role/_get_ground` e `ExploratoryAgent._is_near/_has_role/_build_state` são quase a mesma coisa, duplicadas.
- **Gym wrapper paralelo**: `MarioEnv.build_state` refaz o que o `Task.build_state` já faz, mas de um jeito diferente (remapping de valores). Dois caminhos de código pra mesma fonte de dados.
- **Sem testes**: nenhum `tests/`. O CI só roda flake8. A primeira vez que um parse de protocolo quebra, a gente só descobre rodando end-to-end com a JVM.

---

## Referências de arquivo (rápido)

| Camada | Arquivo |
| --- | --- |
| Socket | `marioai/core/environment.py:188` (`TCPClient`) |
| Orquestração + subprocess | `marioai/core/environment.py:10` (`Environment`) |
| Parsing de protocolo | `marioai/core/utils.py:65` (`extractObservation`) |
| State dict + filtro de ações | `marioai/core/task.py:8` (`Task`) |
| Interface de agente | `marioai/core/agent.py:1` (`Agent`) |
| Loop de episódio | `marioai/core/experiment.py:4` (`Experiment`) |
| Orquestração high-level | `marioai/core/runner.py:8` (`Runner`) |
| Gym wrapper | `marioai/gym/environment.py:32` (`MarioEnv`) |
| Agentes | `marioai/agents/*.py` |
| CLI | `main.py` |
