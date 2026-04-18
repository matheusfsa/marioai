# Melhorias sugeridas — cliente Python

Este doc lista o que, na minha leitura do código (ver [`02-cliente-python.md`](./02-cliente-python.md)), não está bem resolvido e o que proponho mudar. Organizado por severidade decrescente. Cada item aponta `arquivo:linha` e descreve a correção.

> Uma parte expressiva desses itens é resultado natural de um port Python 2 → 3 feito há tempos e de iterações incrementais. Não é uma crítica ao código "em absoluto" — é um checklist do que vale endereçar agora que dá pra olhar com olhos frescos.

---

## 1. Bugs críticos (quebram a execução)

### 1.1 `Environment.__init__` não armazena `name/host/port`

`marioai/core/environment.py:34` recebe `name`, `host`, `port` como parâmetros mas **nunca atribui a `self`**. Em seguida, L60 chama `self._run_server()`, que em L95 faz:

```python
client = TCPClient(self.name, self.host, self.port)
```

Isso dispara `AttributeError` em qualquer uso minimamente configurável (apenas quando o usuário aceita *todos* os defaults a gente escapa — e mesmo assim, só porque a classe não tenta acessar antes).

**Fix:** no `__init__`, adicionar:
```python
self.name = name
self.host = host
self.port = port
```

### 1.2 Import absoluto quebrado em `config/__init__.py`

`marioai/config/__init__.py:1`:
```python
from config import ConfigLoader
```

Isso importa um pacote top-level chamado `config` — que não existe. Deveria ser `from .config import ConfigLoader`.

**Fix:** o módulo `marioai.config` **não é usado por nenhum outro módulo**. Verificado com `grep -r "marioai.config"`. A correção certa é **apagar o diretório inteiro** (abstract_config.py, common.py, config.py, __init__.py). Não é código morto importado, é código morto não-importável.

### 1.3 `gym/__init__.py` exporta nome inexistente

`marioai/gym/__init__.py:1`:
```python
from .environment import Environment
```

Mas a classe em `marioai/gym/environment.py:32` chama `MarioEnv`, não `Environment`. Esse import falha com `ImportError`.

**Fix:**
```python
from .environment import MarioEnv

__all__ = ['MarioEnv']
```

O `main.py:8` já importa `from marioai.gym import MarioEnv` — logo o bug só não está explodindo porque ninguém importa `marioai.gym` via `__init__.py`. Ainda assim, é lixo para limpar.

### 1.4 `dtype=np.int` quebra em NumPy ≥ 1.20

`marioai/core/utils.py:33` e `utils.py:71` usam `dtype=np.int`. Isso foi deprecado em NumPy 1.20 e **removido** em 1.24. Como `requirements.txt` trava `numpy==1.22.3`, hoje só emite `DeprecationWarning`, mas qualquer atualização quebra.

**Fix:** trocar por `dtype=int` (tipo Python nativo, interpretado pelo NumPy como `np.int_`).

---

## 2. Bugs lógicos (o código roda, mas não faz o que deveria)

### 2.1 `MonteCarloAgent._step` usa `epsilon` como fator de desconto

`marioai/agents/monte_carlo_agent.py:156`:
```python
discounts = np.array(
    [self.epsilon**k for k in range(future_rewards.shape[0])]
)
g_t = np.dot(discounts, future_rewards) / self.n_samples
```

A fórmula de retorno descontado do MC control é `G_t = Σ γ^k R_{t+k}`, onde γ é o **discount**, não o ε de exploração. No começo do treino `epsilon=1`, todos os rewards futuros têm peso 1 (como se não houvesse desconto); conforme ε cai, o desconto cresce paradoxalmente. A lógica está trocada.

**Fix:** substituir `self.epsilon` por `self.discount`. O `discount` já é recebido no `__init__` (L44) e guardado em L54.

### 2.2 `MonteCarloAgent.filter_actions` não filtra nada

`marioai/agents/monte_carlo_agent.py:99`:
```python
def filter_actions(self) -> np.array:
    action_pool = np.copy(self._action_pool)
    return action_pool
```

Só faz cópia. O objetivo (herdado do `Task.filter_actions`) é remover ações com `jump=1` quando `can_jump` for falso. Sem essa filtragem, o agente pode pedir pulo inválido (o server ignora, mas o Q-value que o agente aprende fica distorcido).

**Fix:**
```python
def filter_actions(self) -> np.ndarray:
    action_pool = np.copy(self._action_pool)
    if not self.state.get('can_jump', True):
        action_pool = action_pool[action_pool[:, 3] == 0]
    return action_pool
```

Cuidado: isso muda o shape do Q-array conforme o state. O jeito mais limpo é manter **sempre** o pool cheio nas tabelas e só zerar a probabilidade das ações inválidas no `policy` — ou indexar por (state, action_id_global) em vez de position-based.

### 2.3 `Task.compute_reward` é passthrough

`marioai/core/task.py:128`:
```python
def compute_reward(self, reward_data):
    return reward_data
```

Ou seja, `Task.reward` acaba sendo o dict cru do `FIT` no fim do episódio (e o dict zerado antes disso). O `Experiment` chama `agent.give_rewards(task.reward, task.cum_reward)` dentro do loop — o agente genérico recebe sempre dicts iguais antes do episódio acabar.

Isso não é tecnicamente um bug (subclasses podem sobrescrever), mas deixa o sinal de recompensa **nulo** pra qualquer agente que não reimplementar. Documentar explicitamente como contrato, ou fornecer uma implementação default razoável (por exemplo, diff de `mario_floats[0]` entre frames). Ver também `MonteCarloAgent.compute_reward` L76, que já faz esse cálculo mas fica numa posição inconsistente com o design da classe pai.

### 2.4 `ExploratoryAgent.act` não usa o state construído

`marioai/agents/exploratory_agent.py:70`:
```python
def act(self):
    self._build_state()
    return [0, 1, 0, random.randint(0, 1), random.randint(0, 1)]
```

Constrói o state (custo O(k²) no grid) e devolve ação random. Ou o state é usado pra decidir (intenção provável), ou todo o `_build_state` é código morto.

**Fix:** definir a intenção. Sugestão mínima: se detectar `enemy_1 == True`, priorizar pulo. Se não há intenção de usar, remover `_build_state` e colapsar pra cima do `RandomAgent`.

### 2.5 `ExploratoryAgent._build_state` muta o array de entrada

`marioai/agents/exploratory_agent.py:66`:
```python
self.level_scene[11, 11] = 100
```

Isso edita o array que veio do `Task` / `Environment`. Se outro consumidor inspecionar o mesmo array depois, verá `100` em `[11,11]` em vez do valor original. Também pode quebrar hashing se alguém usar o `level_scene` como chave de dict.

**Fix:** trabalhar sobre uma cópia: `scene = self.level_scene.copy()` no início do método e usar `scene` adiante.

### 2.6 `BaseAgent.frames` atribuída duas vezes

`marioai/agents/base_agent.py:12` e `:15` fazem `self.frames = 0` consecutivos. Inofensivo, mas indica código arrastado em merges.

**Fix:** apagar a linha 15.

### 2.7 `_is_near` tem range inconsistente

Em `marioai/core/task.py:182` e `marioai/agents/exploratory_agent.py:31`:
```python
def _is_near(self, ...):
    for i in range(1, 4):   # percorre i=1,2,3
        x = max(0, self.player_pos - i)
        y = min(level_scene.shape[0], self.player_pos + dist)
        if level_scene[x, y] in objects:
            return True
```

Fixa `range(1, 4)` — ignora `max_dist`, só olha as 3 linhas acima do Mario. Provavelmente a intenção era `range(1, self.max_dist + 1)` ou varrer uma caixa 2D (linhas e colunas).

**Fix:** decidir a semântica e documentar. Minimamente, substituir o `4` hardcoded.

---

## 3. Qualidade de código

### 3.1 `print()` em código de biblioteca

Ocorrências:

- `marioai/core/environment.py:64`, `:71`, `:94`
- `marioai/core/utils.py:61`, `:80`

Em uma biblioteca, logging é o padrão certo. `logging.getLogger(__name__)` já é usado nos outros lugares.

**Fix:** trocar por `logger.info` / `logger.warning` / `logger.error`.

### 3.2 File handles sem context manager

`marioai/core/environment.py:82-87`:

```python
stdout=open(
    source_dir / 'server/tmp/server_logOut.log', 'w', encoding='utf-8'
),
stderr=open(
    source_dir / 'server/tmp/server_logErr.log', 'w', encoding='utf-8'
),
```

O `Popen` segura os file descriptors enquanto o subprocess vive, e eles nunca são fechados explicitamente quando o processo termina. Em agentes de treino longo que sobem/derrubam muitos `Environment`, dá pra vazar fd.

**Fix:** guardar os `open()` em atributos e fechá-los em `disconnect()`, ou usar context managers controlados pela classe. Ou deixar o subprocess redirecionar pra `subprocess.DEVNULL` se os logs não forem consumidos.

### 3.3 `socket.socket()` sem argumentos explícitos

`marioai/core/environment.py:227`: chama o construtor sem `AF_INET, SOCK_STREAM`. Funciona porque esses são os defaults — mas torna o código mais difícil de ler e de portar.

**Fix:** `socket.socket(socket.AF_INET, socket.SOCK_STREAM)`.

### 3.4 Re-raise perde contexto

`marioai/core/environment.py:259`:
```python
except socket.error as message:
    logging.error(...)
    raise socket.error
```

`raise socket.error` instancia uma exceção nova, descartando traceback e mensagem original.

**Fix:** `raise` sozinho (re-raise a mesma exceção), ou `raise OSError("...") from message` se quiser reframar.

### 3.5 `dummy = 0` não usada

`marioai/core/utils.py:73`: variável criada mas referenciada só na tupla de retorno do caso `'O'` (L123). Se não vai usar, remover do retorno também e ajustar quem consome.

**Fix:** remover. Consumidores em `task.py:114` já trabalham com `len(sense) == 6` (da tupla com dummy) vs `len(sense) == 5` (FIT) — é uma heurística frágil. Melhor trocar por **tipos explícitos** (`Observation` e `FitnessResult` como dataclasses).

### 3.6 `assert` em validação de dados

`marioai/core/utils.py:41`:
```python
assert len(estate) == reqSize, f'Error in data size given {len(estate)}! ...'
```

Asserts somem com `python -O`. Pra validação de I/O, use exceção.

**Fix:** `raise ValueError(f'...')`.

### 3.7 Falta de type hints

Só `runner.py` e `gym/environment.py` têm hints parciais. O resto (agent, environment, utils, task, experiment, agentes) não tem. Type hints ajudam o leitor e permitem `mypy`/`pyright` no CI.

**Fix:** adicionar hints progressivamente nas APIs públicas.

### 3.8 Duplicação `Task` ↔ `ExploratoryAgent`

`_is_near`, `_has_role`, `_get_ground` aparecem duplicados em:

- `marioai/core/task.py:182-205`
- `marioai/agents/exploratory_agent.py:31-53`

**Fix:** extrair pra `marioai/core/sensing.py` (módulo novo) e importar nos dois lados. De quebra, fica testável isoladamente.

### 3.9 `Task` é string-typed

`build_state` retorna um dict com chaves literais hardcoded (`'can_jump'`, `'soft_1'`, `'has_role_near_2'`, etc.). Qualquer typo em qualquer consumidor gera `KeyError` em runtime.

**Fix:** `TypedDict` ou `dataclass`. Feito uma vez, o IDE autocompleta os consumidores.

### 3.10 `environment.reset` usa interpolação com strings

`marioai/core/environment.py:170` constrói um f-string com 5+ campos. Se amanhã entrar mais um parâmetro, a leitura piora.

**Fix:** construir uma lista de pares `(flag, value)` e fazer `' '.join(f'{k} {v}' for k, v in pairs)`. Menos erro-prono, mais testável.

### 3.11 `Runner.__del__`

`marioai/core/runner.py:68`:
```python
def __del__(self):
    self.close()
```

`__del__` rodando durante GC é problemático: `self.task` pode já ter sido coletado, e exceções em `__del__` são silenciadas. O mesmo pro `TCPClient.__del__` em `environment.py:216`.

**Fix:** preferir context managers (`__enter__`/`__exit__`) e encorajar uso com `with`. Remover `__del__` ou torná-lo defensivo (try/except amplo).

---

## 4. Empacotamento

### 4.1 Sem `pyproject.toml` / `setup.py`

O pacote não é instalável. `import marioai` só funciona se você rodar do root do repo, porque o root está no `sys.path` implicitamente. Qualquer reuso externo exige `pip install -e` — que não existe.

**Fix:** criar `pyproject.toml` com backend setuptools:

```toml
[project]
name = "marioai"
version = "0.2.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.22,<2",
    "click>=8.1",
    "tqdm>=4.64",
    "gym>=0.26",
    "shimmy>=0.2",
    "stable-baselines3>=2.0",
    "PyYAML>=6",
]

[project.optional-dependencies]
dev = ["pytest>=7", "ruff>=0.3"]
notebooks = ["jupyter", "jupyterlab", "pandas", "matplotlib"]

[project.scripts]
marioai = "marioai.cli:cli"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

### 4.2 `main.py` no root

O Click group vive em `main.py` no root. Se você instalar o pacote, o entry point desaparece.

**Fix:** mover pra `marioai/cli.py` e registrar `console_scripts` no `pyproject.toml`. Assim `pip install -e .` te dá `marioai` como comando no PATH.

### 4.3 Pins de versão no `requirements.txt`

`numpy==1.22.3`, `pandas==1.4.2`, `matplotlib==3.5.1` são travas específicas de quando o projeto rodava. Hoje dão conflito com `stable_baselines3>=2` em máquinas novas.

**Fix:** substituir por ranges no `pyproject.toml` e manter `requirements.txt` só como referência (ou apagar).

---

## 5. Testes

### 5.1 Zero testes

Não há `tests/`. O CI só roda `flake8`. A primeira prova de vida de qualquer mudança é "subir a JVM e jogar o Mario".

**Fix:** criar `tests/` com pytest. Prioridades:

- `tests/test_utils.py` — fixtures de bytes para os 3 formatos (`O`, `E`, `FIT`), verificar que `extractObservation` devolve os tipos certos, os shapes certos, e os valores limite (ex.: checksum inválido em `E`)
- `tests/test_task.py` — `build_state` com `level_scene` sintético: conferir que features de proximidade ativam quando esperado, filter_actions remove pulo quando `can_jump=False`
- `tests/test_agents.py` — smoke test: instanciar cada agente, passar um state mockado, garantir que `act()` devolve lista de 5 `{0,1}`

### 5.2 CI só roda lint

`.github/workflows/test.yaml` tem um único step: `make lint`. Não roda pytest.

**Fix:** adicionar step `pytest tests/` e subir Python de 3.8 pra 3.10 ou 3.11 (o `match`/`Self`/`TypeAlias` são úteis e 3.8 sai de suporte).

---

## 6. Arquitetura

### 6.1 `Environment` é god class

Ela faz:

1. Valida e sobe a JVM (`_check_java`, `_run_server`)
2. Gerencia o subprocess (`_server_process.kill`)
3. Gerencia o socket (delegando pro `TCPClient`)
4. Traduz ação Python → bytes TCP
5. Parsea observação de volta
6. Armazena config do próximo `reset`

Três classes fariam esse trabalho melhor:

```
ServerProcess        # subir/derrubar a JVM, redirecionar logs
TCPClient            # socket (já existe, só ampliar)
MarioEnvironment     # orquestra ServerProcess + TCPClient + config
```

### 6.2 `Observation` e `FitnessResult` deveriam ser tipos

Hoje `extractObservation` retorna ora tupla de 3, ora de 5, ora de 6. `Task` descobre o caso por `len(sense)`. É **frágil**: qualquer mudança nos formatos quebra o consumidor silenciosamente.

**Fix:**

```python
@dataclass
class Observation:
    may_jump: bool
    on_ground: bool
    mario_floats: tuple[float, float]
    enemies_floats: list[tuple[float, float]]
    level_scene: np.ndarray

@dataclass
class FitnessResult:
    status: int
    distance: float
    time_left: int
    mario_mode: int
    coins: int

def extract_observation(data: bytes) -> Observation | FitnessResult: ...
```

### 6.3 `Task` cria `Environment` internamente

`Task.__init__` faz `self.env = core.Environment(*args, **kwargs)` (task.py L39). Não dá pra injetar um fake env pra testar o `Task` isolado.

**Fix:** passar `env` por dependency injection: `Task(env, window_size=4, ...)`. Quem constrói a Task constrói o env antes.

### 6.4 `marioai/config/` não serve pra nada

Framework de config style Kedro, sem ninguém chamando. Ocupa espaço mental.

**Fix:** apagar. Se amanhã precisar de config, reescrever com `pydantic` ou `dataclass` + `yaml.safe_load`.

### 6.5 Duas pipelines paralelas pra state

- `Task.build_state` (para agentes "nativos" via `Experiment`)
- `MarioEnv.build_state` (para agentes gym/stable_baselines3)

Elas fazem coisas diferentes com a mesma fonte. Isso é ok pro gym precisar de `Box` e remapping de IDs, mas o ideal é que ambas partam de uma representação comum (`Observation` dataclass) e só divirjam no final.

---

## 7. Documentação

### 7.1 Docstrings incompletas ou ausentes

Algumas classes têm docstrings (ex.: `Environment`, `TCPClient`), outras nem isso. `MonteCarloAgent`, `ExploratoryAgent` e `MarioEnv` são especialmente opacos. Funções como `_step`, `_build_state`, `_is_near` não explicam semântica.

**Fix:** docstring Google-style em toda API pública. Funções privadas, só quando a lógica for não-óbvia.

### 7.2 README desatualizado

O README atual só ensina a rodar o server manualmente e a escrever um agente básico. Não menciona:

- Instalação (`pip install -e .`) quando tiver `pyproject.toml`
- CLI (`marioai random`, `marioai mc`, `marioai dqn`)
- Como treinar com DQN
- Dependência de Java (versão mínima)
- Apontador pros docs detalhados (`docs/01-servidor-java.md`, `docs/02-cliente-python.md`)

**Fix:** atualizar o README com essas seções. Os detalhes profundos ficam nos docs.

### 7.3 Nome `can_jump` vs `mayMarioJump`

O servidor manda `mayMarioJump`; o parser converte pra `mayMarioJump` (snake varia); o state dict usa `can_jump`; o atributo no `Agent` é `can_jump`. Dá pra ficar ok, mas em um PR de renomeação fica tudo mais legível se escolhermos um único nome.

**Fix:** padronizar em `can_jump` em todas as camadas Python. O servidor continua mandando como `mayMarioJump`, isso é protocolo e não muda.

---

## Ordem sugerida de aplicação

Pra não quebrar tudo ao mesmo tempo:

1. **Primeiro os bugs críticos** (seção 1) — 4 fixes, ~30 linhas de diff, imediatamente torna o código executável fora de corredores estreitos.
2. **Depois os bugs lógicos** (seção 2) — precisam de atenção conceitual (ε vs γ, `filter_actions`), mas são localizados.
3. **Limpeza** — apagar `config/`, remover print, context managers. Boa sessão de ~1h.
4. **Sensing.py + TypedDict/dataclass** (3.8, 3.9, 6.2) — mexe em vários arquivos mas é o maior ganho de manutenibilidade.
5. **Empacotamento** (seção 4) — `pyproject.toml`, mover `main.py`. Uma vez feito, simplifica tudo depois.
6. **Testes** (seção 5) — com `extractObservation` já tipado, testar vira um prazer.
7. **Arquitetura** (seção 6) — a quebra do `Environment` em três classes é um refactor grande; idealmente depois de ter testes.
8. **Doc & docstrings** (seção 7) — no final, quando a API já parou de mudar.

Ver [`02-cliente-python.md`](./02-cliente-python.md) para a descrição do estado atual.
