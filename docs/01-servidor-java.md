# Servidor Java

O servidor Java deste repositório é um fork do framework da [Mario AI Competition (2009)](http://julian.togelius.com/mariocompetition2009/), mantido por Togelius. **Só temos os `.class` compilados** — não há código-fonte Java neste repo. Esta doc descreve o comportamento observado, a estrutura de pacotes e o protocolo TCP pelo qual o cliente Python se comunica com ele. Pense no servidor como uma **caixa-preta operacional**: a gente sabe o que entra, o que sai e quais knobs existem.

---

## Onde ele vive

- Diretório: `marioai/core/server/`
- Working directory ao rodar: esse mesmo diretório (os `.class` ficam em `ch/idsia/...` relativos a ele)
- Entry point: `ch.idsia.scenarios.MainRun`
- Arranque manual: `server.bat` (Windows) ou `java ch.idsia.scenarios.MainRun -server on`
- Arranque automático: o `Environment` Python faz `subprocess.Popen` com o mesmo comando (ver `marioai/core/environment.py:75`)

---

## Estrutura de pacotes

Top-level: tudo dentro de `ch.idsia.*`.

```
ch/idsia/
├── scenarios/         # entry points do processo (MainRun, Play, Evolve, ...)
├── tools/
│   └── tcp/           # servidor TCP: Server, ServerAgent, TCP_MODE
├── mario/
│   ├── engine/        # engine do jogo: LevelScene, MarioComponent, sprites, renderização
│   │   ├── level/     # LevelGenerator (procedural), ImprovedNoise, Level, SpriteTemplate
│   │   ├── sprites/   # Mario, Enemy, BulletBill, Mushroom, Fireball, ...
│   │   └── mapedit/   # editor de mapas (não usado pelo cliente Python)
│   ├── simulation/    # BasicSimulator, Simulation, SimulationOptions
│   └── environments/  # Environment, EnvCell — interface interna do mundo
├── ai/
│   ├── agents/        # interface Agent + agentes built-in (Forward, Scared, MLP, SRN, ...)
│   ├── tasks/         # Task, ProgressTask, CoinTask, MultiSeedProgressTask, ...
│   └── es/            # evolution strategies
├── tools/             # Evaluator, EvaluationOptions, ToolsConfigurator, LOGGER, GameViewer
└── utils/             # MathX, ParameterContainer, StatisticalSummary, ...
```

Para o cliente Python, **só os três primeiros grupos importam**: o servidor TCP (entrada), o engine (simulação) e as tasks (critério de avaliação).

---

## Ciclo de vida de um episódio

Em alto nível:

```
1. Python sobe o server com `java ... MainRun -server on`
2. Server abre ServerSocket na porta 4242 e aguarda
3. Python conecta → Server cumprimenta → Python se identifica
4. Python manda `reset ...` com os parâmetros do nível
5. Loop:
     server envia observação   → Python decide ação
     Python envia ação         → server avança 1 frame
   até o episódio acabar (Mario morre, ganha ou time esgota)
6. Server manda `FIT ...` com o resultado final do episódio
7. Pode voltar ao passo 4 pra novo episódio, ou desconectar
```

Cada `reset` instancia um novo nível via `LevelGenerator` (usa Perlin noise — `ImprovedNoise`) com o seed pedido, posiciona o `Mario` e os `Enemy`s, e começa a enviar frames.

---

## Camada de rede

Classes relevantes em `ch.idsia.tools.tcp`:

| Classe | Papel |
| --- | --- |
| `Server` | TCP listener na porta 4242 (default). Aceita conexões. |
| `ServerAgent` | Adaptador que implementa `ch.idsia.ai.agents.Agent` mas, em vez de decidir localmente, pergunta ao cliente remoto via socket. |
| `TCP_MODE` | Enum: modo normal ou "fast TCP" (serialização compacta — formato `E`). |
| `Server$STATUS` | Estado interno do server. |

O `-server on` faz com que o agente efetivo da simulação seja um `ServerAgent` em vez de um agente Java built-in. O engine acha que está conversando com um agente local; na verdade cada chamada de `act()` do `ServerAgent` faz uma round-trip TCP.

### Handshake

1. Server: envia uma string de saudação assim que o cliente conecta
2. Client: responde `Client: Dear Server, hello! I am <nome>\r\n`
3. Server: entra em modo pronto, aguarda `reset` ou ações

Todas as mensagens são **terminadas por `\r\n`**. Não há framing por comprimento — é parsing text-based.

---

## Protocolo de mensagens

Três formatos de mensagem saem do server (identificados pelo primeiro byte), um formato entra (ações).

### 1. Observação `O` (text, verbose)

```
O <mayJump> <onGround> <484 ints separados por espaço> <marioX> <marioY> <enemy1X> <enemy1Y> ...
```

- `mayJump`, `onGround`: `true` ou `false`
- 484 inteiros = grid 22×22 do `level_scene`. Mario sempre em `[11, 11]`
- Valores do grid: ver [tabela no README](../README.md) (ex.: `-10` parede dura, `-11` plataforma mole, `2` goomba, etc.)
- `marioX`, `marioY`: floats, posição real
- Depois vêm pares `(x, y)` de cada inimigo visível

Decodificado no Python por `extractObservation` em `marioai/core/utils.py:93`.

### 2. Observação `E` (bit-packed, fastTCP)

```
E<jumpChar><groundChar><31 chars codificados><checksum>
```

- Cada char codifica 16 bits, totalizando 484+ bits (um bit por célula do grid, binarizado)
- Usado quando `reset` tiver `-fastTCP on`
- Formato mais denso, mas perde informação fina (só sabe se a célula tem "algo" ou não — dependendo da build)
- Decodificado em `marioai/core/utils.py:28` (função `decode`)

### 3. Fitness `FIT` (fim de episódio)

```
FIT <status> <distance> <timeLeft> <marioMode> <coins>\r\n
```

- `status`: `0` sobreviveu, `1` ganhou, `2` morreu
- `distance`: float, deslocamento horizontal total
- `timeLeft`: int, segundos-Mario restantes
- `marioMode`: `0` pequeno, `1` grande, `2` grande com fogo
- `coins`: moedas coletadas

### 4. Ação (Python → server)

```
<b><f><c><j><s>\r\n
```

Cinco caracteres `0`/`1` seguidos de `\r\n`. Cada posição ativa/desativa um botão:

| Posição | Botão |
| --- | --- |
| `b` | backward (esquerda) |
| `f` | forward (direita) |
| `c` | crouch (abaixar) |
| `j` | jump (pulo) |
| `s` | speed / bombs (correr / soltar fogo) |

Exemplos: `01000\r\n` anda pra direita, `01010\r\n` pula pra direita, `00000\r\n` fica parado.

### 5. Reset

```
reset -maxFPS on -ld <ld> -lt <lt> -mm <mm> -ls <ls> -tl <tl> -pw <on|off> -vis <on|off> [-fastTCP on] [extras...]\r\n
```

Parâmetros reconhecidos pelo server (construídos em `Environment.reset`, `marioai/core/environment.py:166`):

| Flag | Nome | Valores | Função |
| --- | --- | --- | --- |
| `-ld` | level difficulty | int (sugerido 0–30) | Quão difícil o nível gerado é |
| `-lt` | level type | 0 overground, 1 underground, 2 castle, 3 random | Tipo de cenário |
| `-mm` | mario mode | 0 small, 1 large, 2 large+fire | Estado inicial do Mario |
| `-ls` | level seed | int | Seed do gerador (reprodutibilidade) |
| `-tl` | time limit | int (segundos-Mario) | Limite de tempo do episódio |
| `-pw` | power/enemies | `on` desabilita criaturas, `off` habilita | Inverte o que o nome sugere; cuidado |
| `-vis` | visualization | `on`/`off` | Se a janela do jogo deve abrir |
| `-fastTCP` | fast TCP | `on`/`off` | Ativa o formato `E` (bit-packed) |
| `-maxFPS` | cap FPS | sempre `on` no cliente | Remove cap de 24 fps, roda o mais rápido possível |

Notas:
- `-pw` é invertido em relação ao nome. `pw=off` → criaturas habilitadas, `pw=on` → criaturas desabilitadas. O cliente Python faz o XOR certo em `Environment.reset`.
- `-maxFPS on` é a flag que permite rodar milhares de frames por segundo para treino. Sem isso, o server tenta manter 24 fps reais.

---

## Engine

O motor do jogo vive em `ch.idsia.mario.engine.*`. Destaques (pelo tamanho dos `.class`, dá pra inferir onde está a complexidade):

- `LevelScene` (23 KB): o mundo do jogo. Mantém o grid, as colisões, os sprites ativos. É daqui que sai o `level_scene` 22×22 mandado pro Python.
- `MarioComponent` (13 KB): o loop principal do jogo — integra inputs, avança a física, chama o render.
- `Art`, `BgRenderer`, `LevelRenderer`: assets e renderização (PNGs `mariosheet.png`, `enemysheet.png`, etc.).
- `Recorder` / `Replayer`: gravação e reprodução de partidas (não usado pelo Python).

Sprites em `ch.idsia.mario.engine.sprites.*`: `Mario` (com enum `Mario$MODE`), `Enemy`, `Sprite`, `BulletBill`, `FlowerEnemy`, `Mushroom`, `FireFlower`, `Fireball`, `Shell`, `CoinAnim`, `Particle`, `Sparkle`.

Geração de nível em `ch.idsia.mario.engine.level.*`: `LevelGenerator`, `BgLevelGenerator`, `Level`, `SpriteTemplate`, `ImprovedNoise` (Perlin 3D). O seed é consumido aqui — mesmo seed + mesmos parâmetros = mesmo nível.

Simulação em `ch.idsia.mario.simulation.*`:
- `Simulation` (interface)
- `BasicSimulator` (implementação default, 8 KB)
- `SimulationOptions` (parâmetros da simulação)

---

## Agentes built-in e tasks (Java)

O repo inclui vários agentes Java prontos em `ch.idsia.ai.agents.*`:

- `ForwardAgent`, `ForwardJumpingAgent`: baselines simples
- `ScaredAgent`, `ScaredSpeedyAgent`: fogem de inimigos
- `RandomAgent`, `TimingAgent`: aleatório e temporal
- Agentes neurais: `SmallMLPAgent`, `MediumMLPAgent`, `LargeMLPAgent`, `SmallSRNAgent`, `MediumSRNAgent`, `LargeSRNAgent`, `SimpleMLPAgent`
- `HumanKeyboardAgent`, `CheaterKeyboardAgent`: controle manual

Esses agentes **não são usados pelo cliente Python** — o `-server on` substitui todos por um `ServerAgent` que delega decisões via TCP. Estão aqui porque o framework original foi desenhado pra rodar competições inteiramente em Java.

Tasks de avaliação em `ch.idsia.ai.tasks.*`:

- `ProgressTask` — pontua pela distância percorrida
- `ProgressPlusTimeLeftTask` — distância + tempo restante
- `CoinTask` — moedas coletadas
- `MultiSeedProgressTask`, `MultiDifficultyProgressTask`, `StochasticProgressTask` — variantes que rodam múltiplas seeds/dificuldades

Hoje o cliente Python só consome o `FIT` bruto e calcula a própria recompensa. As tasks Java ficam dormentes quando o server está em modo `-server on`.

---

## Scenarios

Scenarios são os entry points do processo Java, em `ch.idsia.scenarios.*`:

| Scenario | Uso |
| --- | --- |
| `MainRun` | **O que a gente usa** — sobe o TCP e delega as decisões pro cliente |
| `Play` | Roda um jogo local (precisa de agente Java, ignora o server) |
| `Evolve`, `EvolveIncrementally`, `EvolveMultiSeed` | Evolução de agentes Java com EA |
| `RandomSearch` | Busca aleatória de hiperparâmetros |
| `CustomRun`, `Stats`, `CompetitionScore` | Utilitários de benchmark |

---

## O que pode dar errado

Pontos de atenção observados:

- **Checksum no formato `E`**: o cliente Python calcula e compara, mas só loga se diferir — não reconecta nem re-pede o frame (ver `marioai/core/utils.py:79`).
- **Sinal `ciao`**: quando o server encerra a sessão, manda a string `ciao`. O Python detecta em `Environment.get_sensors` (`marioai/core/environment.py:122`).
- **Timeout de conexão inicial**: o cliente Python tenta 5 vezes com intervalo de 5s antes de desistir (`_run_server` L89–103). Se a JVM demorar muito pra subir, isso às vezes falha.
- **Porta fixa 4242**: não há fallback se a porta estiver ocupada. Matar processos zumbis é manual.
- **Logs em `server/tmp/`**: o cliente redireciona stdout/stderr do server pra arquivos nesse diretório; eles ficam abertos até o processo Python terminar.

---

## Referências de arquivo

| Responsabilidade | Caminho |
| --- | --- |
| Entry point | `marioai/core/server/ch/idsia/scenarios/MainRun.class` |
| Servidor TCP | `marioai/core/server/ch/idsia/tools/tcp/Server.class`, `ServerAgent.class` |
| Simulador | `marioai/core/server/ch/idsia/mario/simulation/BasicSimulator.class` |
| Mundo do jogo | `marioai/core/server/ch/idsia/mario/engine/LevelScene.class` |
| Geração de nível | `marioai/core/server/ch/idsia/mario/engine/level/LevelGenerator.class` |
| Sprites | `marioai/core/server/ch/idsia/mario/engine/sprites/*.class` |
| Assets | `marioai/core/server/ch/idsia/mario/engine/*.png` |

Para entender o cliente que conversa com tudo isso, ver [`02-cliente-python.md`](./02-cliente-python.md).
