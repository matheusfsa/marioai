# Etapa 0 — Investigação empírica do servidor

Coleta de dados de referência rodando `RandomAgent` (20 episódios) e
`ExploratoryAgent` (5 episódios) em cada uma das 5 fases da competição
(total 125 episódios). Comando usado:

```
PYTHONPATH=. python3 competition/investigate.py --random-runs 20 --explore-runs 5
```

Dados crus em `competition/data/investigation.csv` (126 linhas, 1
cabeçalho + 125 episódios) e `competition/data/feature_stats.json`.
Tabelas abaixo foram geradas por `competition/_analyze_investigation.py`.

## Resumo por fase (RandomAgent, 20 runs)

| Fase | N | Vitórias | Distance (média ± std) | Distances únicas | Wall-clock (s) | Hash 1ª cena (únicos) |
|---|---:|---:|---:|---:|---:|---:|
| 1-easy     | 20 | 0/20 | 870 ± 294 | 4 | 1.46 | 1 |
| 2-medium-A | 20 | 0/20 | 203 ± 0   | 1 | 0.48 | 1 |
| 3-medium-B | 20 | 0/20 | 508 ± 1   | 2 | 2.47 | 1 |
| 4-hard-A   | 20 | 0/20 | 220 ± 1   | 2 | 2.96 | 1 |
| 5-hard-B   | 20 | 0/20 | 251 ± 1   | 2 | 2.97 | 1 |

> `coins` foi 0 em todas as fases exceto na 1-easy (3 ou 7 moedas). `final_mario_mode` só caiu abaixo do inicial na 5-hard-B (começa small, permanece small).

## Respostas às 5 perguntas do roadmap

### 1. As 5 fases estão bem calibradas?

**Não na fase fácil.** `RandomAgent` **não venceu nenhuma fase** em 100
runs, incluindo a "fácil". Isso viola a expectativa documentada no
roadmap ("fase fácil deve ter taxa de vitória > 0"). Ainda assim há
sinal útil para distinguir agentes:

- **1-easy**: distância média 870 com std 294 — há variação entre runs
  e o random chega a ~1195 no melhor caso. Existe headroom para
  agentes mais simples mostrarem progresso; a fase só não é
  literalmente ganhável por política sem visão.
- **2-medium-A**: todos os 20 runs morreram exatamente na distância
  203 (std 0). Existe um obstáculo na distância ~203 que random **não
  consegue atravessar** nunca. Fase serve como filtro binário —
  "passou da armadilha" vs "não passou".
- **3-medium-B** (underground), **4-hard-A** (castle), **5-hard-B**
  (random): também travaram sempre nas mesmas distâncias (507, 219,
  251), com variação de ±1 entre runs. Diferenciam agentes pela
  distância atingida antes de morrer.

**Fases 4 e 5 diferenciam bem agentes intermediários?** Provavelmente
sim: distâncias de parada são diferentes (220 vs 251) e dominadas por
obstáculos distintos (castle com Mario large vs random com Mario
small). Mas como random nunca vence, só saberemos com A*/DQN testados
se elas realmente admitem vitória.

**Recomendação**: manter as fases como estão. A Etapa 2 (rule-based,
A*) vai servir como segundo ponto de calibração — se nem rule-based
vencer a 1-easy, abrir issue para trocar `level_seed=1001` ou relaxar
`time_limit=60`. Do contrário, a violação do critério "random > 0%" é
aceitável: rule-based deve vencer trivialmente.

### 2. O desempate por `time_left` é informativo?

**Indeterminado nesta etapa.** `RandomAgent` não vence nenhuma fase,
então não há amostras de `time_left` válidas para medir dispersão. O
que sabemos é que a distância de parada varia pouco entre seeds
(1 a 4 valores distintos em 20 runs). Se o servidor for
determinístico dadas seed e ações, mesmo `distance` implica mesmo
`time_left`.

**Risco**: dois agentes determinísticos ganhando a mesma fase na
mesma trajetória podem empatar em `time_left` também, tornando o
desempate de 1º nível uma no-op. Fica como item para a Etapa 2 —
comparar `time_left` de `RuleBasedAgent` vs `AStarAgent` vencendo a
mesma fase.

**Recomendação**: **manter o desempate** (é barato) e adicionar
`distance` como 2º desempate conforme já previsto no README.

### 3. Estatísticas do estado

**Frequência de features (ExploratoryAgent, todos os runs por fase):**

| Feature           | 1-easy | 2-medium-A | 3-medium-B | 4-hard-A | 5-hard-B |
|-------------------|-------:|-----------:|-----------:|---------:|---------:|
| `can_jump`        |    39% |        38% |        39% |      39% |      39% |
| `on_ground`       |    58% |        56% |        58% |      59% |      59% |
| `enemy_1`         |     1% |         7% |         1% |       0% |       8% |
| `enemy_2`         |     1% |         8% |         0% |       0% |       8% |
| `hard_1`          |    46% |        46% |        50% |      57% |  **95%** |
| `hard_2`          |    47% |        77% |        51% |      57% |       0% |
| `soft_1`          |     6% |         0% |         0% |       0% |       0% |
| `soft_2`          |     7% |         0% |         0% |       0% |       0% |
| `brick_1`         |     0% |         0% |         0% |       0% |       0% |
| `brick_2`         |     0% |         0% |         0% |       0% |       0% |
| `projetil_1`      |     2% |         3% |         3% |       0% |       0% |
| `projetil_2`      |     5% |         1% |         5% |       0% |       0% |
| `has_role_near_1` |     0% |         3% |         0% |       0% |       1% |
| `has_role_near_2` |     1% |         3% |         0% |       0% |       1% |

Pontos críticos:

- **`brick_1` e `brick_2` são sempre `False`** nas 5 fases. Os tiles
  16/21 (bricks) não aparecem na coluna próxima a Mario em nenhum
  nível. **Descartar as duas features** — só adicionam ramificações
  inúteis na Q-table e no estado do rule-based.
- **`soft_1`/`soft_2`** só disparam na 1-easy (blocos-de-item do
  overground com seed baixa). Nas outras fases são 0%. **Manter
  apenas para agentes tabulares** — útil para explicar comportamento
  na 1-easy, neutro nas demais.
- **`projetil_1`/`projetil_2`** disparam 0–5% e caem para 0% nas
  fases 4-5 (Mario começa large/small, sem bolas-de-fogo). Candidatos
  a serem agregados em uma feature única `projetil_near = projetil_1
  or projetil_2` para economizar espaço de estado.
- **`has_role_near_X`** (buraco à frente) é raríssimo (0–3%). Pode
  indicar que a heurística de detecção em `sensing.has_role_near` não
  está calibrada para os tipos `2`/`3` (castle/random), ou que os
  níveis realmente não têm muitos buracos. Investigar na Etapa 2
  quando o rule-based começar a perder por cair em buracos.
- **`hard_1`**: 95% na 5-hard-B — Mario fica quase sempre encostado
  em parede no nível random. Isso explica por que ele trava em
  distance=251. Feature útil (é o sinal que dispara a regra "pular
  muro") — manter.

**Distribuição de tiles (top tiles por fase):**

| Fase       | tipo | empty  | ground (-10) | soft (-11) | hard_pipe (20) | enemy (2–15) | projectile (25) |
|------------|-----:|-------:|-------------:|-----------:|---------------:|-------------:|----------------:|
| 1-easy     |    0 |  85.5% |        12.8% |       1.5% |              — |         0.0% |            0.1% |
| 2-medium-A |    0 |  90.8% |         7.6% |          — |           1.3% |         0.4% |            0.1% |
| 3-medium-B |    1 |  74.2% |        23.5% |          — |           1.8% |         0.4% |            0.1% |
| 4-hard-A   |    2 |  83.0% |        14.9% |          — |           1.7% |         0.4% |               — |
| 5-hard-B   |    3 |  92.3% |         5.1% |          — |           2.4% |         0.2% |               — |

- Só **overground-fácil** (fase 1) tem tiles `-11` (soft). Mecânica
  de power-up via bloco só aparece na fase 1.
- Só da **fase 2 em diante** aparecem tiles `20` (hard pipes). Se o
  agente não aprender a pular pipes, trava na fase 2.
- **Inimigos voadores (tile 8)** predominam na 5-hard-B.
- **Inimigos terrestres (tile 12)** predominam em 3-medium-B e
  4-hard-A. Diversidade suficiente para justificar o agregador
  `enemy = {2..15}`.
- Não há tile **16 nem 21** (bricks) em nenhuma fase — confirma o
  descarte de `brick_X`.

**Estados únicos vistos pelo ExploratoryAgent (proxy para tamanho da Q-table tabular):**

| Fase       | frames | estados únicos | estados/frame |
|------------|-------:|---------------:|--------------:|
| 1-easy     |   1505 |             41 |         0.027 |
| 2-medium-A |    455 |             28 |         0.062 |
| 3-medium-B |   2157 |             37 |         0.017 |
| 4-hard-A   |   3005 |             11 |         0.004 |
| 5-hard-B   |   3005 |             23 |         0.008 |

União sobre as 5 fases: ≤ 140 estados únicos. Com a tupla atual de 14
features (8 booleanas + 6 n/a ocasionais) o espaço **teórico** é
`~2^14 = 16.384`, mas na prática observamos ≤ 50 estados por fase.
**Q-table cabe trivialmente em memória** mesmo para SARSA/Q-learning
com múltiplos milhões de transições — isso confirma que o plano
tabular é viável.

Observação importante: a baixa diversidade na 4-hard-A (11 estados em
3005 frames) reflete o fato de que Mario **fica parado morrendo no
mesmo lugar**. Quando um agente treinado explorar mais, o estado
crescerá. Dimensionar `_Q` como `dict` (não pré-alocar matriz) já é o
padrão em `monte_carlo_agent.py` e continua adequado.

### 4. Tempo wall-clock

**Duração por episódio (média dos 20 runs do RandomAgent, `max_fps=720`):**

| Fase       | s/episódio |
|------------|-----------:|
| 1-easy     |       1.46 |
| 2-medium-A |       0.48 |
| 3-medium-B |       2.47 |
| 4-hard-A   |       2.96 |
| 5-hard-B   |       2.97 |

Média por fase: **2.1 s/episódio**. Fase 2 é curtíssima (0.48 s)
porque Mario morre imediatamente no obstáculo da distance=203. Com
`max_fps=24` (realtime) o custo seria proporcional à duração do
episódio em Mario-seconds (até `time_limit`, i.e. 60–120 s).

**Projeções para o orçamento da Etapa 3 (tabulares):**

| Treino                       | ep.  | s/ep. | Total   |
|------------------------------|-----:|------:|--------:|
| MC, 1 fase, `n_samples=2000` | 2000 |   2.1 |  70 min |
| MC, 1 fase, `n_samples=500`  |  500 |   2.1 |  18 min |
| Q-learning, 5k eps, 1 fase   | 5000 |   2.1 | 175 min |

- **`n_samples=2000` é factível em ~1 h** — de acordo com a meta do
  `monte_carlo/02-modelagem.md`.
- Treinar em **múltiplas fases sequencialmente** por ~2–3 h total é
  viável localmente.
- **DQN com SB3** terá overhead adicional de forward/backward e
  buffer; budget realista é `total_timesteps=500k` rodando durante a
  noite (estimativa grosseira: 2-4 h).

### 5. Sanity-checks do protocolo

**Seeds 1001/2042/2077/3013/3099 geram níveis diferentes?**

Resultado parcial. O hash SHA-1 dos **primeiros** `level_scene` de
cada fase foi:

| Fase       | seed | hash prefix   |
|------------|-----:|---------------|
| 1-easy     | 1001 | `defcf016795f` |
| 2-medium-A | 2042 | `defcf016795f` |
| 3-medium-B | 2077 | `6af5e02531f5` |
| 4-hard-A   | 3013 | `e09ef91baa68` |
| 5-hard-B   | 3099 | `defcf016795f` |

Três hashes distintos em cinco fases. Fases **1, 2 e 5** colidem. Isso
**não** implica que os níveis sejam iguais — a primeira frame
costuma conter só Mario em um tile de chão vazio antes do scroll
revelar a geração procedural. Overground/random exibem a mesma vista
inicial (céu + chão), enquanto underground (3) e castle (4) começam
com paredes e por isso diferem.

**Evidência indireta de que os níveis de fato divergem**: as
distâncias em que `RandomAgent` morre são todas diferentes (203, 507,
219, 251) — se os níveis fossem idênticos, o primeiro obstáculo
estaria na mesma distância. Fica como tarefa da Etapa 1 adicionar um
hash mais tardio (ex.: após 50 steps) para confirmação formal.

**Há vazamento entre episódios?** Dentro de cada fase, 20 runs
consecutivos sempre retornaram a mesma distância até ±1 (ver tabela
do item 1). `Task.reset()` e `Environment.reset()` parecem limpar
estado corretamente. Runs após a fase anterior (ex.: rodar fase
2-medium-A após fase 1-easy) também voltam ao estado esperado (mesmo
hash da 1ª cena entre o smoke e o full run).

**Porta 4242 libera entre runs?** Sim — o script abre e fecha um
`Task` (e portanto um processo Java) por bloco de fase. 125 episódios
rodaram sem erro de `ConnectionRefusedError`. A limitação é que
**não dá para paralelizar** (única porta fixa), o que impacta a
Etapa 4 se quisermos rodar múltiplos DQN/PPO em paralelo.

## Recomendações consolidadas para as próximas etapas

1. **Descartar `brick_1` e `brick_2`** do estado compartilhado em
   `marioai/agents/utils/state.py` (Etapa 3). Atualizar
   `agents/*/02-modelagem.md` de MC/SARSA/Q-learning/ε-greedy para
   refletir a tupla reduzida. (Rule-based já não usa bricks.)
2. **Considerar colapsar `projetil_1 | projetil_2`** em uma única
   feature `projetil_near`. Decisão final na Etapa 3 — compare Q-table
   size antes/depois.
3. **Manter as 5 fases atuais.** Rever apenas `1-easy` se o
   rule-based também não vencê-la (Etapa 2) — candidato a trocar
   `level_seed=1001` para algo que até o rule-based ganhe.
4. **`n_samples=2000` para Monte Carlo é factível** (~1 h por fase).
   Orçamento de treino da Etapa 3 confirmado.
5. **Paralelização de treinos deep (Etapa 4) não é trivial** por
   conta da porta 4242 fixa. Planejar runs sequenciais durante a
   noite ou abrir ticket para suportar `-port N` no servidor Java.
6. **Adicionar hash tardio** (após ~50 steps) em qualquer script de
   diagnóstico futuro — o hash do primeiro frame é insuficiente para
   provar diferença de seeds.
7. **Desempate por `time_left`** permanece mas precisa ser validado
   na Etapa 2 com rule-based/A* vencendo a mesma fase.

## Artefatos

- `competition/investigate.py` — script CLI usado para coleta.
- `competition/_analyze_investigation.py` — helper que gera as
  tabelas acima a partir do CSV/JSON (não versionar saída; rerodar
  quando os dados mudarem).
- `competition/data/investigation.csv` — 125 episódios, 14 colunas.
- `competition/data/feature_stats.json` — agregados por fase (tiles,
  feature hits, estados únicos).
