# Roadmap de Implementação

Controle de progresso da competição. As etapas são **sequenciais** (cada uma depende da anterior), exceto pelos agentes dentro de uma mesma família, que podem ser feitos em paralelo.

## Visão geral

| Etapa | Título | Status |
|---|---|---|
| 0 | Investigação empírica do servidor | ✅ concluída |
| 1 | Infraestrutura compartilhada da competição | ✅ concluída |
| 2 | Agentes sem treino (A*) | 🔄 em andamento (no-crash ok; tuning pendente) |
| 3 | Agentes tabulares (ε-greedy, MC, SARSA, Q-learning) | ⬜ pendente |
| 4 | Agentes deep (DQN, PPO) | ⬜ pendente |
| 5 | Execução da competição e relatório final | ⬜ pendente |

---

## Etapa 0 — Investigação empírica do servidor

**Objetivo**: antes de escrever qualquer agente, executar o servidor Java com agentes já existentes (`RandomAgent`, `ExploratoryAgent`) nas 5 fases propostas e em configurações exploratórias, para **calibrar as escolhas de modelagem** com dados reais. O resultado é um único markdown de referência (`competition/00-investigation.md`) que confirma ou ajusta decisões nos `02-modelagem.md` de cada agente.

### Perguntas a responder

1. **As 5 fases estão bem calibradas?**
   - Qual a distância média alcançada pelo `RandomAgent` em cada fase?
   - Qual a taxa de vitória do `RandomAgent` nas fases fáceis/médias (deve ser > 0 para a fácil)?
   - As fases difíceis são realmente intransponíveis pelo `RandomAgent` (taxa ≈ 0)?
   - Fases 4 e 5 têm distâncias diferentes o suficiente para diferenciar agentes intermediários de bons?

2. **O desempate por `time_left` é informativo?**
   - Na fase fácil, dois `RandomAgent`s com sementes diferentes retornam `time_left` diferentes? Se não, o desempate vai falhar (empate total).

3. **Estatísticas do estado**
   - Distribuição de valores do `level_scene` por `level_type` (overground vs. underground vs. castle): quais tiles são comuns em cada?
   - Frequência de `can_jump == False` ao longo de um episódio (quanto tempo Mario passa sem poder pular)?
   - Distribuição de `on_ground`.
   - Frequência em que as features `enemy_d`, `hard_d`, `has_role_near_d` disparam — alguma é sempre False (inútil)?
   - Número de estados únicos observados pelo `ExploratoryAgent` em 100 episódios (estimativa grosseira da Q-table dos agentes tabulares).

4. **Tempo wall-clock**
   - Duração real (em segundos) de 1 episódio em cada fase com `max_fps=720` e `max_fps=24`.
   - Quantos episódios por hora o tabular consegue treinar? E o DQN (com overhead de rede + SB3)?
   - Isto aterra o budget de treino: `n_samples=2000` para MC é factível?

5. **Sanity-checks do protocolo**
   - O servidor aceita seeds 1001/2042/2077/3013/3099 e gera níveis realmente diferentes? (verificar hash do primeiro `level_scene`).
   - Há vazamento de estado entre episódios (`task.reset()` limpa tudo)?
   - A porta 4242 libera corretamente entre runs?

### Entregáveis

- **`competition/investigate.py`** — script CLI que:
  - Roda `RandomAgent` × `N_RUNS` (default 20) em cada uma das 5 fases.
  - Roda `ExploratoryAgent` × 5 em cada fase para coletar features ao longo dos episódios.
  - Coleta: `status`, `distance`, `time_left`, `coins`, wall-clock, contagem de tiles, frequência de features.
  - Salva resultados em `competition/data/investigation.csv` (por episódio) e `competition/data/feature_stats.json` (agregado).
- **`competition/00-investigation.md`** — relatório com:
  - Tabela de resumo (média ± std por fase: distance, time_left, status, wall-clock).
  - Histograma ASCII/markdown da distribuição de tiles por fase.
  - Resposta às 5 perguntas acima.
  - Recomendações finais para a Etapa 1 em diante (ex.: "descartar feature `projetil_d` porque é sempre False"; "fase 5 é impossível de diferenciar no `RandomAgent` porque ninguém passa dos primeiros 20 tiles").
- **PR de ajuste (se necessário)**: se a investigação revelar que uma fase está mal calibrada ou uma feature é inútil, atualizar `competition/README.md` e os `02-modelagem.md` afetados.

### Critério de "pronto"

- [x] `competition/investigate.py` roda sem crashes nas 5 fases.
- [x] `competition/data/investigation.csv` gerado com ≥ 100 linhas (20 runs × 5 fases). _125 linhas, 1 por episódio (20 random + 5 explore)._
- [x] `competition/00-investigation.md` responde às 5 perguntas com dados (não apenas opinião).
- [x] Decisões de modelagem controversas (ex.: number de features, shaping exato, budget) têm base empírica documentada no markdown.

---

## Etapa 1 — Infraestrutura compartilhada da competição

**Objetivo**: módulo reutilizável que carrega a configuração das 5 fases, roda um agente arbitrário em todas e agrega o placar. Todos os agentes das Etapas 2–4 dependem deste.

### Entregáveis

- **`marioai/competition/__init__.py`** — re-exporta as APIs públicas.
- **`marioai/competition/phases.py`** — constante `PHASES: list[PhaseConfig]` (dataclass com `name`, `level_difficulty`, `level_type`, `level_seed`, `mario_mode`, `time_limit`).
- **`marioai/competition/runner.py`** — `CompetitionRunner(agent)`:
  - Método `evaluate()` → `list[PhaseResult]`. Chama `agent.set_deterministic()` antes, roda `Runner` em cada fase, coleta `status`/`distance`/`time_left`/`coins`/`wallclock`.
  - Gerencia um único `Task` ao longo das 5 fases (reaproveita o servidor Java).
- **`marioai/competition/scoreboard.py`** — `Scoreboard`:
  - `add(agent_name, results)`.
  - `rank()` → lista ordenada aplicando regras: fases vencidas → `time_left` médio → `distance` total.
  - `to_markdown()` para render do placar.
- **Protocolo `DeterministicAgent`** (em `runner.py`): interface opcional (`Protocol` do `typing`) com método `set_deterministic()`. Agentes que não implementarem são aceitos (assume-se já determinístico).
- **Testes**: `tests/competition/test_runner.py` e `test_scoreboard.py` com mocks do `Task` (não precisa subir o servidor Java em CI).

### Critério de "pronto"

- [x] `python -m marioai.competition --help` (ou CLI equivalente via `click`) existe.
- [x] `CompetitionRunner(RandomAgent()).evaluate()` executa as 5 fases sem erro.
- [x] `Scoreboard` produz ordenação estável dado um conjunto de `PhaseResult`s.
- [x] Testes unitários cobrem os casos de empate por `time_left` e `distance`.
- [x] `make lint` e `make test` passam.

---

## Etapa 2 — Agentes sem treino

Valida o `CompetitionRunner` antes dos agentes que exigem treino.

### 2.1 A*
- **Arquivo**: `marioai/agents/astar_agent.py`, classe `AStarAgent(Agent)`.
- **Helpers internos**: `_plan(level_scene)` retornando lista de `(row, col)`; `_path_to_action(next_cell)`.
- **Cache**: invalidar a cada 12 frames ou ao desviar da célula prevista.
- **Teste**: `tests/test_astar.py` — `level_scene` sintético com caminho óbvio → `_plan` retorna a sequência certa.

### Critério de "pronto"

- [x] O agente passa pelo `CompetitionRunner` nas 5 fases sem crash. _Validado em `competition/_validate_etapa2.py`: status=0 em todas, distâncias 140–508._
- [ ] `AStarAgent` vence pelo menos a fase 1 e mostra progresso (distance > RandomAgent) nas fases 2–3. _Gap atual: o planner não modela gravidade (trata ar como andável), então não planeja saltos de pit proativamente. Em fase 1 o agente fica preso em ~372 (abaixo do RandomAgent ~600–1200). Fase 3 (508) já supera um Random estático. Tuning pendente._

---

## Etapa 3 — Agentes tabulares

Dependem da Etapa 1. Podem ser feitos em paralelo entre si.

### Ordem sugerida

1. **Monte Carlo** (já existe em `marioai/agents/monte_carlo_agent.py`) — ajustar hyperparams conforme `02-modelagem.md` e empacotar para a competição.
2. **ε-greedy com features** — mais simples, valida a tupla de estado compartilhada.
3. **Q-learning** — primeiro on-line, referência para SARSA.
4. **SARSA** — quase idêntico ao Q-learning, só muda o target.

### Entregáveis comuns a todos os tabulares

- **`State` compartilhado**: extrair de `monte_carlo_agent.py:14-52` para `marioai/agents/utils/state.py` se ainda não estiver lá. Reutilizar nos quatro.
- **Funções de discretização** em `marioai/agents/utils/features.py` (se necessário além do que `Task.build_state` já fornece).
- **Helper `decay_epsilon(schedule, ep, total)`** em `marioai/agents/utils/exploration.py`.
- **Cada agente** em arquivo próprio (`epsilon_greedy_agent.py`, `sarsa_agent.py`, `q_learning_agent.py`), seguindo o padrão do `MonteCarloAgent`.
- **`fit()` method** que treina e deixa `policy_kind='greedy'` ao fim.
- **Testes**: cobrem update de Q, decay de ε, e integração com `Task` mockado.

### Critério de "pronto"

- [ ] Cada agente tem `fit()` que roda um treino curto (100 episódios, fase fácil) sem erros.
- [ ] Após treino, cada agente vence a fase 1 em > 90% das execuções.
- [ ] Artefatos de treino (Q-tables) são serializáveis (`pickle` ou JSON) para reuso.
- [ ] `make lint` e `make test` passam.

---

## Etapa 4 — Agentes deep

Dependem da Etapa 1 e do `gym` wrapper existente. SB3 já é dependência do projeto.

### 4.1 Shaped environment
- **Arquivo**: `marioai/gym/shaped_environment.py`, classe `ShapedMarioEnv(MarioEnv)`.
- Override de `compute_reward` com o shaping documentado nos `02-modelagem.md` do DQN/PPO.
- Reset de `prev_distance`/`prev_coins` em `reset()`.

### 4.2 Frame stacking
- Usar `gym.wrappers.FrameStack` se compatível com a versão `gym` travada no `pyproject.toml`, ou implementar wrapper mínimo.

### 4.3 DQN
- **Arquivo**: `marioai/agents/dqn_agent.py` com função `train_dqn(env, **hp)` que retorna um `DQN` treinado.
- **Script**: `competition/train_dqn.py` que treina 500k passos e salva `dqn_mario.zip`.

### 4.4 PPO
- **Arquivo**: `marioai/agents/ppo_agent.py` análogo ao DQN.
- **Script**: `competition/train_ppo.py`.

### 4.5 Adaptador para o CompetitionRunner
- Criar classe wrapper que encapsula um `DQN.load(...)` / `PPO.load(...)` e implementa a interface `Agent` (`act()`, `sense()`) consumindo `ShapedMarioEnv`.

### 4.6 DQN pixels + CNN (9º agente)
- **Módulo de captura**: `marioai/capture.py::GameWindowCapture` (mss + pygetwindow; fallback Windows via `PrintWindow`). Integrado em `Agent.observe_frame`, `Experiment` e `Runner`. Ver [`docs/04-captura-janela.md`](../docs/04-captura-janela.md).
- **Ambiente visual**: `marioai/gym/pixel_environment.py::ShapedPixelMarioEnv` — observation = frame capturado (84×84 grayscale uint8), reward shaping herdado de `ShapedMarioEnv`, `FrameStack(num_stack=4)` via `gym.wrappers`.
- **Agente de inferência**: `marioai/agents/dqn_pixels_agent.py::DqnPixelsAgent` — wrapper sobre `DQN.load(...)` que implementa `observe_frame`/`act` para a interface `Agent`, mantendo deque de 4 frames.
- **Script de treino**: `competition/agents/dqn_pixels/train.py` (Click CLI, `--total-timesteps` default 200k, `CnnPolicy` = NatureCNN).
- **Docs**: `competition/agents/dqn_pixels/01-teoria.md` e `02-modelagem.md`.
- **Dependências**: extra opcional `[capture]` no `pyproject.toml` (mss, pygetwindow, opencv-python, pywin32 em Windows).
- **Regra de competição**: treino usa `level_seed=42` (qualquer seed fora de `{1001, 2042, 2077, 3013, 3099}`).

Critérios de "pronto" adicionais:

- [ ] `pytest tests/test_capture.py tests/test_pixel_environment.py tests/test_dqn_pixels_agent.py` — tudo verde.
- [ ] `python -m competition.agents.dqn_pixels.train --total-timesteps 10000` completa sem crash (sanity).
- [ ] `DqnPixelsAgent` carregado de `.zip` vence a fase 1 com pelo menos `distance > 0` após treino longo.

### Critério de "pronto"

- [ ] Scripts de treino rodam end-to-end sem crash em pelo menos 50k passos (sanity check).
- [ ] Agentes carregados vencem a fase 1 em > 90% dos casos após treino completo.
- [ ] Artefatos `.zip` são reproduzíveis (mesma seed → mesmo arquivo).

---

## Etapa 5 — Execução da competição e relatório final

**Objetivo**: rodar os 8 agentes nas 5 fases, coletar métricas, publicar placar.

### Entregáveis

- **`competition/run_competition.py`** — script que:
  - Instancia cada agente (carregando artefatos treinados para os RL).
  - Passa pelo `CompetitionRunner.evaluate()`.
  - Popula `Scoreboard`.
  - Salva:
    - `competition/results/raw.csv` (uma linha por (agente, fase, métrica)).
    - `competition/results/scoreboard.md` (tabela ordenada com desempate).
    - `competition/results/per_phase.md` (tabela de agentes × fases com ✓/✗ e `time_left`).
- **`competition/RESULTS.md`** — relatório final manual:
  - Resumo da competição.
  - Placar final com comentário (quem ganhou, onde cada técnica brilhou/falhou).
  - Gráficos (opcional, ASCII ou vinculados a PNGs no `competition/results/`).
  - Retrospectiva: confirmou as hipóteses da Etapa 0? Houve surpresas?

### Critério de "pronto"

- [ ] `run_competition.py` completa em < 30 min de wall-clock (assumindo agentes já treinados).
- [ ] `scoreboard.md` tem ranking completo dos 8 agentes.
- [ ] `RESULTS.md` está escrito e reviewed.

---

## Notas de trabalho contínuo

- **Convenção de branches**: `feat/competition-etapa-N-<slug>`.
- **Convenção de commits**: prefixos `feat`, `fix`, `docs`, `test`, `refactor` (seguindo o padrão já em uso no repo).
- **CI**: cada PR roda `make lint` e `make test` via `.github/workflows/test.yaml`. Treinos pesados (Etapa 4) **não** rodam em CI — artefatos são versionados via Git LFS ou linkados externamente.
- **Bloqueadores conhecidos**:
  - Porta 4242 fixa no servidor Java impede paralelização do PPO. Se for crítico, abrir ticket/PR para suportar `-port N` (fora do escopo atual).
  - Notebook `notebooks/monte_carlo.ipynb` é anterior à refatoração — considerar arquivá-lo após a Etapa 3.

## Como atualizar este roadmap

- Marque checkboxes `- [x]` à medida que os critérios de "pronto" são atingidos.
- Atualize a tabela de visão geral (`⬜ pendente` → `🔄 em andamento` → `✅ concluída`).
- Se uma etapa revelar mudanças no escopo (ex.: Etapa 0 mostrar que uma fase precisa ser substituída), atualize `competition/README.md` e os `02-modelagem.md` afetados no mesmo PR.
