# CLAUDE.md

Guia rápido para agentes rodando tarefas neste repo.

## Rodando o servidor Java (necessário para testes end-to-end)

O servidor Java é o backend do simulador e escuta na porta **4242**. Sem ele, `CompetitionRunner.evaluate()` e `_validate_etapa2.py` travam no `connect`.

```bash
cd /home/matheus/claude_workspace/marioai/marioai/core/server
java ch.idsia.scenarios.MainRun -server on &
# ~2s de boot; confirme com: ss -tlnp | grep 4242
```

Para parar: `kill <pid>` (o `&` retorna o PID). A porta não libera até o processo morrer — se preso, `fuser -k 4242/tcp`.

Java disponível no sistema: OpenJDK 21. Os `.class` já estão em `marioai/core/server/ch/idsia/scenarios/`; não precisa recompilar.

## Rodando testes e lint

```bash
PYTHONPATH=. pytest tests/ -q                  # suite inteira
PYTHONPATH=. pytest tests/test_astar.py -q     # um arquivo
ruff check .                                   # lint
ruff format .                                  # auto-format
```

`python3` do sistema já tem `numpy`, `pytest`, `ruff`. O pacote `marioai` **não** está instalado como editable — use `PYTHONPATH=.` da raiz.

Testes que dependem de dependências extras (ignorar se a importação falhar):
- `tests/test_capture.py`, `tests/test_pixel_environment.py`, `tests/test_dqn_pixels_agent.py` — exigem o extra `[capture]` (mss, opencv, pygetwindow).

## Validação da Etapa 2 (A*)

Com o servidor Java up:

```bash
cd /home/matheus/claude_workspace/marioai
PYTHONPATH=. python3 competition/_validate_etapa2.py
```

Imprime tabela markdown de `status | distance | time_left | wallclock_s` por fase. Cada fase leva ~2–4 s com `max_fps=720`.

## Git

- Remotes SSH (`git@github.com:...`). Nunca trocar para HTTPS.
- Commits seguem Conventional Commits (`feat(competition):`, `refactor:`, `test:`, etc.) — ver `git log`.
- Branches: `feat/competition-etapa-N-<slug>`.

## Estrutura relevante

- `marioai/agents/` — agentes (A*, Monte Carlo, DQN pixels, etc.).
- `marioai/competition/` — `PhaseConfig`, `CompetitionRunner`, `Scoreboard`.
- `competition/` — scripts, docs por agente (`agents/<nome>/{01-teoria,02-modelagem}.md`), `ROADMAP.md`.
- `marioai/core/server/` — servidor Java bundleado (`.class` + assets).
