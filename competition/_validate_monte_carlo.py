"""Validação de Etapa 3 (Monte Carlo): roda o agente treinado nas 5 fases.

Também mede a taxa de vitória na fase 1 em ``--runs`` execuções (critério
do ROADMAP: > 90%). O servidor Java precisa estar rodando na 4242.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from competition.agents.monte_carlo import ARTIFACT_PATH, load_agent
from marioai.competition import PHASES, CompetitionRunner


def _print_all_phases() -> None:
    print('\n## MonteCarloAgent — 5 fases (1 run cada)\n')
    print('| fase | status | distance | time_left | wallclock_s |')
    print('|---|---|---|---|---|')
    for phase in PHASES:
        agent = load_agent()
        runner = CompetitionRunner(agent, phases=[phase], max_fps=720, visualization=False)
        t0 = time.time()
        try:
            res = runner.evaluate()[0]
            print(f'| {phase.name} | {res.status} | {res.distance:.1f} | {res.time_left} | {res.wallclock_s} |')
        except Exception as e:  # noqa: BLE001
            print(f'| {phase.name} | ERR | {e!r} | - | {time.time() - t0:.1f} |')


def _win_rate_phase1(runs: int) -> float:
    phase = PHASES[0]
    wins = 0
    print(f'\n## Win-rate fase {phase.name} ({runs} runs)\n')
    for i in range(runs):
        agent = load_agent()
        runner = CompetitionRunner(agent, phases=[phase], max_fps=720, visualization=False)
        res = runner.evaluate()[0]
        won = res.status == 1
        wins += int(won)
        print(f'- run {i}: status={res.status} distance={res.distance:.1f} won={won}')
    rate = wins / runs
    print(f'\n**Win-rate**: {wins}/{runs} = {rate:.0%}')
    return rate


if __name__ == '__main__':
    if not Path(ARTIFACT_PATH).exists():
        sys.exit(f'artefato não encontrado: {ARTIFACT_PATH}. Rode competition.agents.monte_carlo.train primeiro.')
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    _print_all_phases()
    _win_rate_phase1(runs)
