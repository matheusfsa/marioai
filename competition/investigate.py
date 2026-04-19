"""Etapa 0 — investigação empírica do servidor MarioAI.

Roda agentes baseline (RandomAgent, ExploratoryAgent) nas 5 fases definidas
em ``competition/README.md`` e coleta métricas para calibrar as decisões de
modelagem dos agentes da competição.

Saídas:

- ``competition/data/investigation.csv`` — uma linha por episódio com
  métricas do FitnessResult e wall-clock.
- ``competition/data/feature_stats.json`` — agregados de features
  (frequência de ``can_jump``, ``on_ground``, ``enemy_d``, ``hard_d``,
  ``has_role_near_d``), histograma de tiles do ``level_scene`` por
  ``level_type`` e número de estados únicos vistos pelo
  ``ExploratoryAgent``.

Uso:

    python competition/investigate.py --random-runs 20 --explore-runs 5

O servidor Java é encerrado e reaberto entre cada bloco de fases para
evitar carregar estado entre execuções (porta 4242 fixa).
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import click
import numpy as np

from marioai.agents import ExploratoryAgent, RandomAgent
from marioai.core import Runner, Task

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'

PHASES: list[dict[str, Any]] = [
    dict(name='1-easy', level_difficulty=0, level_type=0, level_seed=1001, mario_mode=2, time_limit=60),
    dict(name='2-medium-A', level_difficulty=5, level_type=0, level_seed=2042, mario_mode=2, time_limit=80),
    dict(name='3-medium-B', level_difficulty=8, level_type=1, level_seed=2077, mario_mode=2, time_limit=100),
    dict(name='4-hard-A', level_difficulty=15, level_type=2, level_seed=3013, mario_mode=1, time_limit=120),
    dict(name='5-hard-B', level_difficulty=20, level_type=3, level_seed=3099, mario_mode=0, time_limit=120),
]


@dataclasses.dataclass
class EpisodeRecord:
    agent: str
    phase: str
    run: int
    seed: int
    level_difficulty: int
    level_type: int
    mario_mode: int
    status: int
    distance: float
    time_left: int
    coins: int
    final_mario_mode: int
    wallclock_s: float
    first_scene_hash: str


class _SceneHashAgent(RandomAgent):
    """RandomAgent que registra o hash do primeiro level_scene observado."""

    def __init__(self) -> None:
        super().__init__()
        self.first_scene_hash: str | None = None

    def reset(self) -> None:
        super().reset()
        self.first_scene_hash = None

    def sense(self, state: dict[str, Any]) -> None:
        super().sense(state)
        if self.first_scene_hash is None and state.get('level_scene') is not None:
            scene = state['level_scene']
            self.first_scene_hash = hashlib.sha1(scene.tobytes()).hexdigest()[:12]


class _StatsExploratoryAgent(ExploratoryAgent):
    """ExploratoryAgent que acumula estatísticas de features e tiles.

    Registra:
    - tile_counts: Counter[int] com a contagem de cada tile observado em
      todos os ``level_scene`` que viu;
    - feature_hits / feature_total: para cada feature booleana, quantas
      vezes apareceu ``True`` e quantas amostras foram observadas;
    - unique_states: set[tuple] com a tupla de features (já convertidas
      para bool/None), útil para estimar o tamanho da Q-table tabular;
    - first_scene_hash: hash do primeiro level_scene visto no episódio.
    """

    feature_keys = (
        'can_jump',
        'on_ground',
        'soft_1', 'soft_2',
        'hard_1', 'hard_2',
        'enemy_1', 'enemy_2',
        'brick_1', 'brick_2',
        'projetil_1', 'projetil_2',
        'has_role_near_1', 'has_role_near_2',
    )

    def __init__(self) -> None:
        super().__init__()
        self.tile_counts: Counter[int] = Counter()
        self.feature_hits: Counter[str] = Counter()
        self.feature_total: Counter[str] = Counter()
        self.unique_states: set[tuple[Any, ...]] = set()
        self.first_scene_hash: str | None = None
        self.frames_seen: int = 0

    def reset(self) -> None:
        super().reset()
        self.first_scene_hash = None

    def _record(self, state: dict[str, Any]) -> None:
        scene = state.get('level_scene')
        if scene is not None:
            if self.first_scene_hash is None:
                self.first_scene_hash = hashlib.sha1(scene.tobytes()).hexdigest()[:12]
            for tile, n in zip(*np.unique(scene, return_counts=True), strict=True):
                self.tile_counts[int(tile)] += int(n)

        snapshot: list[Any] = []
        for key in self.feature_keys:
            if key not in state:
                snapshot.append(None)
                continue
            val = state[key]
            snapshot.append(val)
            if val is None:
                continue
            self.feature_total[key] += 1
            if bool(val):
                self.feature_hits[key] += 1

        self.unique_states.add(tuple(snapshot))
        self.frames_seen += 1

    def sense(self, state: dict[str, Any]) -> None:
        super().sense(state)
        self._record(state)


def _make_runner(agent: Any, task: Task, phase: dict[str, Any], max_fps: int) -> Runner:
    return Runner(
        agent,
        task,
        max_fps=max_fps,
        level_difficulty=phase['level_difficulty'],
        level_type=phase['level_type'],
        level_seed=phase['level_seed'],
        mario_mode=phase['mario_mode'],
        time_limit=phase['time_limit'],
        visualization=False,
    )


def _record_episode(
    agent_name: str,
    phase: dict[str, Any],
    run: int,
    task: Task,
    wallclock_s: float,
    first_scene_hash: str | None,
) -> EpisodeRecord:
    reward = task.reward
    return EpisodeRecord(
        agent=agent_name,
        phase=phase['name'],
        run=run,
        seed=phase['level_seed'],
        level_difficulty=phase['level_difficulty'],
        level_type=phase['level_type'],
        mario_mode=phase['mario_mode'],
        status=int(reward.get('status', 0) or 0),
        distance=float(reward.get('distance', 0) or 0),
        time_left=int(reward.get('timeLeft', 0) or 0),
        coins=int(reward.get('coins', 0) or 0),
        final_mario_mode=int(reward.get('marioMode', 0) or 0),
        wallclock_s=round(wallclock_s, 3),
        first_scene_hash=first_scene_hash or '',
    )


def _run_random_block(records: list[EpisodeRecord], runs_per_phase: int, max_fps: int) -> None:
    """Roda RandomAgent N vezes em cada fase. Reabre o servidor entre fases."""
    for phase in PHASES:
        click.echo(f'[random] phase={phase["name"]} runs={runs_per_phase}')
        task = Task()
        try:
            for run in range(runs_per_phase):
                agent = _SceneHashAgent()
                runner = _make_runner(agent, task, phase, max_fps)
                t0 = time.time()
                runner.run()
                wall = time.time() - t0
                records.append(_record_episode('RandomAgent', phase, run, task, wall, agent.first_scene_hash))
        finally:
            task.disconnect()


def _run_explore_block(
    records: list[EpisodeRecord],
    feature_stats: dict[str, Any],
    runs_per_phase: int,
    max_fps: int,
) -> None:
    """Roda ExploratoryAgent N vezes em cada fase, agregando estatísticas."""
    for phase in PHASES:
        click.echo(f'[explore] phase={phase["name"]} runs={runs_per_phase}')
        task = Task()
        per_phase_tile_counts: Counter[int] = Counter()
        per_phase_feature_hits: Counter[str] = Counter()
        per_phase_feature_total: Counter[str] = Counter()
        per_phase_unique_states: set[tuple[Any, ...]] = set()
        per_phase_frames = 0
        try:
            for run in range(runs_per_phase):
                agent = _StatsExploratoryAgent()
                runner = _make_runner(agent, task, phase, max_fps)
                t0 = time.time()
                runner.run()
                wall = time.time() - t0
                records.append(_record_episode('ExploratoryAgent', phase, run, task, wall, agent.first_scene_hash))
                per_phase_tile_counts.update(agent.tile_counts)
                per_phase_feature_hits.update(agent.feature_hits)
                per_phase_feature_total.update(agent.feature_total)
                per_phase_unique_states.update(agent.unique_states)
                per_phase_frames += agent.frames_seen
        finally:
            task.disconnect()

        feature_stats[phase['name']] = {
            'level_type': phase['level_type'],
            'frames': per_phase_frames,
            'tile_counts': {str(k): v for k, v in sorted(per_phase_tile_counts.items())},
            'feature_hits': dict(per_phase_feature_hits),
            'feature_total': dict(per_phase_feature_total),
            'unique_states': len(per_phase_unique_states),
        }


def _write_csv(records: Iterable[EpisodeRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [f.name for f in dataclasses.fields(EpisodeRecord)]
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(dataclasses.asdict(rec))


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


@click.command()
@click.option('--random-runs', default=20, type=int, help='Episódios do RandomAgent por fase.')
@click.option('--explore-runs', default=5, type=int, help='Episódios do ExploratoryAgent por fase.')
@click.option('--max-fps', default=720, type=int, help='Cap de FPS do Runner. 720 = rápido, 24 = realtime.')
@click.option(
    '--out-dir',
    default=str(DATA_DIR),
    type=click.Path(file_okay=False, path_type=Path),
    help='Onde gravar investigation.csv e feature_stats.json.',
)
def main(random_runs: int, explore_runs: int, max_fps: int, out_dir: Path) -> None:
    """Coleta dados de baseline para a Etapa 0 do roadmap da competição."""
    out_dir = Path(out_dir)
    records: list[EpisodeRecord] = []
    feature_stats: dict[str, Any] = {}

    if random_runs > 0:
        _run_random_block(records, random_runs, max_fps)
    if explore_runs > 0:
        _run_explore_block(records, feature_stats, explore_runs, max_fps)

    _write_csv(records, out_dir / 'investigation.csv')
    _write_json(feature_stats, out_dir / 'feature_stats.json')

    click.echo(f'wrote {len(records)} rows to {out_dir / "investigation.csv"}')
    click.echo(f'wrote feature stats for {len(feature_stats)} phases to {out_dir / "feature_stats.json"}')


if __name__ == '__main__':
    main()
