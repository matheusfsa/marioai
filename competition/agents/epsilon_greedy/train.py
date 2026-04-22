"""Train the ε-greedy tabular agent for the competition.

Usage::

    python -m competition.agents.epsilon_greedy.train --n-episodes 3000

O servidor Java precisa estar rodando na porta 4242 antes de iniciar
(ver ``CLAUDE.md``). Hiperparâmetros default seguem
``competition/agents/epsilon_greedy/02-modelagem.md``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

from marioai.agents import EpsilonGreedyAgent
from marioai.core import Task

logger = logging.getLogger(__name__)

_DEFAULT_SAVE_PATH = Path(__file__).with_name('eps_agent.pkl')
_RESERVED_SEEDS = {1001, 2042, 2077, 3013, 3099}


@click.command(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--n-episodes', default=3000, show_default=True, type=int)
@click.option('--epsilon-start', default=1.0, show_default=True, type=float)
@click.option('--epsilon-end', default=0.05, show_default=True, type=float)
@click.option('--decay-fraction', default=0.8, show_default=True, type=float)
@click.option('--first-visit/--every-visit', default=True, show_default=True)
@click.option('--level-difficulty', default=3, show_default=True, type=int)
@click.option('--level-type', default=0, show_default=True, type=int)
@click.option('--level-seed', default=42, show_default=True, type=int, help=f'Must not clash with {sorted(_RESERVED_SEEDS)}.')
@click.option('--mario-mode', default=2, show_default=True, type=int)
@click.option('--time-limit', default=180, show_default=True, type=int)
@click.option('--max-fps', default=720, show_default=True, type=int)
@click.option('--visualization/--no-visualization', default=False, show_default=True)
@click.option('--save-path', default=str(_DEFAULT_SAVE_PATH), show_default=True, type=click.Path(dir_okay=False))
def train(
    n_episodes: int,
    epsilon_start: float,
    epsilon_end: float,
    decay_fraction: float,
    first_visit: bool,
    level_difficulty: int,
    level_type: int,
    level_seed: int,
    mario_mode: int,
    time_limit: int,
    max_fps: int,
    visualization: bool,
    save_path: str,
) -> None:
    """Treina o EpsilonGreedyAgent e salva a Q-table em ``save-path``."""
    if level_seed in _RESERVED_SEEDS:
        raise click.BadParameter(
            f'--level-seed {level_seed} conflita com as seeds de avaliação {sorted(_RESERVED_SEEDS)}; escolha outra.',
        )

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')

    agent = EpsilonGreedyAgent(
        n_episodes=n_episodes,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        decay_fraction=decay_fraction,
        first_visit=first_visit,
    )
    task = Task()
    try:
        agent.fit(
            task=task,
            level_difficulty=level_difficulty,
            level_type=level_type,
            level_seed=level_seed,
            mario_mode=mario_mode,
            time_limit=time_limit,
            max_fps=max_fps,
            visualization=visualization,
        )
    finally:
        agent.save(save_path)
        logger.info('saved Q-table to %s (%d states)', save_path, len(agent._Q))


if __name__ == '__main__':
    train()
