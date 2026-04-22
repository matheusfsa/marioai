"""Train the Monte Carlo tabular agent for the competition.

Usage::

    python -m competition.agents.monte_carlo.train --n-samples 2000

O servidor Java precisa estar rodando na porta 4242 antes de iniciar
(ver ``CLAUDE.md``). Hiperparâmetros default seguem
``competition/agents/monte_carlo/02-modelagem.md``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

from marioai.agents import MonteCarloAgent
from marioai.core import Task

logger = logging.getLogger(__name__)

_DEFAULT_SAVE_PATH = Path(__file__).with_name('mc_agent.pkl')
_RESERVED_SEEDS = {1001, 2042, 2077, 3013, 3099}


@click.command(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--n-samples', default=2000, show_default=True, type=int)
@click.option('--discount', default=0.9, show_default=True, type=float)
@click.option('--min-epsilon', default=0.1, show_default=True, type=float)
@click.option('--reward-threshold', default=0.0, show_default=True, type=float)
@click.option('--reward-increment', default=0.5, show_default=True, type=float)
@click.option('--level-difficulty', default=3, show_default=True, type=int)
@click.option('--level-type', default=0, show_default=True, type=int)
@click.option('--level-seed', default=42, show_default=True, type=int, help=f'Must not clash with {sorted(_RESERVED_SEEDS)}.')
@click.option('--mario-mode', default=2, show_default=True, type=int)
@click.option('--time-limit', default=180, show_default=True, type=int)
@click.option('--max-fps', default=720, show_default=True, type=int)
@click.option('--visualization/--no-visualization', default=False, show_default=True)
@click.option('--save-path', default=str(_DEFAULT_SAVE_PATH), show_default=True, type=click.Path(dir_okay=False))
def train(
    n_samples: int,
    discount: float,
    min_epsilon: float,
    reward_threshold: float,
    reward_increment: float,
    level_difficulty: int,
    level_type: int,
    level_seed: int,
    mario_mode: int,
    time_limit: int,
    max_fps: int,
    visualization: bool,
    save_path: str,
) -> None:
    """Treina o MonteCarloAgent e salva a Q-table em ``save-path``."""
    if level_seed in _RESERVED_SEEDS:
        raise click.BadParameter(
            f'--level-seed {level_seed} conflita com as seeds de avaliação {sorted(_RESERVED_SEEDS)}; escolha outra.',
        )

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')

    agent = MonteCarloAgent(
        n_samples=n_samples,
        discount=discount,
        min_epsilon=min_epsilon,
        reward_threshold=reward_threshold,
        reward_increment=reward_increment,
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
