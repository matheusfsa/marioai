"""CLI: ``python -m marioai.competition`` — lista fases ou roda um baseline."""

from __future__ import annotations

import click

from marioai.agents import RandomAgent

from . import PHASES, CompetitionRunner, Scoreboard


@click.group()
def cli() -> None:
    """Utilitários da competição."""


@cli.command('phases')
def list_phases() -> None:
    """Lista as 5 fases de avaliação."""
    click.echo(f'{"name":<12} {"ld":>3} {"lt":>3} {"seed":>6} {"mm":>3} {"tl":>4}')
    for p in PHASES:
        click.echo(
            f'{p.name:<12} {p.level_difficulty:>3} {p.level_type:>3} '
            f'{p.level_seed:>6} {p.mario_mode:>3} {p.time_limit:>4}'
        )


@cli.command('run-random')
@click.option('--max-fps', default=720, type=int, show_default=True)
@click.option('--visualization/--no-visualization', default=False)
def run_random(max_fps: int, visualization: bool) -> None:
    """Roda o RandomAgent nas 5 fases e imprime o placar."""
    runner = CompetitionRunner(RandomAgent(), max_fps=max_fps, visualization=visualization)
    results = runner.evaluate()
    board = Scoreboard()
    board.add('RandomAgent', results)
    click.echo(board.to_markdown())


if __name__ == '__main__':
    cli()
