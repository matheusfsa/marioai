from __future__ import annotations

import logging

import click

from marioai import agents, core
from marioai.capture import GameWindowCapture

logger = logging.getLogger(__name__)

environment_options = [
    click.option('--level_difficulty', '-ld', 'level_difficulty', default=0, type=int),
    click.option('--mario_mode', '-mm', 'mario_mode', default=0, type=int),
    click.option('--time_limit', '-tl', 'time_limit', default=0, type=int),
    click.option('--max_fps', '-fps', 'max_fps', default=720, type=int),
]

capture_options = [
    click.option('--capture/--no-capture', 'capture_enabled', default=False, help='Capture the game window each step.'),
    click.option('--capture-window', 'capture_window', default='Mario', show_default=True, help='Substring of the window title to capture.'),
    click.option('--capture-grayscale/--no-capture-grayscale', 'capture_grayscale', default=False),
    click.option('--capture-resize', 'capture_resize', default=None, help="Resize captured frames, e.g. '84x84'."),
    click.option('--capture-backend', 'capture_backend', type=click.Choice(['mss', 'win32']), default='mss', show_default=True),
]


def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


def _build_capture(
    enabled: bool,
    window: str,
    grayscale: bool,
    resize: str | None,
    backend: str,
) -> GameWindowCapture | None:
    """Construct a :class:`GameWindowCapture` from CLI flags, or ``None``."""
    if not enabled:
        return None
    parsed_resize: tuple[int, int] | None = None
    if resize:
        try:
            w_str, h_str = resize.lower().split('x')
            parsed_resize = (int(w_str), int(h_str))
        except ValueError as exc:
            raise click.BadParameter(f"--capture-resize must look like '84x84', got {resize!r}") from exc
    return GameWindowCapture(
        window_title=window,
        grayscale=grayscale,
        resize=parsed_resize,
        backend=backend,  # type: ignore[arg-type]
    )


@click.group()
def cli() -> None:
    """CLI entry point for the marioai Python client."""


@click.command(name='mc')
@add_options(environment_options)
@add_options(capture_options)
@click.option('--response_delay', '-rd', 'response_delay', default=0, type=int)
def monte_carlo(
    level_difficulty: int,
    mario_mode: int,
    time_limit: int,
    response_delay: int,
    max_fps: int,
    capture_enabled: bool,
    capture_window: str,
    capture_grayscale: bool,
    capture_resize: str | None,
    capture_backend: str,
) -> None:
    """Train a tabular Monte Carlo agent."""
    capture = _build_capture(capture_enabled, capture_window, capture_grayscale, capture_resize, capture_backend)
    task = core.Task(max_dist=4)
    mc_model = agents.MonteCarloAgent(
        n_samples=2,
        discount=0.9,
        min_epsilon=0.3,
        reward_threshold=2,
        reward_increment=0.5,
    )
    mc_model.fit(
        task=task,
        level_difficulty=level_difficulty,
        mario_mode=mario_mode,
        time_limit=time_limit,
        response_delay=response_delay,
        max_fps=max_fps,
        capture=capture,
    )


@click.command(name='dqn')
@add_options(environment_options)
@click.option('--total_timesteps', '-tt', 'total_timesteps', default=100000, type=int)
@click.option('--log_interval', '-li', 'log_interval', default=4, type=int)
def dqn_model(
    level_difficulty: int,
    mario_mode: int,
    time_limit: int,
    max_fps: int,
    total_timesteps: int,
    log_interval: int,
) -> None:
    """Train a DQN agent via stable-baselines3 (symbolic 22x22 observation)."""
    from stable_baselines3 import DQN

    from marioai.gym import MarioEnv

    env = MarioEnv(
        level_difficulty=level_difficulty,
        mario_mode=mario_mode,
        time_limit=time_limit,
        max_fps=max_fps,
    )
    model = DQN(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=0.0001,
        buffer_size=1_000_000,
        learning_starts=50_000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10_000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10,
    )
    model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
    env.close()


@click.command(name='random')
@add_options(environment_options)
@add_options(capture_options)
@click.option('--episodes', '-n', 'episodes', default=1, show_default=True, type=int)
def random_agent(
    level_difficulty: int,
    mario_mode: int,
    time_limit: int,
    max_fps: int,
    episodes: int,
    capture_enabled: bool,
    capture_window: str,
    capture_grayscale: bool,
    capture_resize: str | None,
    capture_backend: str,
) -> None:
    """Run a RandomAgent for ``--episodes`` episodes (great for capture smoke-tests)."""
    capture = _build_capture(capture_enabled, capture_window, capture_grayscale, capture_resize, capture_backend)
    task = core.Task()
    runner = core.Runner(
        agents.RandomAgent(),
        task,
        max_fps=max_fps,
        level_difficulty=level_difficulty,
        mario_mode=mario_mode,
        time_limit=time_limit,
        capture=capture,
    )
    try:
        for _ in range(episodes):
            runner.run()
    finally:
        runner.close()


cli.add_command(monte_carlo)
cli.add_command(dqn_model)
cli.add_command(random_agent)


if __name__ == '__main__':
    cli()
