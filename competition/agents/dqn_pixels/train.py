"""Train the DQN-pixels (CNN) agent for the competition.

Usage::

    python -m competition.agents.dqn_pixels.train --total-timesteps 200000

Training speed is limited by the single-instance server (port 4242 is fixed,
so ``SubprocVecEnv`` is not an option). Expect ~3-5h of wall-clock for the
default 200k steps; crank up ``--total-timesteps`` for real results.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)

_DEFAULT_SAVE_PATH = Path(__file__).with_name('dqn_pixels.zip')


@click.command(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--total-timesteps', default=200_000, show_default=True, type=int)
@click.option('--learning-rate', default=1e-4, show_default=True, type=float)
@click.option('--buffer-size', default=50_000, show_default=True, type=int)
@click.option('--learning-starts', default=5_000, show_default=True, type=int)
@click.option('--batch-size', default=32, show_default=True, type=int)
@click.option('--gamma', default=0.99, show_default=True, type=float)
@click.option('--exploration-fraction', default=0.5, show_default=True, type=float)
@click.option('--exploration-final-eps', default=0.05, show_default=True, type=float)
@click.option('--target-update-interval', default=1_000, show_default=True, type=int)
@click.option('--seed', default=42, show_default=True, type=int)
@click.option('--level-difficulty', default=3, show_default=True, type=int, help='Must differ from all 5 evaluation seeds.')
@click.option('--level-type', default=0, show_default=True, type=int)
@click.option('--level-seed', default=42, show_default=True, type=int, help='Must differ from 1001/2042/2077/3013/3099.')
@click.option('--mario-mode', default=2, show_default=True, type=int)
@click.option('--time-limit', default=100, show_default=True, type=int)
@click.option('--max-fps', default=720, show_default=True, type=int)
@click.option('--window-title', default='Mario Intelligent', show_default=True, help='Substring matched against the window title. Default is the exact title the Java server sets; override only if you rename it.')
@click.option('--capture-backend', default='mss', show_default=True, type=click.Choice(['mss', 'win32']))
@click.option('--save-path', default=str(_DEFAULT_SAVE_PATH), show_default=True, type=click.Path(dir_okay=False))
@click.option('--log-interval', default=10, show_default=True, type=int)
def train(
    total_timesteps: int,
    learning_rate: float,
    buffer_size: int,
    learning_starts: int,
    batch_size: int,
    gamma: float,
    exploration_fraction: float,
    exploration_final_eps: float,
    target_update_interval: int,
    seed: int,
    level_difficulty: int,
    level_type: int,
    level_seed: int,
    mario_mode: int,
    time_limit: int,
    max_fps: int,
    window_title: str,
    capture_backend: str,
    save_path: str,
    log_interval: int,
) -> None:
    """Train a CNN-policy DQN on captured Mario frames."""
    # Heavy imports deferred so `--help` stays snappy and unit tests don't need them.
    from stable_baselines3 import DQN
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    from marioai.capture import GameWindowCapture
    from marioai.gym import ShapedPixelMarioEnv

    _RESERVED_SEEDS = {1001, 2042, 2077, 3013, 3099}
    if level_seed in _RESERVED_SEEDS:
        raise click.BadParameter(
            f'--level-seed {level_seed} clashes with the competition evaluation seeds {_RESERVED_SEEDS}; '
            'training on evaluation seeds is forbidden by the competition rules.',
        )

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')

    capture = GameWindowCapture(
        window_title=window_title,
        grayscale=True,
        resize=(84, 84),
        backend=capture_backend,  # type: ignore[arg-type]
    )
    # SB3's ``VecFrameStack`` runs after the gym→gymnasium compat shim (shimmy)
    # so it's agnostic to ``MarioEnv`` using the classic gym API. ``gym.wrappers.FrameStack``
    # would require the 0.26 (obs, info)/5-tuple API, which this env does not implement.
    base_env = ShapedPixelMarioEnv(
        capture=capture,
        level_difficulty=level_difficulty,
        level_type=level_type,
        level_seed=level_seed,
        mario_mode=mario_mode,
        time_limit=time_limit,
        max_fps=max_fps,
    )
    env = DummyVecEnv([lambda: base_env])
    env = VecFrameStack(env, n_stack=4)

    model = DQN(
        'CnnPolicy',
        env,
        verbose=1,
        seed=seed,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=gamma,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=exploration_final_eps,
        max_grad_norm=10,
    )
    try:
        model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
    finally:
        model.save(save_path)
        logger.info('saved model to %s', save_path)
        env.close()


if __name__ == '__main__':
    train()
