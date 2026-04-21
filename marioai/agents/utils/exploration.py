from __future__ import annotations

__all__ = ['decay_epsilon']


def decay_epsilon(
    schedule: str,
    ep: int,
    total: int,
    *,
    start: float = 1.0,
    end: float = 0.1,
) -> float:
    """Return ε for episode ``ep`` out of ``total`` under ``schedule``.

    Supported schedules:
      - ``'linear'``: linear interpolation from ``start`` to ``end``.
      - ``'exponential'``: geometric decay so that ε(total-1) == ``end``.
      - ``'constant'``: always ``start``.
    """
    if total <= 1:
        return end
    frac = min(max(ep / (total - 1), 0.0), 1.0)
    if schedule == 'linear':
        return start + (end - start) * frac
    if schedule == 'exponential':
        ratio = (end / start) ** frac
        return start * ratio
    if schedule == 'constant':
        return start
    raise ValueError(f'Unknown schedule: {schedule!r}')
