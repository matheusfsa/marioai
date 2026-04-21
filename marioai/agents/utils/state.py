from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ['State']


class State:
    """Hashable wrapper around a dict so it can be used as a Q-table key.

    Accepts numpy arrays and lists as values; they are hashed by their bytes
    or tuple form respectively.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.state_attrs: list[str] = []
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.state_attrs.append(key)

    def __repr__(self) -> str:
        parts = [f'{attr}={getattr(self, attr)}' for attr in self.state_attrs]
        return 'State(' + ', '.join(parts) + ')'

    def __hash__(self) -> int:
        values = []
        for attr in self.state_attrs:
            value = getattr(self, attr)
            if isinstance(value, np.ndarray):
                values.append(value.tobytes())
            elif isinstance(value, list):
                values.append(tuple(value))
            else:
                values.append(value)
        return hash(tuple(values))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return NotImplemented
        if self.state_attrs != other.state_attrs:
            return False
        for attr in self.state_attrs:
            a = getattr(self, attr)
            b = getattr(other, attr)
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                if np.any(a != b):
                    return False
            elif a != b:
                return False
        return True
