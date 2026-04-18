import random

from marioai import core
from marioai.core import sensing

__all__ = ['ExploratoryAgent']


class ExploratoryAgent(core.Agent):
    """Agent that builds a proximity-based state every 24 frames.

    Currently the derived state is only recorded, not used to pick actions —
    :meth:`act` still returns a random forward move. The state dict is kept
    on ``self.state`` so subclasses can consume it.
    """

    def __init__(
        self,
        window_size: int = 4,
        max_dist: int = 2,
        player_pos: int = 11,
    ) -> None:
        super().__init__()
        self.frames = 0
        self.window_size = window_size
        self.max_dist = max_dist
        self.player_pos = player_pos
        self.state: dict[str, object] | None = None
        self.objects = sensing.DEFAULT_OBJECTS

    def _build_state(self) -> None:
        if self.level_scene is None:
            return

        scene = self.level_scene.copy()
        ground_pos = sensing.get_ground(
            scene,
            bool(self.on_ground),
            window_size=self.window_size,
            player_pos=self.player_pos,
        )
        state: dict[str, object] = {
            'episode_starts': bool((scene != 0).any()),
            'on_ground': self.on_ground,
            'can_jump': self.can_jump,
            'episode_over': self.episode_over,
        }

        if self.frames % 24 == 0:
            for o_name, o_values in self.objects.items():
                for dist in range(1, self.max_dist + 1):
                    state[f'{o_name}_{dist}'] = sensing.is_near(scene, o_values, dist, player_pos=self.player_pos)
            for dist in range(1, self.max_dist + 1):
                state[f'has_role_near_{dist}'] = sensing.has_role_near(scene, ground_pos, dist, player_pos=self.player_pos)

        self.frames += 1
        self.state = state

    def act(self) -> list[int]:
        self._build_state()
        return [0, 1, 0, random.randint(0, 1), random.randint(0, 1)]
