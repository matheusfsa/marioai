"""A* planner over ``level_scene``.

Modelagem: ``competition/agents/astar/02-modelagem.md``.
"""

from __future__ import annotations

import heapq

import numpy as np

from marioai.core import Agent

__all__ = ['AStarAgent']


FORWARD: list[int] = [0, 1, 0, 0, 0]
FORWARD_JUMP: list[int] = [0, 1, 0, 1, 0]
FORWARD_JUMP_SPEED: list[int] = [0, 1, 0, 1, 1]
JUMP: list[int] = [0, 0, 0, 1, 0]
BACKWARD: list[int] = [1, 0, 0, 0, 0]

HARD_TILES = frozenset({-10, 20, 42})
BRICK_TILES = frozenset({16, 21})
ENEMY_TILES = frozenset({2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15})
SOFT_TILES = frozenset({-11})

ENEMY_PENALTY = 100
JUMP_COST = 2
WALK_COST = 1

PLAYER_POS = 11
GOAL_COL = 21
REPLAN_EVERY = 12
MAX_JUMP_COLS = 3
MAX_JUMP_ROWS = 3


Cell = tuple[int, int]  # (row, col)


def _is_blocked(tile: int) -> bool:
    return tile in HARD_TILES or tile in BRICK_TILES


def _tile_penalty(tile: int) -> int:
    if tile in ENEMY_TILES:
        return ENEMY_PENALTY
    return 0


def _neighbors(
    scene: np.ndarray,
    node: Cell,
    can_jump: bool,
) -> list[tuple[Cell, int]]:
    rows, cols = scene.shape
    r, c = node
    out: list[tuple[Cell, int]] = []
    # walk and fall (4-connectivity including vertical drop)
    for dr, dc in ((0, 1), (0, -1), (1, 0)):
        nr, nc = r + dr, c + dc
        if not (0 <= nr < rows and 0 <= nc < cols):
            continue
        tile = int(scene[nr, nc])
        if _is_blocked(tile):
            continue
        cost = WALK_COST + _tile_penalty(tile)
        out.append(((nr, nc), cost))
    # jump arcs: up-right/up-left/up within range
    if can_jump:
        for dr in range(-MAX_JUMP_ROWS, 1):
            for dc in range(-MAX_JUMP_COLS, MAX_JUMP_COLS + 1):
                if dr == 0 and dc == 0:
                    continue
                if dr == 0 and abs(dc) == 1:
                    continue  # already covered as walk
                if dr == 1 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                tile = int(scene[nr, nc])
                if _is_blocked(tile):
                    continue
                cost = JUMP_COST + _tile_penalty(tile)
                out.append(((nr, nc), cost))
    return out


def _heuristic(node: Cell, goal: Cell) -> int:
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])


def plan(
    level_scene: np.ndarray,
    start: Cell = (PLAYER_POS, PLAYER_POS),
    goal: Cell = (PLAYER_POS, GOAL_COL),
    can_jump: bool = True,
) -> list[Cell]:
    """Classical A* over the tile grid. Returns the path from ``start`` to ``goal`` inclusive.

    Returns an empty list if no path is found.
    """
    if start == goal:
        return [start]
    open_heap: list[tuple[int, int, Cell]] = []
    counter = 0
    heapq.heappush(open_heap, (_heuristic(start, goal), counter, start))
    came_from: dict[Cell, Cell] = {}
    g_score: dict[Cell, int] = {start: 0}
    closed: set[Cell] = set()
    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path.reverse()
            return path
        closed.add(current)
        for nbr, step_cost in _neighbors(level_scene, current, can_jump):
            if nbr in closed:
                continue
            tentative = g_score[current] + step_cost
            if tentative < g_score.get(nbr, 1 << 30):
                g_score[nbr] = tentative
                came_from[nbr] = current
                counter += 1
                f = tentative + _heuristic(nbr, goal)
                heapq.heappush(open_heap, (f, counter, nbr))
    return []


def path_to_action(current: Cell, next_cell: Cell, can_jump: bool) -> list[int]:
    dr = next_cell[0] - current[0]
    dc = next_cell[1] - current[1]
    if dr < 0:
        if not can_jump:
            return FORWARD if dc >= 0 else BACKWARD
        if dc == 0:
            return JUMP
        return FORWARD_JUMP_SPEED
    if dc > 0:
        return FORWARD
    if dc < 0:
        return BACKWARD
    return FORWARD


class AStarAgent(Agent):
    """Replans a path every :data:`REPLAN_EVERY` frames and follows it."""

    def __init__(self) -> None:
        super().__init__()
        self._plan_path: list[Cell] = []
        self._plan_index: int = 0
        self._frames_since_plan: int = 0

    def reset(self) -> None:
        super().reset()
        self._plan_path = []
        self._plan_index = 0
        self._frames_since_plan = 0

    def _needs_replan(self) -> bool:
        if not self._plan_path:
            return True
        if self._frames_since_plan >= REPLAN_EVERY:
            return True
        if self._plan_index >= len(self._plan_path) - 1:
            return True
        return False

    def _plan(self, scene: np.ndarray) -> list[Cell]:
        return plan(scene, (PLAYER_POS, PLAYER_POS), (PLAYER_POS, GOAL_COL), bool(self.can_jump))

    def act(self) -> list[int]:
        if self.level_scene is None:
            return FORWARD
        scene = self.level_scene
        self._plan_path = self._plan(scene)
        self._plan_index = 0
        self._frames_since_plan = 0
        if len(self._plan_path) < 2:
            return FORWARD_JUMP if self.can_jump else FORWARD
        current = self._plan_path[0]
        next_cell = self._plan_path[1]
        return path_to_action(current, next_cell, bool(self.can_jump))
