"""A* planner over ``level_scene``.

Modelagem: ``competition/agents/astar/02-modelagem.md``.

O grafo respeita gravidade: uma célula só é "estável" se houver um tile sólido
logo abaixo. Arestas de andar conectam apenas células estáveis adjacentes;
cair e pular são arestas explícitas.
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
MAX_JUMP_COLS = 5
MAX_JUMP_ROWS = 4


Cell = tuple[int, int]  # (row, col)


def _is_blocked(tile: int) -> bool:
    return tile in HARD_TILES or tile in BRICK_TILES


def _tile_penalty(tile: int) -> int:
    if tile in ENEMY_TILES:
        return ENEMY_PENALTY
    return 0


def _cell_passable(scene: np.ndarray, r: int, c: int) -> bool:
    """Cell where Mario's body can be. Bloqueio próprio + clearance de cabeça."""
    rows, cols = scene.shape
    if not (0 <= r < rows and 0 <= c < cols):
        return False
    if _is_blocked(int(scene[r, c])):
        return False
    if r - 1 >= 0 and _is_blocked(int(scene[r - 1, c])):
        return False
    return True


def _has_support(scene: np.ndarray, r: int, c: int) -> bool:
    """True se há sólido logo abaixo (ou fora da janela — assume chão)."""
    rows, _ = scene.shape
    if r + 1 >= rows:
        return True
    return _is_blocked(int(scene[r + 1, c]))


def _is_standable(scene: np.ndarray, r: int, c: int) -> bool:
    return _cell_passable(scene, r, c) and _has_support(scene, r, c)


def _fall_landing(scene: np.ndarray, r_start: int, c: int) -> int | None:
    """Linha onde Mario pousa ao entrar na coluna ``c`` em ``r_start`` sem suporte.

    Retorna ``None`` se cair para fora da janela (pit mortal).
    """
    rows, _ = scene.shape
    r = r_start
    while r < rows:
        if _is_blocked(int(scene[r, c])):
            return None
        if _has_support(scene, r, c):
            return r
        r += 1
    return None


def _jump_corridor_clear(scene: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    """Corredor em Γ: sobe até apex, atravessa, desce. Simplifica o arco real."""
    apex = min(r0, r1) - 1 if r0 == r1 else min(r0, r1)
    apex = max(apex, 0)
    for r in range(apex, r0 + 1):
        if not _cell_passable(scene, r, c0):
            return False
    for r in range(apex, r1 + 1):
        if not _cell_passable(scene, r, c1):
            return False
    lo, hi = (c0, c1) if c0 <= c1 else (c1, c0)
    for c in range(lo, hi + 1):
        if not _cell_passable(scene, apex, c):
            return False
    return True


def _neighbors(
    scene: np.ndarray,
    node: Cell,
    can_jump: bool,
) -> list[tuple[Cell, int]]:
    r, c = node
    out: list[tuple[Cell, int]] = []
    rows, cols = scene.shape
    # andar lateral: a célula alvo precisa ser passável; se sem suporte, cai.
    for dc in (-1, 1):
        nc = c + dc
        if not (0 <= nc < cols):
            continue
        if not _cell_passable(scene, r, nc):
            continue
        if _has_support(scene, r, nc):
            tile = int(scene[r, nc])
            out.append(((r, nc), WALK_COST + _tile_penalty(tile)))
        else:
            landing = _fall_landing(scene, r, nc)
            if landing is None:
                continue
            fall = landing - r
            tile = int(scene[landing, nc])
            out.append(((landing, nc), WALK_COST + fall + _tile_penalty(tile)))
    # arcos de pulo: qualquer célula estável dentro do alcance com corredor livre
    if can_jump:
        for dr in range(-MAX_JUMP_ROWS, MAX_JUMP_ROWS + 1):
            for dc in range(-MAX_JUMP_COLS, MAX_JUMP_COLS + 1):
                if dr == 0 and abs(dc) <= 1:
                    continue  # coberto por andar
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if not _is_standable(scene, nr, nc):
                    continue
                if not _jump_corridor_clear(scene, r, c, nr, nc):
                    continue
                tile = int(scene[nr, nc])
                cost = JUMP_COST + abs(dc) + abs(dr) + _tile_penalty(tile)
                out.append(((nr, nc), cost))
    return out


def _heuristic(node: Cell, goal: Cell) -> int:
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])


def _find_start(scene: np.ndarray) -> Cell:
    """Mario é desenhado em ``(PLAYER_POS, PLAYER_POS)``. Se não é estável
    (ex.: mid-air), procura a primeira célula estável abaixo na coluna."""
    if _is_standable(scene, PLAYER_POS, PLAYER_POS):
        return (PLAYER_POS, PLAYER_POS)
    landing = _fall_landing(scene, PLAYER_POS, PLAYER_POS)
    if landing is not None:
        return (landing, PLAYER_POS)
    return (PLAYER_POS, PLAYER_POS)


def _find_goal(scene: np.ndarray, goal_col: int = GOAL_COL) -> Cell:
    """Célula-alvo: estável na coluna ``goal_col`` mais próxima de ``PLAYER_POS``.
    Se a coluna for inteiramente pit, tenta progressivamente colunas à esquerda
    até encontrar uma estável; fallback final é ``(PLAYER_POS, PLAYER_POS + 1)``.
    """
    rows, cols = scene.shape
    target = min(max(goal_col, 0), cols - 1)
    for col in range(target, PLAYER_POS, -1):
        candidates = [r for r in range(rows) if _is_standable(scene, r, col)]
        if candidates:
            return (min(candidates, key=lambda r: abs(r - PLAYER_POS)), col)
    return (PLAYER_POS, min(PLAYER_POS + 1, cols - 1))


def plan(
    level_scene: np.ndarray,
    start: Cell = (PLAYER_POS, PLAYER_POS),
    goal: Cell = (PLAYER_POS, GOAL_COL),
    can_jump: bool = True,
) -> list[Cell]:
    """A* clássico sobre o grafo com gravidade. Para-se ao atingir ``goal[1]``
    (coluna) — o grafo é orientado à direita e linha exata raramente importa.
    """
    if start == goal:
        return [start]
    goal_col = goal[1]
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
        if current[1] >= goal_col:
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
    # subir: arco de pulo para frente ou JUMP vertical
    if dr < 0:
        if not can_jump:
            return FORWARD if dc >= 0 else BACKWARD
        if dc == 0:
            return JUMP
        return FORWARD_JUMP_SPEED
    # qualquer salto para frente que atravessa gap (dc>1) ou cai forte (dr>=2)
    if dc > 1 and can_jump:
        return FORWARD_JUMP_SPEED if dc >= 3 else FORWARD_JUMP
    if dc > 0:
        if dr >= 2 and can_jump:
            return FORWARD_JUMP
        return FORWARD
    if dc < 0:
        return BACKWARD
    return FORWARD


STUCK_FRAMES = 24
STUCK_EPSILON = 0.5


class AStarAgent(Agent):
    """Replaneja todo frame e segue o plano; com stuck-detection."""

    def __init__(self) -> None:
        super().__init__()
        self._plan_path: list[Cell] = []
        self._plan_index: int = 0
        self._last_x: float | None = None
        self._stuck_counter: int = 0

    def reset(self) -> None:
        super().reset()
        self._plan_path = []
        self._plan_index = 0
        self._last_x = None
        self._stuck_counter = 0

    def _plan(self, scene: np.ndarray) -> list[Cell]:
        start = _find_start(scene)
        goal = _find_goal(scene, GOAL_COL)
        return plan(scene, start, goal, bool(self.can_jump))

    def _update_stuck(self) -> None:
        mario = self.mario_floats
        if mario is None:
            return
        x = float(mario[0])
        if self._last_x is None:
            self._last_x = x
            return
        if x - self._last_x < STUCK_EPSILON:
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0
        self._last_x = x

    def act(self) -> list[int]:
        if self.level_scene is None:
            return FORWARD
        self._update_stuck()
        scene = self.level_scene
        # se travado há muitos frames, força um pulo para a frente
        if self._stuck_counter >= STUCK_FRAMES and self.can_jump:
            self._stuck_counter = 0
            return FORWARD_JUMP_SPEED
        self._plan_path = self._plan(scene)
        self._plan_index = 0
        if len(self._plan_path) < 2:
            return FORWARD_JUMP if self.can_jump else FORWARD
        current = self._plan_path[0]
        next_cell = self._plan_path[1]
        return path_to_action(current, next_cell, bool(self.can_jump))
