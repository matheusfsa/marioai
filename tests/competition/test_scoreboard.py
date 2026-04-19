from marioai.competition import PhaseResult, Scoreboard


def _r(phase: str, status: int, distance: float = 0.0, time_left: int = 0) -> PhaseResult:
    return PhaseResult(phase, status, distance, time_left, 0, 0, 0.0)


def test_rank_by_phases_won_first() -> None:
    board = Scoreboard()
    board.add('A', [_r('p1', 1), _r('p2', 1), _r('p3', 0)])
    board.add('B', [_r('p1', 1), _r('p2', 0), _r('p3', 0)])
    ranked = board.rank()
    assert [s.name for s in ranked] == ['A', 'B']


def test_tiebreak_by_avg_time_left_on_won_phases() -> None:
    board = Scoreboard()
    board.add('A', [_r('p1', 1, time_left=30), _r('p2', 1, time_left=40)])
    board.add('B', [_r('p1', 1, time_left=10), _r('p2', 1, time_left=20)])
    ranked = board.rank()
    assert [s.name for s in ranked] == ['A', 'B']
    assert ranked[0].avg_time_left_won == 35.0


def test_tiebreak_by_total_distance_when_times_equal() -> None:
    board = Scoreboard()
    board.add('A', [_r('p1', 1, distance=100, time_left=50), _r('p2', 0, distance=20)])
    board.add('B', [_r('p1', 1, distance=50, time_left=50), _r('p2', 0, distance=10)])
    ranked = board.rank()
    assert [s.name for s in ranked] == ['A', 'B']


def test_zero_wins_has_zero_avg_time() -> None:
    board = Scoreboard()
    board.add('A', [_r('p1', 0, time_left=99)])
    assert board.rank()[0].avg_time_left_won == 0.0


def test_to_markdown_renders_ranked_table() -> None:
    board = Scoreboard()
    board.add('A', [_r('p1', 1, time_left=50, distance=100)])
    board.add('B', [_r('p1', 0, distance=20)])
    md = board.to_markdown()
    assert '| 1 | A |' in md
    assert '| 2 | B |' in md
    assert md.splitlines()[0].startswith('| # |')
