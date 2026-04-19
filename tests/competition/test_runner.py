from unittest.mock import MagicMock

import pytest

from marioai.competition import CompetitionRunner, PhaseConfig, PhaseResult
from marioai.competition.runner import DeterministicAgent


class _StubAgent:
    """Agente que não é usado (o Runner é mockado)."""

    def reset(self) -> None: ...
    def sense(self, state: dict) -> None: ...
    def act(self) -> list[int]:
        return [0, 0, 0, 0, 0]
    def give_rewards(self, *a, **kw) -> None: ...
    def observe_frame(self, frame) -> None: ...


class _StochasticAgent(_StubAgent):
    def __init__(self) -> None:
        self.deterministic_calls = 0

    def set_deterministic(self) -> None:
        self.deterministic_calls += 1


def _fake_task(reward: dict) -> MagicMock:
    task = MagicMock(name='Task')
    task.reward = reward
    return task


def _patch_runner(monkeypatch: pytest.MonkeyPatch, rewards_by_phase: list[dict]) -> None:
    """Substitui ``Runner`` em competition.runner para evitar o servidor Java."""
    calls = {'i': 0}

    def _fake_runner_cls(agent, task, **kwargs):
        idx = calls['i']
        calls['i'] += 1

        def _run():
            task.reward = rewards_by_phase[idx]

        r = MagicMock(name='Runner')
        r.run = _run
        return r

    monkeypatch.setattr('marioai.competition.runner.Runner', _fake_runner_cls)


def test_evaluate_returns_one_result_per_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    rewards = [
        {'status': 1, 'distance': 100.0, 'timeLeft': 50, 'coins': 3, 'marioMode': 2},
        {'status': 0, 'distance': 40.0, 'timeLeft': 0, 'coins': 0, 'marioMode': 0},
    ]
    phases = [
        PhaseConfig('p1', 0, 0, 1, 2, 60),
        PhaseConfig('p2', 5, 0, 2, 2, 80),
    ]
    _patch_runner(monkeypatch, rewards)
    task = _fake_task({})

    results = CompetitionRunner(_StubAgent(), phases=phases).evaluate(task=task)

    assert [r.phase for r in results] == ['p1', 'p2']
    assert results[0].won is True
    assert results[0].time_left == 50
    assert results[1].won is False
    assert results[1].distance == 40.0


def test_evaluate_calls_set_deterministic_on_stochastic_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_runner(monkeypatch, [{'status': 0, 'distance': 0, 'timeLeft': 0, 'coins': 0, 'marioMode': 0}])
    agent = _StochasticAgent()
    assert isinstance(agent, DeterministicAgent)
    CompetitionRunner(agent, phases=[PhaseConfig('p1', 0, 0, 1, 2, 60)]).evaluate(task=_fake_task({}))
    assert agent.deterministic_calls == 1


def test_evaluate_disconnects_owned_task(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_runner(monkeypatch, [{'status': 0, 'distance': 0, 'timeLeft': 0, 'coins': 0, 'marioMode': 0}])

    fake_task = _fake_task({})
    monkeypatch.setattr('marioai.competition.runner.Task', lambda: fake_task)

    CompetitionRunner(_StubAgent(), phases=[PhaseConfig('p1', 0, 0, 1, 2, 60)]).evaluate()
    fake_task.disconnect.assert_called_once()


def test_evaluate_does_not_disconnect_external_task(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_runner(monkeypatch, [{'status': 0, 'distance': 0, 'timeLeft': 0, 'coins': 0, 'marioMode': 0}])
    task = _fake_task({})
    CompetitionRunner(_StubAgent(), phases=[PhaseConfig('p1', 0, 0, 1, 2, 60)]).evaluate(task=task)
    task.disconnect.assert_not_called()


def test_evaluate_disconnects_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_runner_cls(agent, task, **kwargs):
        r = MagicMock()
        r.run.side_effect = RuntimeError('boom')
        return r

    monkeypatch.setattr('marioai.competition.runner.Runner', _fake_runner_cls)
    fake_task = _fake_task({})
    monkeypatch.setattr('marioai.competition.runner.Task', lambda: fake_task)

    with pytest.raises(RuntimeError):
        CompetitionRunner(_StubAgent(), phases=[PhaseConfig('p1', 0, 0, 1, 2, 60)]).evaluate()
    fake_task.disconnect.assert_called_once()


def test_phase_result_won_only_for_status_1() -> None:
    assert PhaseResult('x', 1, 0, 0, 0, 0, 0).won is True
    assert PhaseResult('x', 0, 0, 0, 0, 0, 0).won is False
    assert PhaseResult('x', 2, 0, 0, 0, 0, 0).won is False
