"""Agrega resultados de vários agentes e produz o placar final."""

from __future__ import annotations

from dataclasses import dataclass, field

from .runner import PhaseResult

__all__ = ['AgentScore', 'Scoreboard']


@dataclass
class AgentScore:
    name: str
    results: list[PhaseResult]

    @property
    def phases_won(self) -> int:
        return sum(1 for r in self.results if r.won)

    @property
    def avg_time_left_won(self) -> float:
        won = [r for r in self.results if r.won]
        if not won:
            return 0.0
        return sum(r.time_left for r in won) / len(won)

    @property
    def total_distance(self) -> float:
        return sum(r.distance for r in self.results)

    def rank_key(self) -> tuple[int, float, float]:
        """Chave de ordenação decrescente: mais vitórias > mais tempo > mais distância."""
        return (self.phases_won, self.avg_time_left_won, self.total_distance)


@dataclass
class Scoreboard:
    scores: list[AgentScore] = field(default_factory=list)

    def add(self, agent_name: str, results: list[PhaseResult]) -> None:
        self.scores.append(AgentScore(agent_name, list(results)))

    def rank(self) -> list[AgentScore]:
        """Retorna agentes ordenados pelas regras de desempate da competição."""
        return sorted(self.scores, key=lambda s: s.rank_key(), reverse=True)

    def to_markdown(self) -> str:
        ranked = self.rank()
        lines = [
            '| # | Agente | Fases vencidas | `time_left` médio (vitórias) | Distância total |',
            '|---|---|---:|---:|---:|',
        ]
        for i, s in enumerate(ranked, start=1):
            lines.append(
                f'| {i} | {s.name} | {s.phases_won}/{len(s.results)} | '
                f'{s.avg_time_left_won:.1f} | {s.total_distance:.1f} |'
            )
        return '\n'.join(lines)
