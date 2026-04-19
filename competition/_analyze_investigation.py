"""Helper local para a Etapa 0: converte os CSV/JSON da investigação em
tabelas markdown prontas para colar em ``00-investigation.md``.

Uso:

    python competition/_analyze_investigation.py \
        --csv competition/data/investigation.csv \
        --json competition/data/feature_stats.json

Imprime no stdout, não salva nada. Não é um artefato versionado da
Etapa 0 (o artefato é o markdown); está aqui só para evitar copiar
cálculos à mão.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import click

TILE_NAMES = {
    -11: 'soft (powerup-emit)',
    -10: 'ground (hard)',
    0: 'empty',
    2: 'enemy (goomba)',
    3: 'enemy',
    4: 'enemy',
    5: 'enemy',
    6: 'enemy',
    7: 'enemy',
    8: 'enemy (wing?)',
    9: 'enemy',
    10: 'enemy',
    12: 'enemy',
    13: 'enemy',
    14: 'enemy',
    15: 'enemy',
    16: 'brick',
    20: 'hard pipe',
    21: 'brick',
    25: 'projectile',
}


def _phase_summary(rows: list[dict[str, str]]) -> str:
    by_phase: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        if r['agent'] == 'RandomAgent':
            by_phase[r['phase']].append(r)

    lines = [
        '| Fase | N | Vitórias | Distance (média ± std) | Time_left (média) | Wall-clock (s, média) | Hash 1ª cena (únicos) |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for phase in sorted(by_phase):
        rs = by_phase[phase]
        n = len(rs)
        wins = sum(1 for r in rs if int(r['status']) == 1)
        dist = [float(r['distance']) for r in rs]
        tleft = [int(r['time_left']) for r in rs if int(r['status']) == 1]
        wall = [float(r['wallclock_s']) for r in rs]
        hashes = {r['first_scene_hash'] for r in rs}
        d_mean = statistics.mean(dist) if dist else 0.0
        d_std = statistics.pstdev(dist) if len(dist) > 1 else 0.0
        tl_mean = statistics.mean(tleft) if tleft else float('nan')
        w_mean = statistics.mean(wall) if wall else 0.0
        tl_txt = f'{tl_mean:.1f}' if tleft else '—'
        lines.append(
            f'| {phase} | {n} | {wins}/{n} | {d_mean:.0f} ± {d_std:.0f} | {tl_txt} | {w_mean:.2f} | {len(hashes)} |'
        )
    return '\n'.join(lines)


def _time_left_spread(rows: list[dict[str, str]]) -> str:
    lines = ['| Fase | min | max | n distintos | valores únicos (top 5) |', '|---|---:|---:|---:|---|']
    by_phase: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        if r['agent'] == 'RandomAgent' and int(r['status']) == 1:
            by_phase[r['phase']].append(int(r['time_left']))
    for phase in sorted(by_phase):
        vals = by_phase[phase]
        uniq = sorted(set(vals))
        lines.append(
            f'| {phase} | {min(vals)} | {max(vals)} | {len(uniq)} | {", ".join(str(v) for v in uniq[:5])} |'
        )
    return '\n'.join(lines) if len(lines) > 2 else '_RandomAgent não venceu nenhuma fase — impossível medir._'


def _feature_hit_table(feature_stats: dict) -> str:
    keys = [
        'can_jump', 'on_ground',
        'enemy_1', 'enemy_2',
        'hard_1', 'hard_2',
        'soft_1', 'soft_2',
        'brick_1', 'brick_2',
        'projetil_1', 'projetil_2',
        'has_role_near_1', 'has_role_near_2',
    ]
    phases = sorted(feature_stats.keys())
    header = '| Feature | ' + ' | '.join(phases) + ' |'
    sep = '|---|' + '---:|' * len(phases)
    lines = [header, sep]
    for key in keys:
        row = [key]
        for ph in phases:
            s = feature_stats[ph]
            hits = s['feature_hits'].get(key, 0)
            total = s['feature_total'].get(key, 0)
            if total == 0:
                row.append('n/a')
            else:
                row.append(f'{100 * hits / total:.0f}% ({hits}/{total})')
        lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines)


def _tile_histogram(feature_stats: dict) -> str:
    lines = []
    for ph in sorted(feature_stats.keys()):
        s = feature_stats[ph]
        counts = {int(k): v for k, v in s['tile_counts'].items()}
        total = sum(counts.values())
        lines.append(f'### {ph} (level_type={s["level_type"]}, frames={s["frames"]})')
        lines.append('')
        lines.append('| Tile | Nome | Freq | % | Barra |')
        lines.append('|---:|---|---:|---:|---|')
        for tile, n in sorted(counts.items(), key=lambda x: -x[1])[:12]:
            pct = 100 * n / total
            bar = '█' * int(pct / 2)
            name = TILE_NAMES.get(tile, '?')
            lines.append(f'| {tile} | {name} | {n} | {pct:.1f}% | {bar} |')
        lines.append('')
    return '\n'.join(lines)


def _unique_states_table(feature_stats: dict) -> str:
    lines = ['| Fase | frames | estados únicos | estados/frame |', '|---|---:|---:|---:|']
    for ph in sorted(feature_stats.keys()):
        s = feature_stats[ph]
        frames = s['frames']
        unique = s['unique_states']
        rate = unique / frames if frames else 0
        lines.append(f'| {ph} | {frames} | {unique} | {rate:.3f} |')
    return '\n'.join(lines)


@click.command()
@click.option('--csv', 'csv_path', default='competition/data/investigation.csv', type=click.Path(exists=True, path_type=Path))
@click.option('--json', 'json_path', default='competition/data/feature_stats.json', type=click.Path(exists=True, path_type=Path))
def main(csv_path: Path, json_path: Path) -> None:
    with csv_path.open(encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    with json_path.open(encoding='utf-8') as fh:
        feature_stats = json.load(fh)

    click.echo('## Resumo por fase (RandomAgent)')
    click.echo(_phase_summary(rows))
    click.echo()
    click.echo('## Dispersão de time_left (vitórias do RandomAgent)')
    click.echo(_time_left_spread(rows))
    click.echo()
    click.echo('## Frequência de features (ExploratoryAgent)')
    click.echo(_feature_hit_table(feature_stats))
    click.echo()
    click.echo('## Histograma de tiles (ExploratoryAgent)')
    click.echo(_tile_histogram(feature_stats))
    click.echo('## Estados únicos observados')
    click.echo(_unique_states_table(feature_stats))


if __name__ == '__main__':
    main()
