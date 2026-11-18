"""Paired diagnostics: where does the travel-distance increase come from."""

import json
import sys
from pathlib import Path

import numpy as np


def load(path):
    records = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return {record['episode']: record for record in records}


def episode_overlap(record):
    return float(np.mean(record['overlap_ratio_history']))


def main(directory, base_name, other_names):
    directory = Path(directory)
    base = load(directory / (base_name + '.jsonl'))
    for name in other_names:
        other = load(directory / (name + '.jsonl'))
        episodes = sorted(set(base) & set(other))
        base_dist = np.array([base[e]['travel_dist'] for e in episodes])
        other_dist = np.array([other[e]['travel_dist'] for e in episodes])
        base_ov = np.array([episode_overlap(base[e]) for e in episodes])
        other_ov = np.array([episode_overlap(other[e]) for e in episodes])
        delta = other_dist - base_dist
        ratio = delta / np.maximum(base_dist, 1e-6)

        print('== {} vs {} (n={}) =='.format(name, base_name, len(episodes)))
        print('  mean delta dist {:+.2f} m  (median {:+.2f}, SE {:.2f})'.format(
            delta.mean(), np.median(delta), delta.std(ddof=1) / np.sqrt(len(delta))))
        print('  relative delta  {:+.2%}  (median {:+.2%})'.format(ratio.mean(), np.median(ratio)))
        print('  episodes with shorter path: {} / {}'.format(int((delta < 0).sum()), len(delta)))
        print('  correlation(delta dist, base dist) = {:.3f}'.format(
            float(np.corrcoef(delta, base_dist)[0, 1])))
        print('  correlation(delta overlap, delta dist) = {:.3f}'.format(
            float(np.corrcoef(other_ov - base_ov, delta)[0, 1])))

        order = np.argsort(base_dist)
        chunks = np.array_split(order, 4)
        for index, chunk in enumerate(chunks):
            if len(chunk) == 0:
                continue
            print('    quartile {}: base {:6.1f} m  delta {:+.2f} m ({:+.1%})  delta overlap {:+.3f}'.format(
                index + 1,
                base_dist[chunk].mean(),
                delta[chunk].mean(),
                ratio[chunk].mean(),
                (other_ov - base_ov)[chunk].mean(),
            ))
        worst = np.argsort(-ratio)[:5]
        print('  worst relative detours: ' + ', '.join(
            'ep{} {:+.1%} ({:.0f}->{:.0f} m)'.format(
                episodes[i], ratio[i], base_dist[i], other_dist[i]) for i in worst))
        print()


if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else 'results/overlap_coord/final100'
    base = sys.argv[2] if len(sys.argv) > 2 else 'off'
    others = sys.argv[3:] or ['full_l30', 'fov_l30']
    main(root, base, others)
