"""Paired significance tests for overlap-coordination experiments."""

import argparse
import json
import os

import numpy as np


def load(path):
    with open(path) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {row['episode']: row for row in rows}


def paired_stats(delta):
    n = len(delta)
    mean = float(np.mean(delta))
    se = float(np.std(delta, ddof=1) / np.sqrt(n))
    try:
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(delta, np.zeros(n))
    except Exception:
        t_stat = mean / se if se > 0 else 0.0
        p_value = float('nan')
    return mean, se, float(t_stat), float(p_value), n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('baseline')
    parser.add_argument('treatments', nargs='+')
    args = parser.parse_args()

    base = load(os.path.join(args.directory, args.baseline + '.jsonl'))
    print('baseline = {} ({} episodes)'.format(args.baseline, len(base)))
    for name in args.treatments:
        treat = load(os.path.join(args.directory, name + '.jsonl'))
        episodes = sorted(set(base) & set(treat))
        base_overlap = np.array([np.mean(base[e]['overlap_ratio_history']) for e in episodes])
        treat_overlap = np.array([np.mean(treat[e]['overlap_ratio_history']) for e in episodes])
        base_dist = np.array([base[e]['travel_dist'] for e in episodes])
        treat_dist = np.array([treat[e]['travel_dist'] for e in episodes], dtype=float)
        mask90 = [
            i for i, e in enumerate(episodes)
            if base[e]['dist_to_0_90'] and treat[e]['dist_to_0_90']
        ]
        base90 = np.array([base[episodes[i]]['dist_to_0_90'] for i in mask90], dtype=float)
        treat90 = np.array([treat[episodes[i]]['dist_to_0_90'] for i in mask90], dtype=float)

        print('\n== {} (n={}) =='.format(name, len(episodes)))
        for label, delta in (
            ('overlap', treat_overlap - base_overlap),
            ('travel', treat_dist - base_dist),
            ('dist90', treat90 - base90),
        ):
            mean, se, t_stat, p_value, n = paired_stats(delta)
            low, high = mean - 1.96 * se, mean + 1.96 * se
            better = (delta < 0).sum()
            print('  {:8s} delta={:+.4f}  95%CI=[{:+.4f},{:+.4f}]  t={:+.2f}  p={:.4g}  improved {}/{}'.format(
                label, mean, low, high, t_stat, p_value, better, n))


if __name__ == '__main__':
    main()
