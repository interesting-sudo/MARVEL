"""Summarize seeded MARVEL overlap-coordination JSONL results."""

import argparse
import json
import math
import os

import numpy as np


def load_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def episode_metrics(rows):
    return {
        'dist': np.array([row['travel_dist'] for row in rows], dtype=float),
        'dist90': np.array([row['dist_to_0_90'] for row in rows if row['dist_to_0_90']], dtype=float),
        'overlap': np.array([np.mean(row['overlap_ratio_history']) for row in rows], dtype=float),
        'success': np.array([row['success_rate'] for row in rows], dtype=float),
        'wall': np.array([row['wall_time'] for row in rows], dtype=float),
        'changed': np.array([
            item['changed_actions']
            for row in rows
            for item in row.get('coordination_history', [])
        ], dtype=float),
        'triggered': np.array([
            item['triggered_agents']
            for row in rows
            for item in row.get('coordination_history', [])
        ], dtype=float),
    }


def mean_std(values):
    if len(values) == 0:
        return float('nan'), float('nan')
    return float(np.mean(values)), float(np.std(values))


def paired_change(reference, treatment):
    common_episodes = sorted(set(row['episode'] for row in reference) & set(row['episode'] for row in treatment))
    ref = {row['episode']: row for row in reference}
    trt = {row['episode']: row for row in treatment}
    dist_delta = np.array([trt[ep]['travel_dist'] - ref[ep]['travel_dist'] for ep in common_episodes])
    overlap_delta = np.array([
        np.mean(trt[ep]['overlap_ratio_history']) - np.mean(ref[ep]['overlap_ratio_history'])
        for ep in common_episodes
    ])
    success_delta = np.array([trt[ep]['success_rate'] - ref[ep]['success_rate'] for ep in common_episodes])
    dist90_delta = np.array([
        (trt[ep]['dist_to_0_90'] or 0.0) - (ref[ep]['dist_to_0_90'] or 0.0)
        for ep in common_episodes
        if trt[ep]['dist_to_0_90'] and ref[ep]['dist_to_0_90']
    ])
    relative = dist_delta / np.maximum(
        np.array([ref[ep]['travel_dist'] for ep in common_episodes]), 1e-6
    )
    return dist_delta, overlap_delta, success_delta, dist90_delta, relative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+', help='JSONL result files or directories')
    args = parser.parse_args()

    paths = []
    for raw_path in args.paths:
        if os.path.isdir(raw_path):
            paths.extend(
                os.path.join(raw_path, name)
                for name in sorted(os.listdir(raw_path))
                if name.endswith('.jsonl')
            )
        else:
            paths.append(raw_path)

    runs = {}
    for path in paths:
        label = os.path.splitext(os.path.basename(path))[0]
        rows = load_jsonl(path)
        runs[label] = {'rows': rows, 'metrics': episode_metrics(rows)}

    baseline_label = 'off' if 'off' in runs else sorted(runs)[0]
    baseline = runs[baseline_label]

    print('| Run | Episodes | Overlap ↓ | Travel dist ↓ | Dist@90 ↓ | Success ↑ | Δ overlap | Δ dist | Δ dist% | Δ Dist@90 | Changed/step | Triggered/step |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for label in sorted(runs):
        metrics = runs[label]['metrics']
        overlap_mean, overlap_std = mean_std(metrics['overlap'])
        dist_mean, dist_std = mean_std(metrics['dist'])
        dist90_mean, dist90_std = mean_std(metrics['dist90'])
        success_mean, _ = mean_std(metrics['success'])
        changed_mean, _ = mean_std(metrics['changed'])
        triggered_mean, _ = mean_std(metrics['triggered'])
        if label == baseline_label:
            delta_overlap = delta_dist = np.array([0.0])
            delta_success = np.array([0.0])
            delta_dist90 = np.array([0.0])
            relative_dist = np.array([0.0])
        else:
            delta_dist, delta_overlap, delta_success, delta_dist90, relative_dist = paired_change(
                baseline['rows'], runs[label]['rows']
            )
        print(
            '| {} | {} | {:.4f}±{:.4f} | {:.2f}±{:.2f} | {:.2f}±{:.2f} | {:.3f} | {:+.4f} | {:+.2f} | {:+.1%} | {:+.2f} | {:.2f} | {:.2f} |'.format(
                label,
                len(runs[label]['rows']),
                overlap_mean,
                overlap_std,
                dist_mean,
                dist_std,
                dist90_mean,
                dist90_std,
                success_mean,
                float(np.mean(delta_overlap)),
                float(np.mean(delta_dist)),
                float(np.mean(relative_dist)),
                float(np.mean(delta_dist90)),
                changed_mean,
                triggered_mean,
            )
        )
        if label != baseline_label:
            overlap_se = float(np.std(delta_overlap) / math.sqrt(max(len(delta_overlap), 1)))
            dist_se = float(np.std(delta_dist) / math.sqrt(max(len(delta_dist), 1)))
            dist90_se = float(np.std(delta_dist90) / math.sqrt(max(len(delta_dist90), 1))) if len(delta_dist90) else float('nan')
            print(
                '  - paired versus `{}`: episodes={}, Δoverlap SE={:.4f}, Δdist SE={:.2f}, ΔDist@90 SE={:.2f}, Δsuccess={:+.3f}'.format(
                    baseline_label,
                    len(delta_overlap),
                    overlap_se,
                    dist_se,
                    dist90_se,
                    float(np.mean(delta_success)),
                )
            )


if __name__ == '__main__':
    main()
