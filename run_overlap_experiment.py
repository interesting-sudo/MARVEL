"""Run seeded MARVEL baseline and inference-only coordination comparisons."""

import argparse
import json
import os
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['off', 'fov', 'frontier', 'full'], required=True)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--n-agents', type=int, default=8)
    parser.add_argument('--fov', type=float, default=120)
    parser.add_argument('--sensor-range', type=float, default=10)
    parser.add_argument('--seed', type=int, default=20261118)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def to_builtin(value):
    import numpy as np

    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    args = parse_args()
    os.environ['MARVEL_COORD_MODE'] = args.mode

    import numpy as np
    import ray
    import torch

    from test_parameter import EMBEDDING_DIM, NODE_INPUT_DIM, NUM_ANGLES_BIN, load_path
    from utils.model import PolicyNet
    from utils.test_worker import TestWorker

    torch.set_num_threads(1)
    network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN)
    checkpoint = torch.load(os.path.join(load_path, 'checkpoint.pth'), map_location='cpu')
    network.load_state_dict(checkpoint['policy_model'])
    weights = network.state_dict()

    @ray.remote(num_cpus=1)
    class EpisodeWorker:
        def __init__(self):
            torch.set_num_threads(1)
            self.network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN)

        def run(self, weights, episode):
            import numpy as np
            import torch

            np.random.seed(args.seed + episode)
            torch.manual_seed(args.seed + episode)
            self.network.load_state_dict(weights)
            start_time = time.time()
            worker = TestWorker(
                0,
                self.network,
                episode,
                args.n_agents,
                args.fov,
                args.sensor_range,
                args.sensor_range * 0.9,
                device='cpu',
                save_image=False,
                greedy=True,
            )
            worker.run_episode()
            metrics = worker.perf_metrics
            metrics['episode'] = episode
            metrics['seed'] = args.seed + episode
            metrics['wall_time'] = time.time() - start_time
            return to_builtin(metrics)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    ray.init(num_cpus=args.workers, include_dashboard=False, log_to_driver=False)
    actors = [EpisodeWorker.remote() for _ in range(min(args.workers, args.episodes))]
    pending = []
    episode_queue = list(range(args.start, args.start + args.episodes))
    results = []

    for actor, episode in zip(actors, episode_queue[: len(actors)]):
        pending.append(actor.run.remote(weights, episode))
    next_index = len(actors)

    try:
        while pending:
            done, pending = ray.wait(pending)
            for ref in done:
                result = ray.get(ref)
                results.append(result)
                print(
                    'episode={} success={} dist={:.3f} overlap={:.4f} time={:.2f}s'.format(
                        result['episode'],
                        result['success_rate'],
                        result['travel_dist'],
                        np.mean(result['overlap_ratio_history']),
                        result['wall_time'],
                    ),
                    flush=True,
                )
            while len(pending) < len(actors) and next_index < len(episode_queue):
                episode = episode_queue[next_index]
                actor = actors[next_index % len(actors)]
                pending.append(actor.run.remote(weights, episode))
                next_index += 1
    finally:
        ray.shutdown()

    results.sort(key=lambda item: item['episode'])
    with open(args.output, 'w') as handle:
        for result in results:
            handle.write(json.dumps(result) + '\n')

    def values(key):
        return np.array([result[key] for result in results], dtype=float)

    flat_overlap = np.array([
        value
        for result in results
        for value in result['overlap_ratio_history']
    ], dtype=float)
    dist_90 = np.array([
        result['dist_to_0_90'] for result in results if result['dist_to_0_90']
    ], dtype=float)
    coordination = [
        item
        for result in results
        for item in result.get('coordination_history', [])
    ]
    summary = {
        'config': vars(args),
        'episodes': len(results),
        'travel_dist_mean': float(values('travel_dist').mean()),
        'travel_dist_std': float(values('travel_dist').std()),
        'success_rate': float(values('success_rate').mean()),
        'explored_rate_mean': float(values('explored_rate').mean()),
        'dist_to_0_90_mean': float(dist_90.mean()) if len(dist_90) else None,
        'dist_to_0_90_std': float(dist_90.std()) if len(dist_90) else None,
        'overlap_ratio_mean': float(flat_overlap.mean()),
        'overlap_ratio_std': float(flat_overlap.std()),
        'episode_overlap_mean': float(np.mean([
            np.mean(result['overlap_ratio_history']) for result in results
        ])),
        'wall_time_mean': float(values('wall_time').mean()),
        'changed_actions_per_step': float(np.mean([
            item['changed_actions'] for item in coordination
        ])) if coordination else 0.0,
        'triggered_agents_per_step': float(np.mean([
            item['triggered_agents'] for item in coordination
        ])) if coordination else 0.0,
    }
    summary_path = os.path.splitext(args.output)[0] + '.summary.json'
    with open(summary_path, 'w') as handle:
        json.dump(to_builtin(summary), handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
