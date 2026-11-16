#!/usr/bin/env python3
"""Evaluate the supplied MARVEL checkpoint for Tables I--III.

The original test_driver.py is configured for one setting only.  This runner
uses the same TestWorker and checkpoint, but writes per-map results so that a
long experiment can be resumed and audited.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class Config:
    table: str
    agents: int
    fov_deg: int
    sensor_range_m: int

    @property
    def name(self) -> str:
        return f"table_{self.table}_n{self.agents}_fov{self.fov_deg}_r{self.sensor_range_m}"


CONFIGS = (
    Config("I", 2, 120, 10), Config("I", 4, 120, 10), Config("I", 8, 120, 10),
    Config("II", 4, 90, 10), Config("II", 4, 180, 10),
    Config("III", 4, 120, 12), Config("III", 4, 120, 15),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values))  # author's np.std convention


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_completed(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open(newline="") as stream:
        return {int(row["episode"]) for row in csv.DictReader(stream)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reproduction/results")
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--configs", nargs="*", help="config names; defaults to all seven")
    parser.add_argument("--correct-overlap", action="store_true",
                        help="use actual sensor_range for overlap. Omit for exact author-code behavior.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; rerun with --device cpu")

    from test_parameter import (EMBEDDING_DIM, NODE_INPUT_DIM, NUM_ANGLES_BIN)
    from utils.model import PolicyNet
    from utils.test_worker import TestWorker
    from utils.env_test import Env

    # The original Env selects os.listdir() order.  Sorting makes the set of
    # map-index/seed pairs stable across filesystems without editing author code.
    original_import = Env.import_ground_truth
    def sorted_import(self, map_index):
        map_dir = "maps_test"
        original_listdir = os.listdir
        try:
            os.listdir = lambda path: sorted(original_listdir(path))
            return original_import(self, map_index)
        finally:
            os.listdir = original_listdir
    Env.import_ground_truth = sorted_import

    if args.correct_overlap:
        # Author code uses training SENSOR_RANGE=10 here.  This only changes the
        # overlap metric, and is used as a sensitivity check for Table III.
        def create_sensing_mask(self, location, heading):
            from shapely.geometry import Point, Polygon
            from skimage.draw import polygon as sk_polygon
            from utils.utils import get_cell_position_from_coords, get_free_and_connected_map
            cell = get_cell_position_from_coords(location, self.env.belief_info)
            start = (heading - self.fov / 2 + 360) % 360
            end = (heading + self.fov / 2) % 360
            angles = (np.linspace(start, end, 20) if start <= end else
                      np.concatenate((np.linspace(start, 360, 10), np.linspace(0, end, 10))))
            points = [Point(cell)] + [Point(cell[0] + self.sensor_range / 0.4 * np.cos(np.radians(a)),
                                           cell[1] + self.sensor_range / 0.4 * np.sin(np.radians(a))) for a in angles] + [Point(cell)]
            x, y = Polygon(points).exterior.xy
            rr, cc = sk_polygon(np.rint(y).astype(int), np.rint(x).astype(int), shape=self.env.ground_truth.shape)
            mask = np.zeros_like(self.env.ground_truth)
            connected = get_free_and_connected_map(location, self.env.belief_info)
            mask[rr, cc] = connected[rr, cc] == connected[cell[1], cell[0]]
            return mask
        TestWorker.create_sensing_mask = create_sensing_mask

    device = torch.device(args.device)
    checkpoint_path = root / "load_model/MARVEL/checkpoint.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN).to(device)
    policy.load_state_dict(checkpoint["policy_model"])
    policy.eval()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    selected = [c for c in CONFIGS if not args.configs or c.name in args.configs]
    if args.configs and len(selected) != len(args.configs):
        unknown = sorted(set(args.configs) - {c.name for c in CONFIGS})
        raise ValueError(f"unknown config(s): {unknown}")

    for config in selected:
        csv_path = output / f"{config.name}{'_corrected_overlap' if args.correct_overlap else ''}.csv"
        if args.overwrite and csv_path.exists():
            csv_path.unlink()
        completed = read_completed(csv_path)
        is_new = not csv_path.exists()
        with csv_path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("episode", "seed", "travel_dist", "explored_rate", "success", "dist_to_90", "overlap_mean"))
            if is_new:
                writer.writeheader()
            for episode in range(args.episodes):
                if episode in completed:
                    continue
                episode_seed = args.seed + episode
                set_seed(episode_seed)
                worker = TestWorker(0, policy, episode, config.agents, config.fov_deg,
                                    config.sensor_range_m, config.sensor_range_m * 0.9,
                                    device=device, save_image=False, greedy=True)
                with torch.inference_mode():
                    worker.run_episode()
                metrics = worker.perf_metrics
                writer.writerow({
                    "episode": episode, "seed": episode_seed,
                    "travel_dist": metrics["travel_dist"], "explored_rate": metrics["explored_rate"],
                    "success": int(metrics["success_rate"]), "dist_to_90": metrics["dist_to_0_90"] or "",
                    "overlap_mean": float(np.mean(metrics["overlap_ratio_history"])),
                })
                stream.flush()
                print(f"{config.name}: episode {episode + 1}/{args.episodes}", flush=True)

        with csv_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        travel = [float(r["travel_dist"]) for r in rows]
        coverage90 = [float(r["dist_to_90"]) for r in rows if r["dist_to_90"]]
        overlap = [float(r["overlap_mean"]) for r in rows]
        summary = {
            "config": asdict(config), "episodes": len(rows), "seed_base": args.seed,
            "checkpoint_sha256": sha256(checkpoint_path), "correct_overlap": args.correct_overlap,
            "trajectory_mean_std": mean_std(travel), "coverage90_mean_std": mean_std(coverage90),
            "overlap_mean_std": mean_std(overlap),
            "success_percent": 100 * sum(int(r["success"]) for r in rows) / len(rows),
        }
        (output / f"{config.name}{'_corrected_overlap' if args.correct_overlap else ''}.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
