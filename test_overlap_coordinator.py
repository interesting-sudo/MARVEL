import unittest

import numpy as np

from utils.overlap_coordinator import OverlapCoordinator


class FakeRobot:
    def __init__(self, robot_id):
        self.id = robot_id
        self.location = np.array([0.0, 0.0])


def candidate(robot_id, action_index, mask_key, frontiers=(), logp=0.0, position=None, mask=None,
              node_index=None, heading_index=0):
    item = {
        'robot_id': robot_id,
        'rank': action_index,
        'action_index': action_index,
        'logp': logp,
        'node_index': action_index if node_index is None else node_index,
        'heading_index': heading_index,
        'position': np.array([0.0 if mask_key == 'left' else 1.0, 0.0]) if position is None else np.array(position, dtype=float),
        'execution_heading': 0.0,
        'mask_key': mask_key,
        'frontiers': frozenset(frontiers),
    }
    if mask is not None:
        item['mask'] = np.array(mask, dtype=bool)
    return item


class OverlapCoordinatorTest(unittest.TestCase):
    def setUp(self):
        self.masks = {
            'left': np.array([[1, 1, 0, 0]], dtype=bool),
            'right': np.array([[0, 0, 1, 1]], dtype=bool),
        }

    def sensing_mask(self, position, heading):
        return self.masks['left' if position[0] == 0 else 'right']

    def make_coordinator(self, use_overlap=True, use_frontier=True, max_policy_drop=float('inf'),
                         max_detour_ratio=float('inf'), same_waypoint_only=False,
                         tabu_waypoint_steps=0, detour_budget=float('inf'),
                         min_frontier_ratio=0.0, separation_weight=0.0, separation_radius=0.0,
                         separation_horizon=float('inf')):
        return OverlapCoordinator(
            [FakeRobot(0), FakeRobot(1)],
            self.sensing_mask,
            fov=120,
            sensor_range=10,
            top_k=2,
            overlap_weight=2.0,
            frontier_weight=1.0,
            switch_margin=0.05,
            hysteresis_bonus=0.0,
            overlap_trigger=0.1,
            max_policy_drop=max_policy_drop,
            max_detour_ratio=max_detour_ratio,
            same_waypoint_only=same_waypoint_only,
            tabu_waypoint_steps=tabu_waypoint_steps,
            detour_budget=detour_budget,
            min_frontier_ratio=min_frontier_ratio,
            separation_weight=separation_weight,
            separation_radius=separation_radius,
            separation_horizon=separation_horizon,
            use_overlap=use_overlap,
            use_frontier=use_frontier,
        )

    def test_fov_penalty_selects_nonoverlapping_candidate(self):
        lists = [
            [candidate(0, 0, 'left')],
            [candidate(1, 0, 'left'), candidate(1, 1, 'right')],
        ]
        selected, info = self.make_coordinator(use_frontier=False).select(lists, step=0)
        self.assertEqual(selected[1]['action_index'], 1)
        self.assertEqual(info['changed_actions'], 1)

    def test_frontier_penalty_selects_unique_frontier(self):
        lists = [
            [candidate(0, 0, 'left', frontiers=['shared'])],
            [
                candidate(1, 0, 'left', frontiers=['shared']),
                candidate(1, 1, 'right', frontiers=['unique']),
            ],
        ]
        selected, _ = self.make_coordinator(use_overlap=False).select(lists, step=0)
        self.assertEqual(selected[1]['action_index'], 1)

    def test_low_overlap_keeps_original_policy_action(self):
        lists = [
            [candidate(0, 0, 'left')],
            [candidate(1, 0, 'right'), candidate(1, 1, 'left', logp=-0.1)],
        ]
        selected, info = self.make_coordinator(use_frontier=False).select(lists, step=0)
        self.assertEqual(selected[1]['action_index'], 0)
        self.assertEqual(info['triggered_agents'], 0)

    def test_policy_trust_region_blocks_low_logp_candidate(self):
        lists = [
            [candidate(0, 0, 'left')],
            [
                candidate(1, 0, 'left', logp=0.0),
                candidate(1, 1, 'right', logp=-2.0),
            ],
        ]
        selected, _ = self.make_coordinator(
            use_frontier=False, max_policy_drop=1.0
        ).select(lists, step=0)
        self.assertEqual(selected[1]['action_index'], 0)

    def test_detour_guard_blocks_distant_candidate(self):
        left_mask = np.array([[1, 1, 0, 0]], dtype=bool)
        right_mask = np.array([[0, 0, 1, 1]], dtype=bool)
        lists = [
            [candidate(0, 0, 'left', mask=left_mask)],
            [
                candidate(1, 0, 'left', position=(1.0, 0.0), mask=left_mask),
                candidate(1, 1, 'right', position=(5.0, 0.0), mask=right_mask),
            ],
        ]
        selected, _ = self.make_coordinator(
            use_frontier=False, max_detour_ratio=1.5
        ).select(lists, step=0)
        self.assertEqual(selected[1]['action_index'], 0)

    def test_detour_guard_allows_near_candidate(self):
        left_mask = np.array([[1, 1, 0, 0]], dtype=bool)
        right_mask = np.array([[0, 0, 1, 1]], dtype=bool)
        lists = [
            [candidate(0, 0, 'left', mask=left_mask)],
            [
                candidate(1, 0, 'left', position=(1.0, 0.0), mask=left_mask),
                candidate(1, 1, 'right', position=(1.2, 0.0), mask=right_mask),
            ],
        ]
        selected, _ = self.make_coordinator(
            use_frontier=False, max_detour_ratio=1.5
        ).select(lists, step=0)
        self.assertEqual(selected[1]['action_index'], 1)

    def test_same_waypoint_only_allows_heading_change(self):
        left_mask = np.array([[1, 1, 0, 0]], dtype=bool)
        right_mask = np.array([[0, 0, 1, 1]], dtype=bool)
        lists = [
            [candidate(0, 0, 'left', mask=left_mask)],
            [
                candidate(1, 0, 'left', node_index=0, heading_index=0, mask=left_mask),
                candidate(1, 1, 'right', node_index=0, heading_index=1, mask=right_mask),
            ],
        ]
        selected, _ = self.make_coordinator(
            use_frontier=False, same_waypoint_only=True
        ).select(lists, step=0)
        self.assertEqual(selected[1]['action_index'], 1)

    def test_same_waypoint_only_blocks_waypoint_change(self):
        left_mask = np.array([[1, 1, 0, 0]], dtype=bool)
        right_mask = np.array([[0, 0, 1, 1]], dtype=bool)
        lists = [
            [candidate(0, 0, 'left', mask=left_mask)],
            [
                candidate(1, 0, 'left', node_index=0, mask=left_mask),
                candidate(1, 1, 'right', node_index=1, mask=right_mask),
            ],
        ]
        selected, _ = self.make_coordinator(
            use_frontier=False, same_waypoint_only=True
        ).select(lists, step=0)
        self.assertEqual(selected[1]['action_index'], 0)

    def test_tabu_blocks_returning_to_recent_waypoint(self):
        left_mask = np.array([[1, 1, 0, 0]], dtype=bool)
        right_mask = np.array([[0, 0, 1, 1]], dtype=bool)
        lists = [
            [candidate(0, 0, 'left', mask=left_mask)],
            [
                candidate(1, 0, 'left', node_index=0, mask=left_mask),
                candidate(1, 1, 'right', node_index=7, mask=right_mask),
            ],
        ]
        coordinator = self.make_coordinator(use_frontier=False, tabu_waypoint_steps=2)
        coordinator.recent_waypoints[1] = [7]
        selected, _ = coordinator.select(lists, step=0)
        self.assertEqual(selected[1]['action_index'], 0)

    def test_tabu_allows_unvisited_waypoint(self):
        left_mask = np.array([[1, 1, 0, 0]], dtype=bool)
        right_mask = np.array([[0, 0, 1, 1]], dtype=bool)
        lists = [
            [candidate(0, 0, 'left', mask=left_mask)],
            [
                candidate(1, 0, 'left', node_index=0, mask=left_mask),
                candidate(1, 1, 'right', node_index=7, mask=right_mask),
            ],
        ]
        coordinator = self.make_coordinator(use_frontier=False, tabu_waypoint_steps=2)
        coordinator.recent_waypoints[1] = [3, 4]
        selected, _ = coordinator.select(lists, step=0)
        self.assertEqual(selected[1]['action_index'], 1)

    def test_detour_budget_limits_cumulative_extra_distance(self):
        left_mask = np.array([[1, 1, 0, 0]], dtype=bool)
        right_mask = np.array([[0, 0, 1, 1]], dtype=bool)
        lists = [
            [candidate(0, 0, 'left', mask=left_mask)],
            [
                candidate(1, 0, 'left', node_index=0, position=(1.0, 0.0), mask=left_mask),
                candidate(1, 1, 'right', node_index=1, position=(2.0, 0.0), mask=right_mask),
            ],
        ]
        coordinator = self.make_coordinator(use_frontier=False, detour_budget=0.5)
        coordinator.detour_used[1] = 0.4
        selected, info = coordinator.select(lists, step=0)
        self.assertEqual(selected[1]['action_index'], 0)
        self.assertEqual(info['budget_exhausted'], 0)

    def test_detour_budget_allows_affordable_detour(self):
        left_mask = np.array([[1, 1, 0, 0]], dtype=bool)
        right_mask = np.array([[0, 0, 1, 1]], dtype=bool)
        lists = [
            [candidate(0, 0, 'left', mask=left_mask)],
            [
                candidate(1, 0, 'left', node_index=0, position=(1.0, 0.0), mask=left_mask),
                candidate(1, 1, 'right', node_index=1, position=(1.2, 0.0), mask=right_mask),
            ],
        ]
        coordinator = self.make_coordinator(use_frontier=False, detour_budget=0.5)
        selected, _ = coordinator.select(lists, step=0)
        self.assertEqual(selected[1]['action_index'], 1)
        self.assertAlmostEqual(coordinator.detour_used[1], 0.2, places=6)

    def test_frontier_floor_blocks_low_utility_candidate(self):
        left_mask = np.array([[1, 1, 0, 0]], dtype=bool)
        right_mask = np.array([[0, 0, 1, 1]], dtype=bool)
        lists = [
            [candidate(0, 0, 'left', frontiers=['a', 'b', 'c', 'd'], mask=left_mask)],
            [
                candidate(1, 0, 'left', frontiers=['a', 'b', 'c', 'd'], mask=left_mask),
                candidate(1, 1, 'right', frontiers=['x'], mask=right_mask),
            ],
        ]
        selected, _ = self.make_coordinator(
            use_frontier=False, min_frontier_ratio=0.5
        ).select(lists, step=0)
        self.assertEqual(selected[1]['action_index'], 0)

    def test_frontier_floor_allows_rich_candidate(self):
        left_mask = np.array([[1, 1, 0, 0]], dtype=bool)
        right_mask = np.array([[0, 0, 1, 1]], dtype=bool)
        lists = [
            [candidate(0, 0, 'left', frontiers=['a', 'b', 'c', 'd'], mask=left_mask)],
            [
                candidate(1, 0, 'left', frontiers=['a', 'b', 'c', 'd'], mask=left_mask),
                candidate(1, 1, 'right', frontiers=['x', 'y', 'z'], mask=right_mask),
            ],
        ]
        selected, _ = self.make_coordinator(
            use_frontier=False, min_frontier_ratio=0.5
        ).select(lists, step=0)
        self.assertEqual(selected[1]['action_index'], 1)

    def test_separation_prefers_far_candidate(self):
        lists = [
            [candidate(0, 0, 'left', position=[0.0, 0.0])],
            [candidate(1, 0, 'right', position=[1.0, 0.0]),
             candidate(1, 1, 'right', position=[9.0, 0.0])],
        ]
        selected, info = self.make_coordinator(
            use_overlap=False, use_frontier=False,
            separation_weight=1.0, separation_radius=12.0,
        ).select(lists, step=0)
        self.assertEqual(selected[1]['action_index'], 1)
        self.assertEqual(info['crowded'], 1)

    def test_separation_inactive_when_fleet_is_sparse(self):
        lists = [
            [candidate(0, 0, 'left', position=[0.0, 0.0])],
            [candidate(1, 0, 'right', position=[1.0, 0.0]),
             candidate(1, 1, 'right', position=[9.0, 0.0])],
        ]
        coordinator = self.make_coordinator(
            use_overlap=False, use_frontier=False,
            separation_weight=1.0, separation_radius=12.0,
        )
        for robot in coordinator.robots:
            robot.location = np.array([40.0 * (robot.id + 1), 0.0])
        selected, info = coordinator.select(lists, step=0)
        self.assertEqual(selected[1]['action_index'], 0)
        self.assertEqual(info['crowded'], 0)

    def test_separation_disabled_keeps_policy_choice(self):
        lists = [
            [candidate(0, 0, 'left', position=[0.0, 0.0])],
            [candidate(1, 0, 'right', position=[1.0, 0.0]),
             candidate(1, 1, 'right', position=[9.0, 0.0])],
        ]
        selected, info = self.make_coordinator(
            use_overlap=False, use_frontier=False,
        ).select(lists, step=0)
        self.assertEqual(selected[1]['action_index'], 0)
        self.assertEqual(info['crowded'], 0)

    def test_separation_horizon_turns_off_late(self):
        lists = [
            [candidate(0, 0, 'left', position=[0.0, 0.0])],
            [candidate(1, 0, 'right', position=[1.0, 0.0]),
             candidate(1, 1, 'right', position=[9.0, 0.0])],
        ]
        coordinator = self.make_coordinator(
            use_overlap=False, use_frontier=False,
            separation_weight=1.0, separation_radius=12.0,
            separation_horizon=5,
        )
        selected, info = coordinator.select(lists, step=4)
        self.assertEqual(selected[1]['action_index'], 1)
        self.assertEqual(info['separation_active'], 1)

        coordinator = self.make_coordinator(
            use_overlap=False, use_frontier=False,
            separation_weight=1.0, separation_radius=12.0,
            separation_horizon=5,
        )
        selected, info = coordinator.select(lists, step=5)
        self.assertEqual(selected[1]['action_index'], 0)
        self.assertEqual(info['separation_active'], 0)
        self.assertEqual(info['crowded'], 0)

    def test_priority_rotates(self):
        lists = [[candidate(0, 0, 'left')], [candidate(1, 0, 'left')]]
        _, info = self.make_coordinator(use_overlap=False, use_frontier=False).select(lists, step=1)
        self.assertEqual(info['priority_start'], 1)


if __name__ == '__main__':
    unittest.main()
