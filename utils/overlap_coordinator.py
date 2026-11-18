"""Inference-only coordination for redundant frontier and FoV coverage."""

import numpy as np


class OverlapCoordinator:
    """Greedily commit Top-K policy candidates with soft redundancy penalties."""

    def __init__(
        self,
        robots,
        sensing_mask_fn,
        fov,
        sensor_range,
        top_k=12,
        overlap_weight=2.0,
        frontier_weight=1.0,
        frontier_softness=1.0,
        switch_margin=0.05,
        hysteresis_bonus=0.05,
        overlap_trigger=0.10,
        max_policy_drop=np.inf,
        max_detour_ratio=np.inf,
        same_waypoint_only=False,
        tabu_waypoint_steps=0,
        detour_budget=np.inf,
        min_frontier_ratio=0.0,
        separation_weight=0.0,
        separation_radius=0.0,
        separation_reduce="mean",
        separation_horizon=float("inf"),
        use_overlap=True,
        use_frontier=True,
    ):
        self.robots = robots
        self.sensing_mask_fn = sensing_mask_fn
        self.fov = fov
        self.sensor_range = sensor_range
        self.top_k = top_k
        self.overlap_weight = overlap_weight
        self.frontier_weight = frontier_weight
        self.frontier_softness = frontier_softness
        self.switch_margin = switch_margin
        self.hysteresis_bonus = hysteresis_bonus
        self.overlap_trigger = overlap_trigger
        self.max_policy_drop = max_policy_drop
        self.max_detour_ratio = max_detour_ratio
        self.same_waypoint_only = same_waypoint_only
        self.use_overlap = use_overlap
        self.use_frontier = use_frontier
        self.tabu_waypoint_steps = tabu_waypoint_steps
        self.detour_budget = detour_budget
        self.min_frontier_ratio = min_frontier_ratio
        self.detour_used = [0.0] * len(robots)
        self.previous_actions = [None] * len(robots)
        self.recent_waypoints = [[] for _ in robots]
        self.separation_weight = separation_weight
        # 0 表示自动取 1.2 倍传感半径：该尺度内任意朝向的 120° 扇形必然互相覆盖
        self.separation_radius = separation_radius if separation_radius > 0 else 1.2 * sensor_range
        # 'mean' 用平均间距，'min' 只看最近队友——重叠由最近的那台决定，min 更贴合几何
        self.separation_reduce = separation_reduce
        # 只在开局 crowding 阶段启用分离项（重叠质量集中在最初若干决策步），之后回归纯策略
        self.separation_horizon = separation_horizon

    def _candidate_frontiers(self, robot, candidate):
        if "frontiers" in candidate:
            return candidate["frontiers"]

        coords = candidate["position"]
        node = robot.node_manager.nodes_dict.find((coords[0], coords[1])).data
        points = np.array(list(node.observable_frontiers), dtype=float).reshape(-1, 2)
        if points.shape[0] == 0:
            frontiers = frozenset()
        else:
            relative = points - coords.reshape(1, 2)
            distances = np.linalg.norm(relative, axis=1)
            angles = np.degrees(np.arctan2(relative[:, 1], relative[:, 0])) % 360
            heading = candidate["execution_heading"]
            angle_delta = np.abs((angles - heading + 180) % 360 - 180)
            visible = (distances <= self.sensor_range + 1e-6) & (angle_delta <= self.fov / 2 + 1e-6)
            frontiers = frozenset(map(tuple, points[visible]))
        candidate["frontiers"] = frontiers
        candidate["raw_frontier_count"] = len(frontiers)
        return frontiers

    def _candidate_mask(self, candidate):
        if "mask" not in candidate:
            candidate["mask"] = self.sensing_mask_fn(
                candidate["position"], candidate["execution_heading"]
            ).astype(bool, copy=False)
        return candidate["mask"]

    def _frontier_ratio(self, candidate, committed_counts):
        frontiers = self._candidate_frontiers(self.robots[candidate["robot_id"]], candidate)
        if not frontiers:
            return 1.0
        residual = sum(
            1.0 / (1.0 + self.frontier_softness * committed_counts.get(frontier, 0))
            for frontier in frontiers
        )
        return residual / len(frontiers)

    def _overlap_ratio(self, candidate, committed_mask):
        if committed_mask is None:
            return 0.0
        mask = self._candidate_mask(candidate)
        area = np.count_nonzero(mask)
        if area == 0:
            return 0.0
        return np.count_nonzero(mask & committed_mask) / area

    def _travel_distance(self, robot, candidate):
        return float(np.linalg.norm(np.asarray(candidate["position"], dtype=float) - robot.location))

    def _crowded(self):
        """机群是否处于“几何上必然重叠”的拥挤态。

        只要存在一对智能体间距小于 separation_radius，任何候选朝向的 120° 扇形都会与
        队友大面积相交，此时重叠惩罚对所有候选近似等值（梯度饱和），必须改用位移分离项。
        """
        if self.separation_weight <= 0.0:
            return False
        locations = np.array([robot.location for robot in self.robots], dtype=float).reshape(-1, 2)
        for i in range(locations.shape[0]):
            for j in range(i + 1, locations.shape[0]):
                if np.linalg.norm(locations[i] - locations[j]) < self.separation_radius:
                    return True
        return False

    def _separation_gain(self, candidate, committed_positions):
        """候选落点相对已承诺队友的分离度，取值 [0, 1]，越大越分散。"""
        if not committed_positions:
            return 0.0
        position = np.asarray(candidate["position"], dtype=float)
        gains = [
            min(float(np.linalg.norm(position - other)) / self.separation_radius, 1.0)
            for other in committed_positions
        ]
        if self.separation_reduce == "min":
            return float(np.min(gains))
        return float(np.mean(gains))

    def _eligible(self, robot, candidates, original, original_distance):
        eligible = []
        remaining_budget = self.detour_budget - self.detour_used[robot.id]
        recent = set(self.recent_waypoints[robot.id][-self.tabu_waypoint_steps:]) if self.tabu_waypoint_steps > 0 else set()
        frontier_floor = 0.0
        if self.min_frontier_ratio > 0.0:
            frontier_floor = self.min_frontier_ratio * len(self._candidate_frontiers(robot, original))
        for candidate in candidates:
            if candidate["logp"] < original["logp"] - self.max_policy_drop:
                continue
            if self.same_waypoint_only and candidate["node_index"] != original["node_index"]:
                continue
            if recent and candidate["node_index"] in recent:
                continue
            if frontier_floor > 0.0:
                if len(self._candidate_frontiers(robot, candidate)) < frontier_floor:
                    continue
            if (self.max_detour_ratio < np.inf) or (remaining_budget < np.inf):
                distance = self._travel_distance(robot, candidate)
                if distance > self.max_detour_ratio * original_distance + 1e-6:
                    continue
                if distance - original_distance > remaining_budget + 1e-6:
                    continue
            eligible.append(candidate)
        return eligible

    def _same_action(self, lhs, rhs):
        if lhs is None or rhs is None:
            return False
        return (
            lhs["node_index"] == rhs["node_index"]
            and lhs["heading_index"] == rhs["heading_index"]
        )

    def select(self, candidate_lists, step):
        """Select one candidate per robot using a rotating-priority commit order."""
        n_agents = len(self.robots)
        start = step % n_agents
        order = list(range(start, n_agents)) + list(range(start))
        selected = [None] * n_agents
        committed_mask = None
        committed_counts = {}
        committed_positions = []
        separation_active = (self.separation_weight > 0.0) and (step < self.separation_horizon)
        crowded = self._crowded() if separation_active else False
        has_committed = False
        changed = 0
        triggered = 0
        predicted_overlaps = []

        for robot_id in order:
            candidates = candidate_lists[robot_id][: self.top_k]
            if not candidates:
                raise ValueError("Robot {} has no action candidates".format(robot_id))

            original = candidates[0]
            original_overlap = (
                self._overlap_ratio(original, committed_mask) if self.use_overlap else 0.0
            )
            original_frontier_ratio = (
                self._frontier_ratio(original, committed_counts) if self.use_frontier else 1.0
            )
            should_coordinate = has_committed and self.detour_used[robot_id] < self.detour_budget and (
                crowded
                or (self.use_overlap and original_overlap > self.overlap_trigger)
                or (self.use_frontier and original_frontier_ratio < 1.0)
            )

            if not should_coordinate:
                chosen = original
                chosen_overlap = original_overlap
            else:
                triggered += 1
                original_distance = max(self._travel_distance(self.robots[robot_id], original), 1e-6)
                eligible_candidates = self._eligible(
                    self.robots[robot_id], candidates, original, original_distance
                )
                if not eligible_candidates:
                    eligible_candidates = [original]
                best_score = -np.inf
                chosen = original
                chosen_overlap = original_overlap
                for candidate in eligible_candidates:
                    rank = candidate["rank"]
                    overlap = (
                        self._overlap_ratio(candidate, committed_mask)
                        if self.use_overlap
                        else 0.0
                    )
                    frontier_ratio = (
                        self._frontier_ratio(candidate, committed_counts)
                        if self.use_frontier
                        else 1.0
                    )
                    separation_gain = (
                        self._separation_gain(candidate, committed_positions) if crowded else 0.0
                    )
                    score = (
                        candidate["logp"]
                        + self.frontier_weight * frontier_ratio
                        - self.overlap_weight * overlap
                        + self.separation_weight * separation_gain
                    )
                    if rank == 0:
                        score += self.switch_margin
                    if self._same_action(candidate, self.previous_actions[robot_id]):
                        score += self.hysteresis_bonus
                    if score > best_score:
                        best_score = score
                        chosen = candidate
                        chosen_overlap = overlap

            if chosen["action_index"] != original["action_index"]:
                changed += 1
                extra = self._travel_distance(self.robots[robot_id], chosen) - self._travel_distance(
                    self.robots[robot_id], original
                )
                if extra > 0:
                    self.detour_used[robot_id] += extra
            selected[robot_id] = chosen
            self.previous_actions[robot_id] = chosen
            self.recent_waypoints[robot_id].append(chosen["node_index"])
            committed_positions.append(np.asarray(chosen["position"], dtype=float))
            if len(self.recent_waypoints[robot_id]) > 8:
                del self.recent_waypoints[robot_id][:-8]
            predicted_overlaps.append(chosen_overlap)

            if self.use_overlap:
                chosen_mask = self._candidate_mask(chosen)
                committed_mask = (
                    chosen_mask.copy()
                    if committed_mask is None
                    else committed_mask | chosen_mask
                )
            if self.use_frontier:
                for frontier in self._candidate_frontiers(self.robots[robot_id], chosen):
                    committed_counts[frontier] = committed_counts.get(frontier, 0) + 1
            has_committed = True

        info = {
            "changed_actions": changed,
            "triggered_agents": triggered,
            "budget_exhausted": int(sum(1 for used in self.detour_used if used >= self.detour_budget)),
            "predicted_overlap": float(np.mean(predicted_overlaps)) if predicted_overlaps else 0.0,
            "priority_start": start,
            "crowded": int(crowded),
            "separation_active": int(separation_active),
        }
        return selected, info
