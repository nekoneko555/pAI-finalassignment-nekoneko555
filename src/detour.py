"""Blocked direct routes, independently generated train/validation/test splits.

Certified waypoints are used only to check feasibility and choose curriculum
reset states. They are never observations or inference-time control commands.
"""

import argparse
import json
from pathlib import Path
import numpy as np

from .config import ROOT
from .env import InstructionArmEnv
from .geometry import points, clearance, path_clearance


def sample_detour(rng):
    start = np.array([-0.8, 1.6])
    for _ in range(2000):
        target = rng.uniform([0.10, 1.15], [0.65, 1.75])
        middle = (1 - (t := rng.uniform(0.35, 0.65))) * start + t * target
        _, elbow, tip = points(middle)
        center = (
            elbow
            + rng.uniform(0.88, 1.0) * (tip - elbow)
            + rng.uniform(-0.012, 0.012, 2)
        )
        obstacle = np.r_[center, rng.uniform(0.035, 0.05)]
        if np.linalg.norm(center) < 0.42:
            continue
        if min(clearance(start, obstacle), clearance(target, obstacle)) < 0.03:
            continue
        route = np.array([start, [start[0], 2.55], [target[0], 2.55], target])
        direct = path_clearance([start, target], obstacle)
        detour = path_clearance(route, obstacle)
        if direct < -0.012 and detour > 0.018:
            return {
                "initial_q": start.tolist(),
                "goal": points(target)[2].tolist(),
                "obstacle": obstacle.tolist(),
                "certified_waypoints": route.tolist(),
                "certificate_min_clearance_m": detour,
                "direct_route_min_clearance_m": direct,
            }
    raise RuntimeError("Could not generate a blocked but solvable route")


def generate_detour_cases(count, seed):
    cases = []
    for i in range(count):
        case = sample_detour(np.random.default_rng(seed + i))
        case.update(id=f"detour-{seed}-{i:04d}", seed=seed + i)
        cases.append(case)
    return cases


def waypoint_action(env, route, index):
    """Feasibility oracle baseline, separate from the learned policy."""
    while (
        index < len(route) - 1 and np.max(np.abs(env.data.qpos - route[index])) < 0.055
    ):
        index += 1
    speed = env.filter.hard_speed * env.filter.limits.speed_scale
    action = np.clip(
        3 * (np.asarray(route[index]) - env.data.qpos) / speed, -1, 1
    ).astype(np.float32)
    return action, index


class DetourArmEnv(InstructionArmEnv):
    def __init__(self, training=False, **kwargs):
        self.training = training
        self.curriculum_progress = 1.0
        self.current_route = None
        self._pool = None
        if training:
            self._pool = json.loads(
                (ROOT / "data/detour_train.json").read_text(encoding="utf-8")
            )
        super().__init__(**kwargs)

    def set_curriculum(self, progress):
        self.curriculum_progress = float(np.clip(progress, 0, 1))

    def _sample(self):
        case = (
            self._pool[int(self.np_random.integers(len(self._pool)))]
            if self._pool
            else sample_detour(self.np_random)
        )
        route = np.asarray(case["certified_waypoints"])
        start = route[0].copy()
        if self.training and self.curriculum_progress < 0.85:
            # Expand backward from the final part of a known collision-free route.
            lower = 0.85 * (1 - self.curriculum_progress / 0.85)
            fraction = self.np_random.uniform(lower, 1)
            segment = min(2, int(fraction * 3))
            start = (1 - (t := fraction * 3 - segment)) * route[segment] + t * route[
                segment + 1
            ]
        self.current_route = route
        return start, np.asarray(case["goal"]), np.asarray(case["obstacle"])

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # Distance cost removes the incentive to stop just outside the 5cm goal.
        reward -= 0.3 * info["distance"]
        return obs, float(reward), terminated, truncated, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=910000)
    parser.add_argument("--output", type=Path, default=ROOT / "data/detour_test.json")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(generate_detour_cases(args.count, args.seed), indent=2),
        encoding="utf-8",
    )
    print(f"Saved {args.count} detour cases: {args.output.name}")


if __name__ == "__main__":
    main()
