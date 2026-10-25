"""Gymnasium API, velocity actuators and full-link collision detection."""

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from .config import ROOT, load_config
from .controller import SafetyFilter
from .geometry import clearance, inverse_kinematics, points


class InstructionArmEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self,
        render_mode=None,
        mode="standard",
        task="fixed",
        max_steps=None,
        renderer="mujoco",
        width=960,
        height=640,
    ):
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError("Only rgb_array rendering is supported")
        if task not in ("fixed", "random", "obstacle"):
            raise ValueError("Unknown task")
        if renderer not in ("mujoco", "2d"):
            raise ValueError("Unknown renderer")
        self.cfg = load_config()["environment"]
        # Windows MuJoCo file APIs can mishandle non-ASCII paths. Python handles
        # Unicode paths correctly; this XML has no external mesh dependencies.
        self.model = mujoco.MjModel.from_xml_string(
            (ROOT / "assets/arm.xml").read_text(encoding="utf-8")
        )
        self.model.opt.timestep = self.cfg["timestep"]
        self.data = mujoco.MjData(self.model)
        self.repeat = self.cfg["action_repeat"]
        self.dt = self.repeat * self.model.opt.timestep
        self.max_steps = max_steps if max_steps is not None else self.cfg["max_steps"]
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.filter = SafetyFilter(
            mode,
            self.cfg["hard_speed"],
            self.cfg["hard_acceleration"],
            self.cfg["joint_limit"],
        )
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        # sin/cos(4), qvel(2), goal-tip(2), obstacle-tip(2), radius, flag,
        # previous filtered command(2). The last two make the slew filter observable.
        low = np.array(
            [-1] * 4 + [-2] * 2 + [-1.4] * 4 + [0, 0] + [-1.5] * 2, np.float32
        )
        high = np.array(
            [1] * 4 + [2] * 2 + [1.4] * 4 + [0.1, 1] + [1.5] * 2, np.float32
        )
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.render_mode, self.renderer_kind = render_mode, renderer
        self.task, self.width, self.height = task, width, height
        self._renderer = None
        self._tip_id = self.model.site("tip").id
        self._obstacle_id = self.model.geom("obstacle_geom").id
        self._link_ids = {self.model.geom(n).id for n in ("link1", "link2")}
        self._target_mocap = self.model.body("target").mocapid[0]
        self._obstacle_mocap = self.model.body("obstacle").mocapid[0]
        self.goal = np.array(self.cfg["fixed_goal"], dtype=float)
        self.obstacle = None
        self.steps, self.collision, self.done = 0, False, True

    @property
    def tip(self):
        return self.data.site_xpos[self._tip_id, :2].copy()

    def _sample(self):
        q = np.array(self.cfg["initial_q"], dtype=float)
        if self.task == "fixed":
            return q, np.array(self.cfg["fixed_goal"], dtype=float), None
        # Restricted positive-elbow configurations avoid singular and joint-limit targets.
        for _ in range(200):
            target_q = self.np_random.uniform([-1.1, 0.7], [0.45, 2.1])
            goal = points(target_q)[2]
            if np.linalg.norm(goal - points(q)[2]) < 0.12:
                continue
            obstacle = None
            if self.task == "obstacle":
                obstacle = np.r_[
                    self.np_random.uniform([-0.12, -0.3], [0.48, 0.45]), 0.055
                ]
                # Reject obstacles intersecting a known joint-space route (conservative).
                if (
                    min(clearance(v, obstacle) for v in np.linspace(q, target_q, 40))
                    < 0.025
                ):
                    continue
            return q, goal, obstacle
        raise RuntimeError("Could not sample a feasible scenario")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "case" in options:
            case = options["case"]
            q = np.asarray(case["initial_q"], dtype=float)
            goal = np.asarray(case["goal"], dtype=float)
            obstacle = case.get("obstacle")
            obstacle = None if obstacle is None else np.asarray(obstacle, dtype=float)
        else:
            q, goal, obstacle = self._sample()
        if q.shape != (2,) or not np.isfinite(q).all() or np.any(np.abs(q) > 2.7):
            raise ValueError("Initial joints outside allowed range")
        if goal.shape != (2,) or not np.isfinite(goal).all():
            raise ValueError("Invalid goal")
        target_q = inverse_kinematics(goal)
        if np.any(np.abs(target_q) > 2.7):
            raise ValueError("Target violates joint limits")
        if obstacle is not None:
            if (
                obstacle.shape != (3,)
                or not np.isfinite(obstacle).all()
                or not 0.02 <= obstacle[2] <= 0.1
                or np.any(np.abs(obstacle[:2]) > 0.7)
            ):
                raise ValueError("Invalid obstacle")
            if (
                clearance(q, obstacle) <= 0.005
                or clearance(target_q, obstacle) <= 0.005
            ):
                raise ValueError("Obstacle overlaps start or target configuration")
        mujoco.mj_resetData(self.model, self.data)
        self.goal, self.obstacle = goal.copy(), obstacle
        self.data.qpos[:] = q
        self.data.mocap_pos[self._target_mocap] = [*goal, 0.10]
        self.data.mocap_pos[self._obstacle_mocap] = (
            [2, 2, 0.10] if obstacle is None else [*obstacle[:2], 0.10]
        )
        self.model.geom_size[self._obstacle_id, 0] = (
            0.055 if obstacle is None else obstacle[2]
        )
        # Recompute broad-phase bounds as well as the narrow-phase radius.
        radius, half_height = self.model.geom_size[self._obstacle_id, :2]
        self.model.geom_rbound[self._obstacle_id] = np.hypot(radius, half_height)
        self.model.geom_aabb[self._obstacle_id, 3:] = [radius, radius, half_height]
        self.model.geom_rgba[self._obstacle_id, 3] = 0 if obstacle is None else 1
        mujoco.mj_forward(self.model, self.data)
        self.steps, self.collision, self.done = 0, False, False
        self.filter.reset()
        self.distance = float(np.linalg.norm(self.goal - self.tip))
        self.min_distance = self.distance
        self._last_obs = self._obs()
        return self._last_obs.copy(), self._info()

    def _obs(self):
        q = self.data.qpos
        rel = np.zeros(2) if self.obstacle is None else self.obstacle[:2] - self.tip
        return np.concatenate(
            [
                np.sin(q),
                np.cos(q),
                self.data.qvel,
                self.goal - self.tip,
                rel,
                [
                    0 if self.obstacle is None else self.obstacle[2],
                    self.obstacle is not None,
                ],
                self.filter.previous,
            ]
        ).astype(np.float32)

    def _touching(self):
        for c in self.data.contact[: self.data.ncon]:
            pair = {c.geom1, c.geom2}
            if self._obstacle_id in pair and pair.intersection(self._link_ids):
                return True
        return False

    def _info(self, success=False, invalid=False):
        return {
            "is_success": bool(success),
            "collision": bool(self.collision),
            "invalid_state": bool(invalid),
            "distance": self.distance,
            "min_goal_distance_m": self.min_distance,
            "elapsed_time_s": self.steps * self.dt,
            "mode": self.filter.mode,
            "command": self.filter.previous.astype(np.float32).copy(),
        }

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset before stepping a finished episode")
        if (
            not np.isfinite(self.data.qpos).all()
            or not np.isfinite(self.data.qvel).all()
        ):
            self.done = True
            self.filter.reset()
            return self._last_obs.copy(), -5.0, True, False, self._info(invalid=True)
        command = self.filter.apply(action, self.data.qpos, self.dt)
        previous_tip, previous_distance = self.tip, self.distance
        self.data.ctrl[:] = command
        invalid = False
        for _ in range(self.repeat):
            # Check at every physics substep, rather than only at the RL boundary.
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self.collision |= self._touching()
            invalid = (
                not np.isfinite(self.data.qpos).all()
                or not np.isfinite(self.data.qvel).all()
                or np.any(np.abs(self.data.qpos) > self.cfg["joint_limit"])
                or np.any(np.abs(self.data.qvel) > self.cfg["hard_speed"] + 0.01)
            )
            if invalid or self.collision:
                break
        self.steps += 1
        if invalid:
            # Return a finite absorbing observation; never feed NaN into PPO.
            self.data.qpos[:] = np.nan_to_num(self.data.qpos, nan=0, posinf=0, neginf=0)
            self.data.qpos[:] = np.clip(self.data.qpos, -2.8, 2.8)
            self.data.qvel[:] = 0
            mujoco.mj_forward(self.model, self.data)
        self.distance = float(np.linalg.norm(self.goal - self.tip))
        self.min_distance = min(self.min_distance, self.distance)
        success = (
            self.distance < self.cfg["goal_tolerance"]
            and not self.collision
            and not invalid
        )
        self.done = bool(
            success or self.collision or invalid or self.steps >= self.max_steps
        )
        reward = (
            2 * (previous_distance - self.distance)
            - 0.01
            + 10 * success
            - 10 * self.collision
            - 5 * invalid
        )
        terminated = bool(success or self.collision or invalid)
        truncated = bool(self.steps >= self.max_steps and not terminated)
        info = self._info(success, invalid)
        info["tip_speed_m_s"] = float(np.linalg.norm(self.tip - previous_tip) / self.dt)
        self._last_obs = self._obs()
        return self._last_obs.copy(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            raise RuntimeError("Construct with render_mode='rgb_array'")
        if self.renderer_kind == "2d":
            from .rendering import render_2d

            return render_2d(self)
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.height, width=self.width
            )
        self._renderer.update_scene(self.data, camera="overview")
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
