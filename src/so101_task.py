"""Gymnasium task: physical pick/place with PPO residual joint-speed control.

IK supplies joint waypoints and a state machine supplies gripper open/close.
PPO learns five bounded speed corrections. This is not end-to-end learned grasping.
"""

from functools import lru_cache
import json
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .config import ROOT
from .so101_pickplace import PickPlaceSimulation

PHASES = (
    "APPROACH A",
    "DESCEND",
    "GRASP",
    "LIFT",
    "TRANSPORT TO B",
    "LOWER AT B",
    "RELEASE",
    "RETREAT",
)
DWELL = (0.25, 0.35, 1.0, 0.3, 0.3, 0.3, 0.8, 1.0)


@lru_cache(maxsize=4)
def cases_for(split):
    return json.loads((ROOT / f"data/so101_{split}.json").read_text(encoding="utf-8"))


def generate_case(seed):
    rng = np.random.default_rng(seed)
    return {
        "id": f"so101-pickplace-{seed}",
        "seed": int(seed),
        "a": [float(rng.uniform(0.222, 0.234)), float(rng.uniform(-0.100, -0.078))],
        "b": [float(rng.uniform(0.222, 0.234)), float(rng.uniform(0.078, 0.100))],
    }


class SO101PickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self, mode="standard", render_mode=None, training=False, max_steps=900
    ):
        super().__init__()
        if render_mode not in (None, "rgb_array") or max_steps < 1:
            raise ValueError("Invalid task options")
        self.sim = PickPlaceSimulation(mode)
        self.render_mode, self.training, self.max_steps = (
            render_mode,
            training,
            max_steps,
        )
        self.action_space = spaces.Box(-1.0, 1.0, (5,), dtype=np.float32)
        # joint target error, servo lag, measured velocity, command velocity,
        # sin/cos joints, stage, object-to-tip, mode's speed and acceleration.
        self.observation_space = spaces.Box(-np.inf, np.inf, (43,), dtype=np.float32)
        self.done = True
        self._target_cache = {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        case = options.get("case") if options else None
        if case is None:
            if self.training:
                cases = cases_for("train")
                case = cases[int(self.np_random.integers(len(cases)))]
            else:
                case = generate_case(int(self.np_random.integers(0, 2**31)))
        a, b = np.asarray(case["a"], float), np.asarray(case["b"], float)
        if a.shape != (2,) or b.shape != (2,) or not np.isfinite(np.r_[a, b]).all():
            raise ValueError("Expected finite A/B positions")
        if not (
            0.20 < a[0] < 0.25
            and -0.12 < a[1] < -0.06
            and 0.20 < b[0] < 0.25
            and 0.06 < b[1] < 0.12
        ):
            raise ValueError("A/B positions outside the validated task workspace")
        self.case = case
        self.sim.reset(a, b)
        key = tuple(np.r_[a, b])
        if key not in self._target_cache:
            poses = [
                self.sim.grasp_site(a, 0.07),
                self.sim.grasp_site(a, 0.025),
                self.sim.grasp_site(a, 0.025),
                self.sim.grasp_site(a, 0.075),
                self.sim.grasp_site(b, 0.075),
                self.sim.grasp_site(b, 0.026),
                self.sim.grasp_site(b, 0.026),
                self.sim.grasp_site(b, 0.075),
            ]
            q = self.sim.data.qpos[self.sim.qids].copy()
            targets = []
            for pose in poses:
                q = self.sim.inverse_pose(pose, start=q)
                targets.append(q)
            self._target_cache[key] = targets
        self.targets = self._target_cache[key]
        self.phase, self.stable, self.phase_steps, self.steps = 0, 0, 0, 0
        self.done, self.invalid, self.phase_timeout = False, False, False
        self.dropped = False
        self.release_started = False
        self.energy = 0.0
        self.last_obs = self._obs()
        return self.last_obs.copy(), self.info()

    def _obs(self):
        s = self.sim
        q = s.data.qpos[s.qids]
        phase = np.zeros(8)
        phase[min(self.phase, 7)] = 1
        return np.r_[
            self.targets[min(self.phase, 7)] - q,
            s.command - q,
            s.data.qvel[s.vids] / 3,
            s.velocity / 1.2,
            np.sin(q),
            np.cos(q),
            phase,
            s.object_position - s.tip,
            s.speed,
            s.acceleration / 3,
        ].astype(np.float32)

    def info(self):
        s = self.sim
        obj = s.object_position
        fixed, moving = s.jaw_contacts()
        released = bool(not fixed and not moving and s.data.qpos[s.grip_qid] > 0.5)
        final_error = float(np.linalg.norm(obj[:2] - s.b))
        settled = bool(
            np.linalg.norm(s.data.qvel[s.model.joint("parcel_free").dofadr[0] :][:3])
            < 0.03
        )
        success = bool(
            self.phase == 8
            and s.lifted
            and s.both_contact_while_lifted
            and released
            and final_error < 0.035
            and 0.01 < obj[2] < 0.035
            and settled
            and not self.invalid
        )
        return {
            "is_success": success,
            "lifted": bool(s.lifted),
            "physical_grasp": bool(s.both_contact_while_lifted),
            "released": released,
            "phase": PHASES[min(self.phase, 7)],
            "completed_phases": int(self.phase),
            "phase_timeout": bool(self.phase_timeout),
            "invalid_state": bool(self.invalid),
            "dropped": bool(self.dropped),
            "elapsed_time_s": round(s.elapsed, 6),
            "final_xy_error_m": final_error,
            "object_height_m": float(obj[2]),
            "max_object_height_m": s.max_height,
            "object_settled": settled,
            "tip_speed_m_s": s.trace[-1]["tip_speed_m_s"] if s.trace else 0.0,
            "mode": s.mode,
            "object_is_free_body": True,
            "weld_constraints": int(s.model.neq),
        }

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset before stepping")
        action = np.asarray(action, dtype=float)
        if action.shape != (5,) or not np.isfinite(action).all():
            raise ValueError("Expected five finite PPO residuals")
        action = np.clip(action, -1, 1)
        phase = self.phase
        target = self.targets[phase]
        previous = np.linalg.norm(target - self.sim.data.qpos[self.sim.qids])
        grip = -0.15 if phase in (2, 3, 4, 5) else 0.65
        self.sim.step(target, grip, PHASES[phase], residual=action)
        self.steps += 1
        self.phase_steps += 1
        self.invalid = not bool(
            np.isfinite(self.sim.data.qpos).all()
            and np.isfinite(self.sim.data.qvel).all()
        )
        if self.invalid:
            self.done = True
            # A finite absorbing observation rather than propagating non-finite state.
            return (
                self.last_obs.copy(),
                -20.0,
                True,
                False,
                {**self.info(), "is_success": False},
            )
        error = np.linalg.norm(target - self.sim.data.qpos[self.sim.qids])
        arrived = (
            np.max(np.abs(self.sim.command - target)) < 0.002
            and np.max(np.abs(self.sim.data.qpos[self.sim.qids] - target)) < 0.035
        )
        self.stable = self.stable + 1 if arrived else 0
        completed = self.stable * self.sim.dt >= DWELL[phase]
        if completed:
            self.phase += 1
            self.stable, self.phase_steps = 0, 0
        self.release_started |= self.phase >= 6
        if (
            self.sim.lifted
            and self.phase in (3, 4)
            and self.sim.object_position[2] < 0.04
        ):
            self.dropped = True
        self.phase_timeout = self.phase_steps >= 350
        terminated = bool(self.phase == 8 or self.phase_timeout or self.dropped)
        truncated = bool(self.steps >= self.max_steps and not terminated)
        self.done = terminated or truncated
        info = self.info()
        reward = (
            1.5 * (previous - error)
            - 0.04
            - 0.0002 * float(action @ action)
            + 0.5 * completed
        )
        if self.done:
            reward += 12 * info["is_success"] - 8 * (not info["is_success"])
        self.last_obs = self._obs()
        return self.last_obs.copy(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            raise RuntimeError("render_mode='rgb_array' is required")
        return self.sim.render()

    def close(self):
        self.sim.close()
