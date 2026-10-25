"""The non-learning safety filter limits commands, not collision risk."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Limits:
    speed_scale: float
    acceleration: float


MODES = {
    "careful": Limits(0.35, 1.0),
    "standard": Limits(0.80, 4.0),
    "fast": Limits(0.80, 4.0),
}


class SafetyFilter:
    def __init__(
        self, mode="standard", hard_speed=1.5, hard_acceleration=4.0, joint_limit=2.8
    ):
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.limits = MODES[mode]
        self.hard_speed = float(hard_speed)
        self.hard_acceleration = float(hard_acceleration)
        self.joint_limit = float(joint_limit)
        self.previous = np.zeros(2)

    def reset(self):
        self.previous[:] = 0

    def apply(self, action, q, dt):
        action = np.asarray(action, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Action must contain two finite values")
        if q.shape != (2,) or not np.isfinite(q).all() or dt <= 0:
            raise ValueError("Invalid joint state or timestep")
        speed = min(self.hard_speed, self.hard_speed * self.limits.speed_scale)
        delta = min(self.hard_acceleration, self.limits.acceleration) * dt
        command = np.clip(action, -1, 1) * speed
        command = np.clip(command, self.previous - delta, self.previous + delta)
        # Reserve braking distance. Physical qvel is separately monitored by the env.
        margin = 0.06
        lower = (-self.joint_limit + margin - q) / dt
        upper = (self.joint_limit - margin - q) / dt
        command = np.clip(command, np.minimum(lower, 0), np.maximum(upper, 0))
        self.previous = np.clip(command, -speed, speed)
        return self.previous.copy()
