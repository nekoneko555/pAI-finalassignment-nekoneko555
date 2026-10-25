"""Analytic IK followed by proportional joint velocity tracking; no avoidance."""

import numpy as np
from .geometry import inverse_kinematics


def ik_action(env):
    target = inverse_kinematics(env.goal)
    error = target - env.data.qpos
    speed = env.filter.hard_speed * env.filter.limits.speed_scale
    return np.clip(3.0 * error / speed, -1, 1).astype(np.float32)
