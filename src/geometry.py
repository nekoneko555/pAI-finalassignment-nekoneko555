import numpy as np

L1, L2 = 0.35, 0.30
LINK_RADIUS = 0.022


def points(q):
    a, b = q
    elbow = L1 * np.array([np.cos(a), np.sin(a)])
    tip = elbow + L2 * np.array([np.cos(a + b), np.sin(a + b)])
    return np.zeros(2), elbow, tip


def inverse_kinematics(goal):
    x, y = goal
    c = (x * x + y * y - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if not -1 <= c <= 1:
        raise ValueError("Unreachable target")
    b = np.arccos(c)
    a = np.arctan2(y, x) - np.arctan2(L2 * np.sin(b), L1 + L2 * np.cos(b))
    return np.array([a, b])


def segment_distance(p, a, b):
    t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1)
    return float(np.linalg.norm(p - (a + t * (b - a))))


def clearance(q, obstacle):
    if obstacle is None:
        return float("inf")
    a, b, c = points(q)
    p = np.asarray(obstacle[:2])
    return (
        min(segment_distance(p, a, b), segment_distance(p, b, c))
        - obstacle[2]
        - LINK_RADIUS
    )


def path_clearance(waypoints, obstacle, resolution=0.01):
    """Conservative sampled certificate for both entire links along a route."""
    route = np.asarray(waypoints, dtype=float)
    samples = np.concatenate(
        [
            np.linspace(
                a, b, max(2, int(np.ceil(np.max(np.abs(b - a)) / resolution)) + 1)
            )
            for a, b in zip(route[:-1], route[1:])
        ]
    )
    elbow = L1 * np.column_stack((np.cos(samples[:, 0]), np.sin(samples[:, 0])))
    tip = elbow + L2 * np.column_stack(
        (np.cos(samples.sum(axis=1)), np.sin(samples.sum(axis=1)))
    )
    p = np.asarray(obstacle[:2])
    distances = []
    for a, b in [(np.zeros_like(elbow), elbow), (elbow, tip)]:
        v = b - a
        t = np.clip(((p - a) * v).sum(axis=1) / (v * v).sum(axis=1), 0, 1)
        distances.append(np.linalg.norm(p - a - t[:, None] * v, axis=1))
    return float(np.min(distances) - obstacle[2] - LINK_RADIUS)
