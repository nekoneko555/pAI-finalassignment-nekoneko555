import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env
from src.env import InstructionArmEnv
from src.geometry import points


@pytest.mark.parametrize("task", ["fixed", "random", "obstacle"])
def test_gymnasium_and_sb3_contracts(task):
    env = InstructionArmEnv(task=task)
    check_env(env, skip_render_check=True)
    sb3_check_env(env, warn=True)
    env.close()


def test_seed_reproduces_problem_and_trajectory():
    a, b = InstructionArmEnv(task="obstacle"), InstructionArmEnv(task="obstacle")
    x, _ = a.reset(seed=77)
    y, _ = b.reset(seed=77)
    np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(a.obstacle, b.obstacle)
    for _ in range(10):
        u = np.array([0.2, -0.1], np.float32)
        x = a.step(u)
        y = b.step(u)
        np.testing.assert_array_equal(x[0], y[0])
        assert x[1:4] == y[1:4]


def test_timeout_is_not_termination():
    env = InstructionArmEnv(max_steps=1)
    env.reset(seed=42)
    obs, _, term, trunc, _ = env.step(np.zeros(2))
    assert not term and trunc
    assert env.observation_space.contains(obs)
    with pytest.raises(RuntimeError):
        env.step(np.zeros(2))


def test_goal_success_and_no_double_count():
    env = InstructionArmEnv(max_steps=1)
    q = [-0.8, 1.6]
    env.reset(
        options={"case": {"initial_q": q, "goal": points(q)[2], "obstacle": None}}
    )
    _, _, term, trunc, info = env.step(np.zeros(2))
    assert term and not trunc and info["is_success"]


@pytest.mark.parametrize("action", [[np.nan, 0], [np.inf, 0], [0, 0, 0]])
def test_invalid_actions_are_rejected(action):
    env = InstructionArmEnv()
    env.reset()
    with pytest.raises(ValueError):
        env.step(action)


@pytest.mark.parametrize(
    "case",
    [
        {"initial_q": [0, 0], "goal": [2, 2]},
        {"initial_q": [3, 0], "goal": [0.3, 0.3]},
        {"initial_q": [0, 0], "goal": [0.3, 0.3], "obstacle": [0.2, 0, 0.05]},
    ],
)
def test_invalid_cases_rejected(case):
    with pytest.raises(ValueError):
        InstructionArmEnv().reset(options={"case": case})


def test_full_link_collision_not_just_tip():
    env = InstructionArmEnv()
    env.reset(
        options={
            "case": {
                "initial_q": [-0.8, 1.6],
                "goal": [0.0, 0.5],
                "obstacle": [0.18, 0.01, 0.055],
            }
        }
    )
    # A fault-injection state hits the middle of link1, while the tip stays clear.
    env.data.qpos[:] = [0, 1]
    import mujoco

    mujoco.mj_forward(env.model, env.data)
    assert np.linalg.norm(env.tip - env.obstacle[:2]) > 0.2
    _, _, term, _, info = env.step(np.zeros(2))
    assert term and info["collision"] and not info["is_success"]


def test_baseline_reaches_fixed_goal():
    from src.baseline import ik_action

    env = InstructionArmEnv()
    env.reset(seed=42)
    for _ in range(200):
        obs, _, term, trunc, info = env.step(ik_action(env))
        assert env.observation_space.contains(obs)
        if term or trunc:
            break
    assert info["is_success"] and not info["collision"]


def test_corrupted_physics_state_terminates_with_finite_observation():
    env = InstructionArmEnv()
    env.reset(seed=42)
    env.data.qpos[0] = np.nan
    obs, reward, term, trunc, info = env.step([0, 0])
    assert term and not trunc and info["invalid_state"]
    assert np.isfinite(obs).all() and env.observation_space.contains(obs)


def test_real_render_frame():
    env = InstructionArmEnv(render_mode="rgb_array", width=480, height=320)
    try:
        env.reset(seed=42)
        frame = env.render()
        assert frame.shape == (320, 480, 3) and frame.dtype == np.uint8
        assert frame.std() > 10
    finally:
        env.close()
