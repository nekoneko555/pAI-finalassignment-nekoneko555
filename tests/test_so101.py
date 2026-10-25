import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from src.so101_task import SO101PickPlaceEnv, generate_case
from src.so101_env import make_model


def test_real_model_has_five_arm_joints_and_gripper():
    model = make_model()
    assert model.nu == 6
    assert model.nmesh >= 15
    assert model.joint("wrist_roll").id >= 0


def test_pickplace_gym_determinism_and_action_validation():
    env = SO101PickPlaceEnv()
    try:
        check_env(env, skip_render_check=True)
        first, _ = env.reset(seed=42)
        second, _ = env.reset(seed=42)
        np.testing.assert_array_equal(first, second)
        for action in [np.ones(4), np.full(5, np.nan), np.full(5, np.inf)]:
            with pytest.raises(ValueError):
                env.step(action)
    finally:
        env.close()


def test_physical_grasp_carry_release_without_attachment():
    env = SO101PickPlaceEnv()
    try:
        env.reset(
            options={
                "case": {
                    "id": "fixture",
                    "seed": 777,
                    "a": [0.23, -0.09],
                    "b": [0.23, 0.09],
                }
            }
        )
        assert env.sim.model.neq == 0
        while True:
            _, _, t, tr, info = env.step(np.zeros(5))
            if t or tr:
                break
        assert info["is_success"], info
        assert info["physical_grasp"] and info["lifted"] and info["released"]
        assert info["max_object_height_m"] > 0.06
        assert info["final_xy_error_m"] < 0.02
        assert any(r["object_y"] > 0.05 and r["object_z"] > 0.05 for r in env.sim.trace)
    finally:
        env.close()


def test_careful_command_bounds_and_native_render():
    env = SO101PickPlaceEnv(mode="careful", render_mode="rgb_array")
    try:
        env.reset(seed=101)
        previous = np.zeros(5)
        for _ in range(25):
            env.step(np.ones(5))
            current = env.sim.velocity.copy()
            assert np.abs(current).max() <= 0.4 + 1e-12
            assert np.abs(current - previous).max() <= 0.8 * 0.04 + 1e-12
            previous = current
        frame = env.render()
        assert frame.shape == (640, 960, 3) and frame.std() > 15
    finally:
        env.close()


def test_invalid_case_and_timeout():
    env = SO101PickPlaceEnv(max_steps=1)
    try:
        with pytest.raises(ValueError):
            env.reset(options={"case": {"a": [2, 2], "b": [0.23, 0.09]}})
        env.reset(options={"case": generate_case(888)})
        _, _, terminated, truncated, _ = env.step(np.zeros(5))
        assert truncated and not terminated
        with pytest.raises(RuntimeError):
            env.step(np.zeros(5))
    finally:
        env.close()
