import numpy as np
import pandas as pd
from src.benchmark_detour import aggregate_seeds
from src.detour import DetourArmEnv


def test_seed_sd_is_not_episode_sd():
    frame = pd.DataFrame(
        {
            "mode": ["standard"] * 3,
            "episodes": [100] * 3,
            "success_rate": [0.7, 0.8, 0.9],
            "collision_rate": [0.3, 0.2, 0.1],
            "mean_speed": [0.2, 0.3, 0.4],
        }
    )
    result = aggregate_seeds(frame).iloc[0]
    assert result.training_seeds == 3 and result.cases_per_seed == 100
    assert np.isclose(result.mean_success_rate, 0.8)
    assert np.isclose(result.sd_success_rate, 0.1)


def test_curriculum_resets_revert_to_full_start():
    env = DetourArmEnv(training=True)
    env.set_curriculum(0)
    env.reset(seed=42)
    easy = env.data.qpos.copy()
    env.set_curriculum(1)
    obs, _ = env.reset(seed=42)
    np.testing.assert_allclose(env.data.qpos, [-0.8, 1.6])
    assert not np.allclose(easy, env.data.qpos)
    assert obs.shape == (14,)
    env.close()


def test_changed_obstacle_radius_changes_actual_contacts():
    from src.env import InstructionArmEnv
    import mujoco

    env = InstructionArmEnv()
    for radius, expected in [(0.02, False), (0.10, True), (0.02, False)]:
        env.reset(
            options={
                "case": {
                    "initial_q": [-0.8, 1.6],
                    "goal": [0, 0.6],
                    "obstacle": [0.2, 0.105, radius],
                }
            }
        )
        env.data.qpos[:] = [0, 1]
        mujoco.mj_forward(env.model, env.data)
        assert env._touching() == expected
    env.close()
