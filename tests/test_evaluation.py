import numpy as np
import pandas as pd
from src.evaluate import summarize, rollout
from src.env import InstructionArmEnv
from src.scenarios import generate_cases


def test_completion_time_uses_successes_only():
    df = pd.DataFrame(
        [
            dict(
                controller="ik",
                success=True,
                collision=False,
                invalid_state=False,
                completion_time_s=2.0,
                min_goal_distance_m=0.04,
                mean_speed=0.2,
                mean_abs_action_delta=0.1,
            ),
            dict(
                controller="ik",
                success=False,
                collision=True,
                invalid_state=False,
                completion_time_s=np.nan,
                min_goal_distance_m=0.3,
                mean_speed=0.1,
                mean_abs_action_delta=0.1,
            ),
        ]
    )
    row = summarize(df).iloc[0]
    assert row.success_rate == 0.5 and row.collision_rate == 0.5
    assert row.mean_completion_time_s == 2.0


def test_zero_successes_have_no_completion_time():
    env = InstructionArmEnv(max_steps=1)
    row = rollout(env, generate_cases(1)[0], "random")
    assert (
        not row["success"] and np.isnan(row["completion_time_s"]) and row["truncated"]
    )


def test_cases_are_deterministic_and_feasible():
    a = generate_cases(20, seed=901, task="obstacle")
    assert a == generate_cases(20, seed=901, task="obstacle")
    env = InstructionArmEnv()
    from src.baseline import ik_action

    for case in a:
        env.reset(options={"case": case})
        for _ in range(200):
            _, _, term, trunc, info = env.step(ik_action(env))
            if term or trunc:
                break
        assert info["is_success"] and not info["collision"]
