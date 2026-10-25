import mujoco
from gymnasium.utils.env_checker import check_env
from src.baseline import ik_action
from src.detour import DetourArmEnv, generate_detour_cases, waypoint_action
from src.geometry import path_clearance


def test_blocked_cases_have_certified_clear_detours():
    cases = generate_detour_cases(12, 610000)
    assert cases == generate_detour_cases(12, 610000)
    for case in cases:
        route = case["certified_waypoints"]
        assert path_clearance([route[0], route[-1]], case["obstacle"]) < 0
        assert path_clearance(route, case["obstacle"]) > 0.018


def test_oracle_reaches_but_direct_ik_collides():
    env = DetourArmEnv()
    for case in generate_detour_cases(12, 610000):
        for policy in ["oracle", "ik"]:
            env.reset(seed=case["seed"], options={"case": case})
            index = 1
            for _ in range(200):
                if policy == "oracle":
                    action, index = waypoint_action(
                        env, case["certified_waypoints"], index
                    )
                else:
                    action = ik_action(env)
                _, _, term, trunc, info = env.step(action)
                if term or trunc:
                    break
            if policy == "oracle":
                assert info["is_success"] and not info["collision"]
            else:
                assert info["collision"] and not info["is_success"]
    env.close()


def test_detour_contract_and_radius_bounds():
    env = DetourArmEnv()
    check_env(env, skip_render_check=True)
    case = generate_detour_cases(1, 610001)[0]
    env.reset(options={"case": case})
    g = env._obstacle_id
    radius = case["obstacle"][2]
    assert env.model.geom_aabb[g, 3] == radius
    assert env.model.geom_rbound[g] >= radius
    mujoco.mj_forward(env.model, env.data)
    assert not env._touching()
    env.close()
