"""Prepare isolated splits, train bounded residual PPO, and evaluate frozen models."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402
from gymnasium.utils.env_checker import check_env as gym_check  # noqa: E402
from stable_baselines3.common.env_checker import check_env  # noqa: E402
from src.so101_task import SO101PickPlaceEnv, generate_case, cases_for  # noqa: E402


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def episode(env, case, model=None, random=False):
    obs, _ = env.reset(seed=case["seed"], options={"case": case})
    rng = np.random.default_rng(case["seed"] + 97)
    velocities = []
    actions = []
    while True:
        action = (
            rng.uniform(-1, 1, 5)
            if random
            else (
                np.zeros(5)
                if model is None
                else model.predict(obs, deterministic=True)[0]
            )
        )
        obs, _, term, trunc, info = env.step(action)
        velocities.append(info["tip_speed_m_s"])
        actions.append(np.abs(action).mean())
        if term or trunc:
            return {
                "case_id": case["id"],
                "seed": case["seed"],
                **info,
                "mean_tip_speed_m_s": float(np.mean(velocities)),
                "mean_abs_residual": float(np.mean(actions)),
            }


class Validation(BaseCallback):
    def __init__(self, output, every=32768):
        super().__init__()
        self.output, self.every = output, every
        self.env = SO101PickPlaceEnv()
        self.rows, self.last, self.best = [], 0, (-1, -float("inf"))
        self.episodes = []

    def evaluate(self):
        rows = [episode(self.env, case, self.model) for case in cases_for("validation")]
        rate = np.mean([r["is_success"] for r in rows])
        duration = np.mean([r["elapsed_time_s"] for r in rows])
        record = {
            "timesteps": self.num_timesteps,
            "success_rate": float(rate),
            "mean_time_s": float(duration),
        }
        self.rows.append(record)
        if (rate, -duration) > self.best:
            self.best = (rate, -duration)
            self.model.save(self.output / "best_model")
        write_json(self.output / "validation.json", self.rows)
        print("SO101 validation", record, flush=True)

    def _on_training_start(self):
        self.evaluate()

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.episodes.append(
                    {
                        "timesteps": self.num_timesteps,
                        "success": info["is_success"],
                        "return": info["episode"]["r"],
                        "length": info["episode"]["l"],
                    }
                )
        if self.num_timesteps - self.last >= self.every:
            self.evaluate()
            self.last = self.num_timesteps
        return True

    def _on_training_end(self):
        self.evaluate()
        pd.DataFrame(self.episodes).to_csv(
            self.output / "training_episodes.csv", index=False
        )
        self.env.close()


def prepare():
    for split, count, start in [
        ("train", 128, 1200000),
        ("validation", 30, 1300000),
        ("test", 100, 1400000),
    ]:
        path = ROOT / f"data/so101_{split}.json"
        cases = [generate_case(start + i) for i in range(count)]
        if path.exists() and json.loads(path.read_text(encoding="utf8")) != cases:
            raise FileExistsError(f"Refusing to change existing split: {path}")
        write_json(path, cases)
    e = SO101PickPlaceEnv()
    try:
        gym_check(e, skip_render_check=True)
        check_env(e)
        first = episode(e, cases_for("validation")[0])
        print("Gymnasium + SB3 passed. IK validation first case:", first, flush=True)
    finally:
        e.close()


def train(seed, steps):
    out = ROOT / f"checkpoints/so101_seed{seed}"
    if (out / "training.json").exists():
        raise FileExistsError(f"Completed model already exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    env = DummyVecEnv(
        [
            lambda: Monitor(
                SO101PickPlaceEnv(training=True), info_keywords=("is_success",)
            )
            for _ in range(4)
        ]
    )
    env.seed(seed)
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        seed=seed,
        verbose=0,
        n_steps=512,
        batch_size=128,
        learning_rate=3e-4,
        ent_coef=0.001,
        policy_kwargs={"net_arch": [64, 64]},
    )
    model.save(out / "initial_model")
    cb = Validation(out)
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    try:
        model.learn(total_timesteps=steps, callback=cb)
        model.save(out / "final_model")
        obs = env.reset()
        restored = PPO.load(out / "final_model.zip", device="cpu")
        np.testing.assert_array_equal(
            model.predict(obs, deterministic=True)[0],
            restored.predict(obs, deterministic=True)[0],
        )
        write_json(
            out / "training.json",
            {
                "seed": seed,
                "started_utc": started_utc,
                "wall_time_s": time.perf_counter() - started,
                "actual_timesteps": model.num_timesteps,
                "requested_timesteps": steps,
                "device": "cpu",
                "resume": None,
                "n_envs": 4,
                "algorithm": "PPO bounded residual joint-speed control; IK waypoints and rule gripper",
                "action_scale": 0.35,
                "save_reload_equal": True,
                "heldout_used_for_selection": False,
            },
        )
    finally:
        env.close()


def evaluate(seeds, output):
    output = ROOT / output
    if (output / "episode_results.csv").exists():
        raise FileExistsError(
            "Use a new output directory; final test results are immutable"
        )
    torch.set_num_threads(1)
    if len(set(seeds)) != len(seeds) or len(seeds) < 2 or 42 not in seeds:
        raise ValueError("Use independent seeds including the preselected demo seed 42")
    paths = [ROOT / f"checkpoints/so101_seed{s}/best_model.zip" for s in seeds]
    for seed in seeds:
        meta = json.loads(
            (ROOT / f"checkpoints/so101_seed{seed}/training.json").read_text(
                encoding="utf8"
            )
        )
        assert (
            meta["seed"] == seed
            and meta["resume"] is None
            and meta["save_reload_equal"]
        )
    split_sets = [
        {c["seed"] for c in cases_for(split)}
        for split in ["train", "validation", "test"]
    ]
    assert all(
        split_sets[i].isdisjoint(split_sets[j])
        for i in range(3)
        for j in range(i + 1, 3)
    )
    protocol = {
        "models": [
            {
                "seed": s,
                "file": p.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for s, p in zip(seeds, paths)
        ],
        "test_cases_sha256": hashlib.sha256(
            (ROOT / "data/so101_test.json").read_bytes()
        ).hexdigest(),
        "test_cases": 100,
        "demo_seed_selected_before_test": 42,
        "controller": "IK waypoints + PPO residual joint speeds + staged gripper",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "modes": {"standard": [1.2, 3.0], "fast": [1.2, 3.0], "careful": [0.4, 0.8]},
        "data_sha256": {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "data").glob("so101_*.json")
        },
        "source_sha256": {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "src").glob("so101*.py")
        },
        "demo_case_selected_before_final_test": "first case in data/so101_test.json",
    }
    write_json(output / "protocol.json", protocol)
    rows = []
    specs = [
        ("ik", None, "standard", False),
        ("random_residual", None, "standard", True),
    ]
    specs += [
        (f"ppo_seed{s}_{mode}", p, mode, False)
        for s, p in zip(seeds, paths)
        for mode in ["standard", "careful", "fast"]
    ]
    specs += [
        ("ik_careful", None, "careful", False),
        (
            "untrained_residual",
            ROOT / "checkpoints/so101_seed42/initial_model.zip",
            "standard",
            False,
        ),
    ]
    for label, path, mode, random in specs:
        model = None if path is None else PPO.load(path, device="cpu")
        e = SO101PickPlaceEnv(mode=mode)
        try:
            for case in cases_for("test"):
                rows.append({"controller": label, **episode(e, case, model, random)})
        finally:
            e.close()
        subset = [r for r in rows if r["controller"] == label]
        print(
            label,
            "success",
            sum(r["is_success"] for r in subset),
            "/",
            len(subset),
            flush=True,
        )
        pd.DataFrame(rows).to_csv(output / "episode_results.csv", index=False)
    frame = pd.DataFrame(rows)
    summary = frame.groupby("controller", sort=False).agg(
        episodes=("is_success", "size"),
        successes=("is_success", "sum"),
        success_rate=("is_success", "mean"),
        grasp_rate=("physical_grasp", "mean"),
        lift_rate=("lifted", "mean"),
        drop_rate=("dropped", "mean"),
        phase_timeout_rate=("phase_timeout", "mean"),
        mean_time_s=("elapsed_time_s", "mean"),
        mean_tip_speed_m_s=("mean_tip_speed_m_s", "mean"),
        mean_final_error_m=("final_xy_error_m", "mean"),
    )
    summary.to_csv(output / "summary.csv")
    aggregate = []
    for mode in ["standard", "careful", "fast"]:
        part = summary.loc[[f"ppo_seed{s}_{mode}" for s in seeds]]
        aggregate.append(
            {
                "mode": mode,
                "training_seeds": len(seeds),
                "cases_per_seed": 100,
                "mean_success_rate": part.success_rate.mean(),
                "sd_success_rate": part.success_rate.std(ddof=1),
                "mean_time_s": part.mean_time_s.mean(),
                "mean_tip_speed_m_s": part.mean_tip_speed_m_s.mean(),
            }
        )
    pd.DataFrame(aggregate).to_csv(output / "seed_aggregate.csv", index=False)
    print(summary.to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare", "train", "evaluate"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    p.add_argument("--steps", type=int, default=131072)
    p.add_argument("--output", type=Path, default=Path("results/so101/final"))
    a = p.parse_args()
    if a.action == "prepare":
        prepare()
    elif a.action == "train":
        for seed in a.seeds:
            train(seed, a.steps)
    else:
        evaluate(a.seeds, a.output)


if __name__ == "__main__":
    main()
