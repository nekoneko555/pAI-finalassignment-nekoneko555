"""CPU PPO; checkpoint selection uses validation cases, never heldout_eval_cases."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import torch
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .config import ROOT, load_config
from .env import InstructionArmEnv
from .scenarios import generate_cases
from .scenarios import load_cases


def make_task_env(task, training=False):
    if task == "detour":
        from .detour import DetourArmEnv

        return DetourArmEnv(training=training)
    return InstructionArmEnv(task=task)


class ValidationCallback(BaseCallback):
    def __init__(self, output, task, every=10000, cases_path=None):
        super().__init__()
        self.output = Path(output)
        self.cases = (
            load_cases(cases_path)
            if cases_path
            else generate_cases(10, seed=708000, task=task)
        )
        self.env = make_task_env(task)
        self.every = every
        self.last = 0
        self.best = (-1.0, -float("inf"))
        self.rows = []

    def evaluate_now(self):
        successes, distances, collisions = [], [], []
        for case in self.cases:
            obs, _ = self.env.reset(seed=case["seed"], options={"case": case})
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = self.env.step(action)
                done = term or trunc
            successes.append(info["is_success"])
            collisions.append(info["collision"])
            distances.append(info["min_goal_distance_m"])
        rate = float(np.mean(successes))
        row = {
            "timesteps": self.num_timesteps,
            "success_rate": rate,
            "collision_rate": float(np.mean(collisions)),
            "mean_min_goal_distance_m": float(np.mean(distances)),
        }
        self.rows.append(row)
        score = (rate, -float(np.mean(distances)))
        if score > self.best:
            self.best = score
            self.model.save(self.output / "best_model")
        (self.output / "validation.json").write_text(
            json.dumps(self.rows, indent=2), encoding="utf-8"
        )
        self.logger.record("validation/success_rate", rate)
        self.logger.record("validation/collision_rate", row["collision_rate"])
        print(f"Validation: {row}", flush=True)

    def _on_step(self):
        if self.num_timesteps - self.last >= self.every:
            self.evaluate_now()
            self.last = self.num_timesteps
        return True

    def _on_training_start(self):
        # SB3 creates its logger in learn(), before invoking this callback.
        self.evaluate_now()

    def _on_training_end(self):
        self.evaluate_now()
        self.env.close()


class EpisodeLogCallback(BaseCallback):
    def __init__(self, output):
        super().__init__()
        self.output = Path(output)
        self.rows = []

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.rows.append(
                    {
                        "timesteps": self.num_timesteps,
                        "success": info.get("is_success", False),
                        "collision": info.get("collision", False),
                        "reward": info["episode"]["r"],
                        "length": info["episode"]["l"],
                    }
                )
        return True

    def _on_training_end(self):
        pd.DataFrame(self.rows).to_csv(
            self.output / "training_episodes.csv", index=False
        )


class CurriculumCallback(BaseCallback):
    def __init__(self, steps):
        super().__init__()
        self.steps = steps
        self.last = -4096

    def _on_step(self):
        if self.num_timesteps - self.last >= 4096:
            self.training_env.env_method(
                "set_curriculum", min(1, self.num_timesteps / max(1, self.steps))
            )
            self.last = self.num_timesteps
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--task", choices=["fixed", "random", "obstacle", "detour"], default="fixed"
    )
    p.add_argument("--validation-cases", type=Path)
    p.add_argument("--validation-every", type=int, default=10000)
    p.add_argument("--curriculum-steps", type=int, default=200000)
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--subproc", action="store_true")
    p.add_argument("--resume", type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "checkpoints")
    args = p.parse_args()
    if args.timesteps < 1 or args.n_envs < 1:
        p.error("timesteps and n-envs must be positive")
    if args.validation_every < 1 or args.curriculum_steps < 0:
        p.error("Invalid validation frequency or curriculum length")
    if args.task == "detour" and args.validation_cases is None:
        args.validation_cases = ROOT / "data/detour_validation.json"
    if args.validation_cases and any(
        s in args.validation_cases.name.lower() for s in ("heldout", "test")
    ):
        p.error("Do not use final evaluation cases to select checkpoints")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    check = make_task_env(args.task)
    check_env(check)
    check.close()
    factories = [
        lambda: Monitor(
            make_task_env(args.task, training=True),
            info_keywords=("is_success", "collision"),
        )
        for _ in range(args.n_envs)
    ]
    env = (SubprocVecEnv if args.subproc else DummyVecEnv)(factories)
    env.seed(args.seed)
    if args.task == "detour":
        env.env_method("set_curriculum", 0 if args.curriculum_steps else 1)
    cfg = load_config()["ppo"]
    if args.resume:
        model = PPO.load(
            str(args.resume),
            env=env,
            device="cpu",
            seed=args.seed,
            tensorboard_log="logs",
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            device="cpu",
            seed=args.seed,
            verbose=0,
            learning_rate=cfg["learning_rate"],
            gamma=cfg["gamma"],
            n_steps=cfg["n_steps"],
            batch_size=cfg["batch_size"],
            policy_kwargs={"net_arch": cfg["net_arch"]},
            tensorboard_log="logs",
        )
    validation = ValidationCallback(
        args.output,
        args.task,
        every=args.validation_every,
        cases_path=args.validation_cases,
    )
    model.save(args.output / "initial_model")
    checkpoint = CheckpointCallback(
        save_freq=max(1, 10000 // args.n_envs),
        save_path=str(args.output / "snapshots"),
        name_prefix="ppo",
    )
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=CallbackList(
                [validation, checkpoint, EpisodeLogCallback(args.output)]
                + (
                    [CurriculumCallback(args.curriculum_steps)]
                    if args.task == "detour" and args.curriculum_steps
                    else []
                )
            ),
            tb_log_name=f"{args.task}_seed{args.seed}",
            progress_bar=False,
        )
        model.save(args.output / "final_model")
        # Reload the exact serialized checkpoint and compare deterministic actions.
        obs = env.reset()
        reloaded = PPO.load(str(args.output / "best_model.zip"), device="cpu")
        best_actions = reloaded.predict(obs, deterministic=True)[0]
        assert (
            best_actions.shape == (args.n_envs, 2) and np.isfinite(best_actions).all()
        )
        final_loaded = PPO.load(str(args.output / "final_model.zip"), device="cpu")
        np.testing.assert_array_equal(
            model.predict(obs, deterministic=True)[0],
            final_loaded.predict(obs, deterministic=True)[0],
        )
        metadata = {
            "started_utc": started_utc,
            "wall_time_s": time.perf_counter() - started,
            "requested_timesteps": args.timesteps,
            "actual_timesteps": model.num_timesteps,
            "seed": args.seed,
            "task": args.task,
            "n_envs": args.n_envs,
            "device": "cpu",
            "resume": None if args.resume is None else str(args.resume),
            "save_reload_action_equal": True,
            "validation_cases": len(validation.cases),
            "validation_seed": validation.cases[0]["seed"],
            "validation_cases_file": None
            if args.validation_cases is None
            else args.validation_cases.name,
            "curriculum_steps": args.curriculum_steps
            if args.task == "detour"
            else None,
            "reward_profile": "original_minus_0.3_distance"
            if args.task == "detour"
            else "original",
            "heldout_used_for_selection": False,
        }
        (args.output / "training.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(json.dumps(metadata, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
