"""Paired evaluation with per-episode raw data; no completion time for failures."""

import argparse
import hashlib
from dataclasses import asdict
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .baseline import ik_action
from .config import ROOT
from .env import InstructionArmEnv
from .scenarios import load_cases


def rollout(env, case, policy, model=None):
    obs, _ = env.reset(seed=case["seed"], options={"case": case})
    rng = np.random.default_rng(case["seed"])
    speeds, deltas = [], []
    previous = np.zeros(2)
    done = False
    total_reward = 0.0
    while not done:
        if policy == "random":
            action = rng.uniform(-1, 1, 2).astype(np.float32)
        elif policy == "ik":
            action = ik_action(env)
        else:
            action = model.predict(obs, deterministic=True)[0]
        obs, reward, term, trunc, info = env.step(action)
        speeds.append(info["tip_speed_m_s"])
        deltas.append(float(np.abs(info["command"] - previous).mean()))
        previous = info["command"].copy()
        total_reward += reward
        done = term or trunc
    return {
        "case_id": case["id"],
        "seed": case["seed"],
        "success": info["is_success"],
        "collision": info["collision"],
        "invalid_state": info["invalid_state"],
        "truncated": trunc,
        "elapsed_time_s": info["elapsed_time_s"],
        "completion_time_s": info["elapsed_time_s"] if info["is_success"] else np.nan,
        "min_goal_distance_m": info["min_goal_distance_m"],
        "mean_speed": float(np.mean(speeds)),
        "mean_abs_action_delta": float(np.mean(deltas)),
        "total_reward": total_reward,
    }


def summarize(df):
    return (
        df.groupby("controller", sort=False)
        .agg(
            episodes=("success", "size"),
            successes=("success", "sum"),
            success_rate=("success", "mean"),
            collision_rate=("collision", "mean"),
            invalid_state_rate=("invalid_state", "mean"),
            mean_completion_time_s=("completion_time_s", "mean"),
            mean_min_goal_distance_m=("min_goal_distance_m", "mean"),
            mean_speed=("mean_speed", "mean"),
            mean_abs_action_delta=("mean_abs_action_delta", "mean"),
        )
        .reset_index()
    )


def plot_summary(summary, path):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8))
    labels = {
        "random": "Random",
        "ik": "IK",
        "ppo_untrained": "PPO\ninitial",
        "ppo_standard": "PPO\nstd",
        "ppo_careful": "PPO\ncareful",
        "ppo_fast": "PPO\nfast",
        "ppo_llm_careful": "PPO\nLLM\ncareful",
        "ppo_llm_fast": "PPO\nLLM\nfast",
    }
    names = [labels.get(name, name.replace("_", "\n")) for name in summary.controller]
    for ax, key, label in zip(
        axs,
        ["success_rate", "collision_rate", "mean_speed"],
        [
            "Success rate (all trials)",
            "Collision rate (all trials)",
            "Mean tip speed (m/s)",
        ],
    ):
        ax.bar(
            names,
            summary[key],
            color=["#8095aa", "#23acbd", "#687dea", "#39b98d", "#eead41"][
                : len(summary)
            ],
        )
        ax.set_title(label)
        ax.tick_params(axis="x", labelsize=8)
        if key.endswith("rate"):
            ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        f"Measured paired evaluation | n={int(summary.episodes.iloc[0])} per controller"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=ROOT / "checkpoints/best_model.zip")
    p.add_argument("--initial-model", type=Path)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--cases", type=Path, default=ROOT / "heldout_eval_cases.json")
    p.add_argument("--output", type=Path, default=ROOT / "results")
    p.add_argument("--baseline-only", action="store_true")
    p.add_argument("--use-llm", action="store_true")
    p.add_argument("--llm-model", default="qwen2.5:3b")
    args = p.parse_args()
    cases = load_cases(args.cases)
    if not 1 <= args.episodes <= len(cases):
        p.error("episodes must be between 1 and the available case count")
    torch.set_num_threads(1)
    cases = cases[: args.episodes]
    from stable_baselines3 import PPO

    controllers = [
        ("random", "standard", None, "explicit_mode"),
        ("ik", "standard", None, "explicit_mode"),
    ]
    model_hash = None
    interpretations = []
    if not args.baseline_only:
        model = PPO.load(str(args.model), device="cpu")
        model_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
        if args.initial_model:
            controllers.append(
                (
                    "ppo_untrained",
                    "standard",
                    PPO.load(str(args.initial_model), device="cpu"),
                    "explicit_mode",
                )
            )
        controllers += [
            ("ppo_standard", "standard", model, "explicit_mode"),
            ("ppo_careful", "careful", model, "explicit_mode"),
            ("ppo_fast", "fast", model, "explicit_mode"),
        ]
        if args.use_llm:
            from .llm_parser import parse_instruction

            for label, text in [
                ("careful", "精密部品なのでゆっくり動いて"),
                ("fast", "急ぎの工程だから速く動かして"),
            ]:
                result = parse_instruction(text, model=args.llm_model)
                interpretations.append({"text": text, **asdict(result)})
                controllers.append(
                    (f"ppo_llm_{label}", result.mode, model, result.source)
                )
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, mode, model, source in controllers:
        env = InstructionArmEnv(mode=mode)
        try:
            for case in cases:
                row = rollout(
                    env, case, name if name in ("random", "ik") else "ppo", model
                )
                row.update(controller=name, mode=mode, instruction_source=source)
                rows.append(row)
        finally:
            env.close()
        print(f"Finished {name}: {len(cases)} episodes", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output / "episode_results.csv", index=False)
    summary = summarize(df)
    summary.to_csv(args.output / "summary.csv", index=False)
    plot_summary(summary, args.output / "comparison.png")
    metadata = {
        "cases_file": args.cases.name,
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "episodes_per_controller": len(cases),
        "model": args.model.name if model_hash else None,
        "model_sha256": model_hash,
        "llm_used": any(r["source"] == "ollama" for r in interpretations),
        "llm_interpretations": interpretations,
        "llm_model": args.llm_model if args.use_llm else None,
        "action_delta_definition": "mean per RL step of abs(filtered command difference), rad/s",
        "completion_time_population": "successful episodes only; NaN when no successes",
        "case_ids": [v["id"] for v in cases],
    }
    (args.output / "evaluation.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
