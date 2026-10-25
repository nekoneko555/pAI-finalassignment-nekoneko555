"""Frozen-model paired benchmark, with seed variability kept separate from trials."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from .baseline import ik_action
from .config import ROOT
from .detour import DetourArmEnv, waypoint_action
from .evaluate import summarize
from .scenarios import load_cases


def evaluate_case(env, case, kind, model=None):
    obs, _ = env.reset(seed=case["seed"], options={"case": case})
    rng = np.random.default_rng(case["seed"])
    speeds, deltas = [], []
    previous = np.zeros(2)
    route_index = 1
    reward_sum = 0.0
    while True:
        if kind == "random":
            action = rng.uniform(-1, 1, 2).astype(np.float32)
        elif kind == "ik":
            action = ik_action(env)
        elif kind == "oracle":
            action, route_index = waypoint_action(
                env, case["certified_waypoints"], route_index
            )
        else:
            action = model.predict(obs, deterministic=True)[0]
        obs, reward, term, trunc, info = env.step(action)
        reward_sum += reward
        speeds.append(info.get("tip_speed_m_s", 0))
        deltas.append(float(np.abs(info["command"] - previous).mean()))
        previous = info["command"].copy()
        if term or trunc:
            break
    return {
        "case_id": case["id"],
        "case_seed": case["seed"],
        "success": info["is_success"],
        "collision": info["collision"],
        "invalid_state": info["invalid_state"],
        "truncated": trunc,
        "elapsed_time_s": info["elapsed_time_s"],
        "completion_time_s": info["elapsed_time_s"] if info["is_success"] else np.nan,
        "min_goal_distance_m": info["min_goal_distance_m"],
        "mean_speed": float(np.mean(speeds)),
        "mean_abs_action_delta": float(np.mean(deltas)),
        "total_reward": reward_sum,
    }


def aggregate_seeds(seed_summary):
    rows = []
    for mode, group in seed_summary.groupby("mode", sort=False):
        rows.append(
            {
                "mode": mode,
                "training_seeds": len(group),
                "cases_per_seed": int(group.episodes.iloc[0]),
                "mean_success_rate": float(group.success_rate.mean()),
                "sd_success_rate": float(group.success_rate.std(ddof=1))
                if len(group) > 1
                else np.nan,
                "mean_collision_rate": float(group.collision_rate.mean()),
                "sd_collision_rate": float(group.collision_rate.std(ddof=1))
                if len(group) > 1
                else np.nan,
                "mean_speed": float(group.mean_speed.mean()),
            }
        )
    return pd.DataFrame(rows)


def plots(output):
    output = Path(output)
    summary = pd.read_csv(output / "summary.csv")
    selected = summary[summary.controller.isin(["ik", "previous_ppo", "oracle"])]
    aggregate = pd.read_csv(output / "seed_aggregate.csv")
    short = {
        "ik": "Direct\nIK",
        "oracle": "Route\noracle",
        "previous_ppo": "Previous\nPPO",
    }
    labels = [short[n] for n in selected.controller] + [
        f"PPO\n{r.mode}\n{r.training_seeds} seeds" for r in aggregate.itertuples()
    ]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, metric in zip(axs, ["success", "collision"]):
        values = (
            selected[f"{metric}_rate"].tolist()
            + aggregate[f"mean_{metric}_rate"].tolist()
        )
        errors = [0.0] * len(selected) + aggregate[f"sd_{metric}_rate"].fillna(
            0
        ).tolist()
        ax.bar(
            labels,
            values,
            yerr=errors,
            capsize=4,
            color=["#30a7b5", "#8e9bb3", "#6a7b9c", "#6685e3", "#39b98d", "#e8ae45"],
        )
        ax.set_ylim(0, 1.12)
        ax.set_title(f"{metric.title()} rate | error bars: SD across training seeds")
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", labelsize=8)
    count = int(summary.episodes.iloc[0])
    fig.suptitle(f"Blocked direct routes | {count} identical heldout cases per model")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/detour_test.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/detour")
    args = parser.parse_args()
    if len(args.models) != len(args.seeds):
        parser.error("One seed per model is required")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Training seeds must be unique")
    for seed, path in zip(args.seeds, args.models):
        metadata = json.loads(
            (path.parent / "training.json").read_text(encoding="utf-8")
        )
        if metadata["seed"] != seed or metadata["resume"] is not None:
            parser.error(
                "Expected independent initializations with matching seed metadata"
            )
    if (args.output / "episode_results.csv").exists():
        raise FileExistsError(
            "Choose a new output directory to preserve the frozen benchmark"
        )
    torch.set_num_threads(1)
    models = [PPO.load(path, device="cpu") for path in args.models]
    previous = PPO.load(ROOT / "checkpoints/best_model.zip", device="cpu")
    cases = load_cases(args.cases)
    protocol = {
        "cases_file": args.cases.name,
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "train_cases_sha256": hashlib.sha256(
            (ROOT / "data/detour_train.json").read_bytes()
        ).hexdigest(),
        "validation_cases_sha256": hashlib.sha256(
            (ROOT / "data/detour_validation.json").read_bytes()
        ).hexdigest(),
        "models": [
            {
                "seed": seed,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for seed, path in zip(args.seeds, args.models)
        ],
        "episodes_per_model": len(cases),
        "demo_seed_selected_before_final_test": 42,
        "selection": "Per-seed maximum validation success rate, then minimum distance",
        "oracle_note": "Privileged route-certification baseline; not a learned policy",
        "statistics_note": "Same scenarios reused across seeds; SD is across independent training runs, not 300 independent scenarios",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    controllers = [
        ("random", "standard", "random", None, None),
        ("ik", "standard", "ik", None, None),
        ("oracle", "standard", "oracle", None, None),
        ("previous_ppo", "standard", "ppo", previous, None),
    ]
    for seed, model in zip(args.seeds, models):
        for mode in ("standard", "careful", "fast"):
            controllers.append((f"ppo_seed{seed}_{mode}", mode, "ppo", model, seed))
    started = time.perf_counter()
    rows = []
    for name, mode, kind, model, seed in controllers:
        env = DetourArmEnv(mode=mode)
        try:
            for case in cases:
                row = evaluate_case(env, case, kind, model)
                row.update(controller=name, mode=mode, training_seed=seed)
                rows.append(row)
        finally:
            env.close()
        print(
            f"{name}: {sum(r['success'] for r in rows if r['controller'] == name)}/{len(cases)} successes",
            flush=True,
        )
    df = pd.DataFrame(rows)
    df.to_csv(args.output / "episode_results.csv", index=False)
    summary = summarize(df)
    summary.to_csv(args.output / "summary.csv", index=False)
    seed_summary = summary.merge(
        df[["controller", "training_seed", "mode"]].drop_duplicates(), on="controller"
    )
    seed_summary = seed_summary[seed_summary.training_seed.notna()]
    seed_summary.to_csv(args.output / "seed_summary.csv", index=False)
    aggregate_seeds(seed_summary).to_csv(
        args.output / "seed_aggregate.csv", index=False
    )
    plots(args.output)
    protocol["wall_time_s"] = time.perf_counter() - started
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
