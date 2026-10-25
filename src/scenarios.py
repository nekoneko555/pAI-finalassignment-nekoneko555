"""Separate held-out cases from training and checkpoint-selection cases."""

import argparse
import json
from pathlib import Path
from .config import ROOT
from .env import InstructionArmEnv


def generate_cases(count=100, seed=20261023, task="random"):
    if count < 1:
        raise ValueError("count must be positive")
    env = InstructionArmEnv(task=task)
    cases = []
    for i in range(count):
        env.reset(seed=seed + i)
        cases.append(
            {
                "id": f"{task}-{seed}-{i:03d}",
                "seed": seed + i,
                "initial_q": env.data.qpos.tolist(),
                "goal": env.goal.tolist(),
                "obstacle": None if env.obstacle is None else env.obstacle.tolist(),
            }
        )
    env.close()
    return cases


def load_cases(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--task", choices=["fixed", "random", "obstacle"], default="random")
    p.add_argument("--seed", type=int, default=20261023)
    p.add_argument("--output", type=Path, default=ROOT / "heldout_eval_cases.json")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(
            "Preserve existing evaluation cases; choose a new output filename"
        )
    cases = generate_cases(args.count, args.seed, args.task)
    args.output.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"Saved {len(cases)} cases to {args.output}")


if __name__ == "__main__":
    main()
