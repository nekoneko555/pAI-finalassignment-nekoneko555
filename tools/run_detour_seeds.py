"""Run independent initializations sequentially; never select using final test cases."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[123, 456])
    args = parser.parse_args()
    for seed in args.seeds:
        output = ROOT / "checkpoints" / f"detour_seed{seed}"
        if (output / "training.json").exists():
            print(f"Keeping completed seed {seed}", flush=True)
            continue
        command = [
            sys.executable,
            "-m",
            "src.train",
            "--task",
            "detour",
            "--timesteps",
            "300000",
            "--curriculum-steps",
            "200000",
            "--validation-every",
            "25000",
            "--seed",
            str(seed),
            "--n-envs",
            "4",
            "--output",
            str(output.relative_to(ROOT)),
        ]
        print("Running: " + " ".join(command[1:]), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
