"""Stream RGB frames to H.264 and decode again to verify the saved artifact."""

import argparse
import hashlib
import csv
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np

from .baseline import ik_action
from .config import ROOT
from .env import InstructionArmEnv
from .rendering import caption, title_frame
from .scenarios import generate_cases


def verify_video(path):
    reader = imageio.get_reader(str(path))
    try:
        meta = reader.get_meta_data()
        count = reader.count_frames()
        first = reader.get_data(0)
        last = reader.get_data(count - 1)
        if count < 2 or first.shape != last.shape:
            raise RuntimeError("Video verification failed")
        result = {
            "path": str(Path(path).name),
            "bytes": Path(path).stat().st_size,
            "frames": count,
            "fps": meta["fps"],
            "duration_s": count / meta["fps"],
            "frame_shape": list(first.shape),
            "decoded_first_and_last": True,
        }
        return result
    finally:
        reader.close()


def record(
    output,
    policy="ik",
    model_path=None,
    mode="standard",
    renderer="mujoco",
    cases=None,
    text="",
    source="explicit mode",
    title_seconds=2,
    minimum_seconds=0,
):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if policy not in ("ik", "ppo", "random"):
        raise ValueError("Unknown policy")
    model = None
    if policy == "ppo":
        from stable_baselines3 import PPO

        model = PPO.load(str(model_path), device="cpu")
    cases = cases or generate_cases(3, task="random")
    env = InstructionArmEnv(render_mode="rgb_array", mode=mode, renderer=renderer)
    rng = np.random.default_rng(42)
    episode_log = []
    trajectory = []
    count = 0
    with imageio.get_writer(
        str(output), fps=25, codec="libx264", quality=8, macro_block_size=1
    ) as writer:
        intro = title_frame(
            [
                "Instruction-Aware Safe Motion",
                f"Controller: {policy} | mode: {mode}",
                f"Interpreter: {source}",
                text or "2-DOF end-effector motion, no grasping",
                f"Renderer: {renderer}; physics: MuJoCo",
                "Playback: real time (1x)",
            ]
        )
        for _ in range(round(25 * title_seconds)):
            writer.append_data(intro)
            count += 1
        try:
            for i, case in enumerate(cases):
                obs, info = env.reset(seed=case["seed"], options={"case": case})
                speeds, deltas = [], []
                prev = np.zeros(2)
                done = False
                frame = caption(
                    env.render(),
                    [
                        f"{policy.upper()} | {mode} | case {i + 1}/{len(cases)} | t=0.00s",
                        text or "Gold: goal / Red: obstacle",
                        f"distance={info['distance']:.3f}m | interpreter={source}",
                    ],
                )
                writer.append_data(frame)
                count += 1
                while not done:
                    if policy == "ik":
                        action = ik_action(env)
                    elif policy == "random":
                        action = rng.uniform(-1, 1, 2).astype(np.float32)
                    else:
                        action = model.predict(obs, deterministic=True)[0]
                    obs, _, term, trunc, info = env.step(action)
                    trajectory.append(
                        {
                            "case_id": case["id"],
                            "time_s": info["elapsed_time_s"],
                            "mode": mode,
                            "q1_rad": float(env.data.qpos[0]),
                            "q2_rad": float(env.data.qpos[1]),
                            "qvel1_rad_s": float(env.data.qvel[0]),
                            "qvel2_rad_s": float(env.data.qvel[1]),
                            "action1": float(action[0]),
                            "action2": float(action[1]),
                            "command1_rad_s": float(info["command"][0]),
                            "command2_rad_s": float(info["command"][1]),
                            "tip_x_m": float(env.tip[0]),
                            "tip_y_m": float(env.tip[1]),
                            "distance_m": info["distance"],
                            "collision": info["collision"],
                            "success": info["is_success"],
                        }
                    )
                    speeds.append(info["tip_speed_m_s"])
                    deltas.append(float(np.abs(info["command"] - prev).mean()))
                    prev = info["command"]
                    done = term or trunc
                    status = (
                        "SUCCESS"
                        if info["is_success"]
                        else "COLLISION"
                        if info["collision"]
                        else "INVALID"
                        if info["invalid_state"]
                        else "TIMEOUT"
                        if trunc
                        else "MOVING"
                    )
                    frame = caption(
                        env.render(),
                        [
                            f"{policy.upper()} | {mode} | case {i + 1}/{len(cases)} | t={info['elapsed_time_s']:.2f}s | {status}",
                            text or "Gold: goal / Red: obstacle",
                            f"distance={info['distance']:.3f}m | speed={speeds[-1]:.3f}m/s | interpreter={source}",
                        ],
                    )
                    writer.append_data(frame)
                    count += 1
                episode_log.append(
                    {
                        "case_id": case["id"],
                        "success": info["is_success"],
                        "collision": info["collision"],
                        "invalid_state": info["invalid_state"],
                        "time_s": info["elapsed_time_s"],
                        "min_goal_distance_m": info["min_goal_distance_m"],
                        "mean_tip_speed_m_s": float(np.mean(speeds)),
                        "mean_abs_command_delta": float(np.mean(deltas)),
                    }
                )
                end = caption(
                    frame,
                    [
                        f"Case {i + 1}: {status}",
                        f"distance={info['distance']:.3f}m | duration={info['elapsed_time_s']:.2f}s",
                        "Pause for explanation (simulation stopped)",
                    ],
                )
                for _ in range(25):
                    writer.append_data(end)
                    count += 1
            if count < minimum_seconds * 25:
                end = title_frame(
                    [
                        "Measured demo results",
                        f"Controller: {policy} | mode: {mode}",
                        f"Episodes: {len(episode_log)}",
                        f"Successes: {sum(x['success'] for x in episode_log)}",
                        f"Collisions: {sum(x['collision'] for x in episode_log)}",
                        "End card (simulation stopped)",
                    ]
                )
                for _ in range(int(minimum_seconds * 25 - count)):
                    writer.append_data(end)
        finally:
            env.close()
    result = verify_video(output)
    if trajectory:
        with output.with_suffix(".trajectory.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(trajectory[0]))
            writer.writeheader()
            writer.writerows(trajectory)
    result.update(
        {
            "policy": policy,
            "mode": mode,
            "renderer": renderer,
            "interpreter": source,
            "instruction": text,
            "episodes": episode_log,
            "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
            if policy == "ppo"
            else None,
        }
    )
    output.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def comparison(output, inputs, summary_path=None):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(output), fps=25, codec="libx264", quality=8, macro_block_size=1
    ) as writer:
        intro = title_frame(
            [
                "Instruction-Aware Safe Motion",
                "Factory-inspired 2-DOF end-effector motion",
                "PPO learns reaching; a rule filter sets speed limits.",
                "Natural language selects careful / fast.",
                "No grasping. No real-world safety guarantee.",
                "Clips: real time (1x); title cards pause simulation.",
            ]
        )
        for _ in range(25 * 6):
            writer.append_data(intro)
        curve_path = ROOT / "results/learning_curve.png"
        if curve_path.is_file():
            from PIL import Image, ImageOps

            canvas = Image.new("RGB", (960, 640), (12, 20, 34))
            with Image.open(curve_path) as raw:
                chart = ImageOps.contain(raw.convert("RGB"), (920, 470))
                canvas.paste(chart, ((960 - chart.width) // 2, 125))
            frame = caption(
                np.asarray(canvas),
                [
                    "Measured PPO validation history (10 cases)",
                    "Stage 1: fixed goal / Stage 2: random goal",
                    "Model selection uses validation cases, separate from heldout evaluation.",
                ],
                footer="Chart display (simulation stopped)",
            )
            for _ in range(25 * 8):
                writer.append_data(frame)
        for path in inputs:
            reader = imageio.get_reader(str(path))
            try:
                for frame in reader:
                    writer.append_data(frame)
            finally:
                reader.close()
        if summary_path and Path(summary_path).is_file():
            import pandas as pd

            summary = pd.read_csv(summary_path)
            lines = [
                "Measured paired evaluation",
                f"n={int(summary.episodes.iloc[0])} per controller; no obstacles",
            ]
            for row in summary.itertuples():
                if "llm" in row.controller:
                    continue
                lines.append(
                    f"{row.controller}: {row.success_rate:.0%} success | speed {row.mean_speed:.3f} m/s"
                )
            lines += [
                "IK outperformed PPO on this simple task.",
                "Obstacle avoidance has not been demonstrated.",
            ]
            frame = title_frame(lines)
            for _ in range(25 * 12):
                writer.append_data(frame)
    result = verify_video(output)
    result["inputs"] = [str(Path(v).name) for v in inputs]
    result["summary_source"] = None if summary_path is None else Path(summary_path).name
    output.with_suffix(".json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "videos/baseline.mp4")
    p.add_argument("--renderer", choices=["mujoco", "2d"], default="mujoco")
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--policy", choices=["ik", "ppo", "random"], default="ik")
    p.add_argument("--model", type=Path, default=ROOT / "checkpoints/best_model.zip")
    p.add_argument(
        "--mode", choices=["standard", "careful", "fast"], default="standard"
    )
    p.add_argument("--comparison", action="store_true")
    p.add_argument("--inputs", nargs="*", type=Path)
    p.add_argument("--summary", type=Path, default=ROOT / "results/summary.csv")
    args = p.parse_args()
    if args.episodes < 1:
        p.error("episodes must be positive")
    if args.comparison:
        comparison(
            args.output,
            args.inputs
            or [
                ROOT / "videos/baseline.mp4",
                ROOT / "videos/careful.mp4",
                ROOT / "videos/fast.mp4",
            ],
            args.summary,
        )
    else:
        record(
            args.output,
            args.policy,
            args.model,
            args.mode,
            args.renderer,
            generate_cases(args.episodes, task="random"),
        )


if __name__ == "__main__":
    main()
