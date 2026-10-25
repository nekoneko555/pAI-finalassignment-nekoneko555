"""Record true local LLM -> residual PPO -> physical SO-101 pick and place."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from src.llm_parser import parse_instruction  # noqa: E402
from src.so101_task import SO101PickPlaceEnv, cases_for, PHASES  # noqa: E402
from src.render_video import verify_video  # noqa: E402
from src.rendering import caption  # noqa: E402

LABELS = {
    "APPROACH A": "A地点へ近づく",
    "DESCEND": "物体へ指を下ろす",
    "GRASP": "指を閉じてつかむ",
    "LIFT": "持ち上げる",
    "TRANSPORT TO B": "B地点へ運ぶ",
    "LOWER AT B": "B地点へ下ろす",
    "RELEASE": "指を開いて離す",
    "RETREAT": "手先を退避する",
}
TEXTS = {
    "careful": "精密部品なので慎重にA地点からB地点へ運んで",
    "fast": "急いでA地点からB地点へ運んで",
}


def record(label, mode, model, interpretation, case):
    out = ROOT / f"videos/so101_{label}.mp4"
    env = SO101PickPlaceEnv(mode=mode, render_mode="rgb_array")
    try:
        obs, _ = env.reset(seed=case["seed"], options={"case": case})
        with imageio.get_writer(
            str(out), fps=25, codec="libx264", quality=8, macro_block_size=1
        ) as writer:
            while True:
                phase = PHASES[env.phase]
                action = (
                    np.zeros(5)
                    if model is None
                    else model.predict(obs, deterministic=True)[0]
                )
                obs, _, term, trunc, info = env.step(action)
                text = "IK基準制御" if model is None else TEXTS[mode]
                frame = caption(
                    env.render(),
                    [
                        f"SO-101 | {LABELS[phase]} | {env.sim.elapsed:.2f}秒 | {mode}",
                        text,
                        f"物体高さ {info['object_height_m']:.3f}m / Bまで {info['final_xy_error_m']:.3f}m | A=橙 B=緑",
                    ],
                    footer="MuJoCo simulation | real time 1x | contact grasp; no attachment",
                )
                writer.append_data(frame)
                if term or trunc:
                    break
        result = {
            "video": verify_video(out),
            "case": case,
            "outcome": info,
            "interpreter": interpretation,
            "controller": "IK baseline"
            if model is None
            else "IK waypoints + PPO residual joint speeds + rule gripper",
            "model_sha256": None
            if model is None
            else hashlib.sha256(
                (ROOT / "checkpoints/so101_seed42/best_model.zip").read_bytes()
            ).hexdigest(),
        }
        out.with_suffix(".json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8"
        )
        pd.DataFrame(env.sim.trace).to_csv(
            out.with_suffix(".trajectory.csv"), index=False
        )
        print(label, info, flush=True)
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reuse-interpretations", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(1)
    source = ROOT / "results/so101/demo_instructions.json"
    if a.reuse_interpretations:
        interpretations = json.loads(source.read_text(encoding="utf8"))
    else:
        interpretations = {}
        for mode, text in TEXTS.items():
            parsed = parse_instruction(text, model="qwen2.5:1.5b")
            if parsed.source != "ollama" or parsed.mode != mode:
                raise RuntimeError(f"Actual LLM interpretation required: {parsed}")
            interpretations[mode] = {
                "text": text,
                "llm_model": "qwen2.5:1.5b",
                **asdict(parsed),
            }
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(
            json.dumps(interpretations, ensure_ascii=False, indent=2), encoding="utf8"
        )
    for mode, text in TEXTS.items():
        assert (
            interpretations[mode]["text"] == text
            and interpretations[mode]["source"] == "ollama"
            and interpretations[mode]["mode"] == mode
        )
    case = cases_for("test")[0]
    model = PPO.load(ROOT / "checkpoints/so101_seed42/best_model.zip", device="cpu")
    record("ik_reference", "standard", None, {"source": "explicit_mode"}, case)
    for mode in TEXTS:
        record(f"{mode}_ppo", mode, model, interpretations[mode], case)


if __name__ == "__main__":
    main()
