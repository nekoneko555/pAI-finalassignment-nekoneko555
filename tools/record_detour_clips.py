"""Record predetermined first cases; real LLM calls happen before robot rollout."""

from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.llm_parser import parse_instruction  # noqa: E402
from src.render_video import record  # noqa: E402
from src.scenarios import load_cases  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-interpretations",
        action="store_true",
        help="Rerender using the previously measured Ollama results",
    )
    args = parser.parse_args()
    requests = [
        ("careful", "精密部品だから慎重に移動して"),
        ("fast", "急いで素早く移動して"),
    ]
    interpretations = []
    if args.reuse_interpretations:
        interpretations = json.loads(
            (ROOT / "results/detour/demo_instructions.json").read_text(encoding="utf-8")
        )
        if [
            (r["requested_label"], r["text"]) for r in interpretations
        ] != requests or any(r["source"] != "ollama" for r in interpretations):
            raise ValueError("Expected the original measured Ollama interpretations")
    else:
        for expected, text in requests:
            result = parse_instruction(text, model="qwen2.5:1.5b")
            if result.source != "ollama":
                raise RuntimeError(
                    f"Real Ollama required for this recording: {result.error}"
                )
            interpretations.append(
                {"requested_label": expected, "text": text, **asdict(result)}
            )
    (ROOT / "results/detour/demo_instructions.json").write_text(
        json.dumps(interpretations, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cases = load_cases(ROOT / "data/detour_test.json")
    model = ROOT / "checkpoints/detour_seed42/best_model.zip"
    record(
        ROOT / "videos/detour_ik.mp4",
        policy="ik",
        cases=cases[:2],
        text="直接IK: 迂回なし",
        source="explicit_mode",
    )
    record(
        ROOT / "videos/detour_before.mp4",
        policy="ppo",
        model_path=ROOT / "checkpoints/best_model.zip",
        cases=cases[:2],
        text="前回PPO: 障害物なしで学習",
        source="explicit_mode",
    )
    record(
        ROOT / "videos/detour_standard.mp4",
        policy="ppo",
        model_path=model,
        cases=cases[:3],
        text="追加学習PPO: 肘を畳む迂回 / seed 42",
        source="explicit_mode",
    )
    for row in interpretations:
        record(
            ROOT / f"videos/detour_{row['requested_label']}.mp4",
            policy="ppo",
            model_path=model,
            mode=row["mode"],
            cases=cases[:3],
            text=row["text"],
            source=row["source"],
        )


if __name__ == "__main__":
    main()
