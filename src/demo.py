import argparse
from dataclasses import asdict
import json
from pathlib import Path
from .config import ROOT
from .llm_parser import ParseResult, keyword_parse, parse_instruction
from .render_video import record
from .scenarios import generate_cases, load_cases


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="")
    p.add_argument("--mode", choices=["careful", "fast"])
    p.add_argument("--parser", choices=["ollama", "keyword"], default="ollama")
    p.add_argument("--llm-model", default="qwen2.5:3b")
    p.add_argument("--model", type=Path, default=ROOT / "checkpoints/best_model.zip")
    p.add_argument("--policy", choices=["ppo", "ik"], default="ppo")
    p.add_argument("--renderer", choices=["mujoco", "2d"], default="mujoco")
    p.add_argument("--output", type=Path, default=ROOT / "videos/demo.mp4")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--cases", type=Path)
    a = p.parse_args()
    if not a.mode and not a.text:
        p.error("Provide --text or --mode")
    if a.episodes < 1:
        p.error("episodes must be positive")
    cases = (
        load_cases(a.cases)
        if a.cases
        else generate_cases(a.episodes, seed=20261023, task="random")
    )
    if a.episodes > len(cases):
        p.error("Not enough cases in the selected file")
    result = (
        ParseResult(a.mode, "明示的なモード選択", "explicit_mode")
        if a.mode
        else keyword_parse(a.text)
        if a.parser == "keyword"
        else parse_instruction(a.text, a.llm_model)
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    record(
        a.output,
        policy=a.policy,
        model_path=a.model,
        mode=result.mode,
        renderer=a.renderer,
        cases=cases[: a.episodes],
        text=a.text,
        source=result.source,
    )
    a.output.with_suffix(".instruction.json").write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
