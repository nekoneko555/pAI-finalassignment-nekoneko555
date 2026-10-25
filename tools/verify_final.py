"""Verify the latest frozen benchmark, model reloads, PDF and final video."""

import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.benchmark_detour import aggregate_seeds, evaluate_case  # noqa: E402
from src.detour import DetourArmEnv  # noqa: E402
from src.evaluate import summarize  # noqa: E402
from src.render_video import verify_video  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    tree = ET.parse(ROOT / "results/pytest.xml")
    suites = list(tree.getroot().iter("testsuite"))
    tests = sum(int(s.attrib.get("tests", 0)) for s in suites)
    assert tests >= 37
    assert (
        sum(
            int(s.attrib.get("failures", 0)) + int(s.attrib.get("errors", 0))
            for s in suites
        )
        == 0
    )
    for s in suites:
        s.attrib.pop("hostname", None)
    tree.write(ROOT / "results/pytest.xml", encoding="utf-8", xml_declaration=True)
    folder = ROOT / "results/detour"
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    cases = json.loads((ROOT / "data/detour_test.json").read_text(encoding="utf-8"))
    assert sha(ROOT / "data/detour_test.json") == protocol["cases_sha256"]
    assert sha(ROOT / "data/detour_train.json") == protocol["train_cases_sha256"]
    assert (
        sha(ROOT / "data/detour_validation.json") == protocol["validation_cases_sha256"]
    )
    splits = [
        json.loads((ROOT / f"data/detour_{name}.json").read_text(encoding="utf-8"))
        for name in ("train", "validation", "test")
    ]
    seed_sets = [{r["seed"] for r in split} for split in splits]
    assert all(not seed_sets[a] & seed_sets[b] for a, b in [(0, 1), (0, 2), (1, 2)])
    frame = pd.read_csv(folder / "episode_results.csv")
    assert (
        len(frame) == 1300
        and frame.groupby("controller").case_id.nunique().eq(100).all()
    )
    pd.testing.assert_frame_equal(
        summarize(frame),
        pd.read_csv(folder / "summary.csv"),
        check_dtype=False,
        atol=1e-10,
    )
    seed_summary = pd.read_csv(folder / "seed_summary.csv")
    pd.testing.assert_frame_equal(
        aggregate_seeds(seed_summary),
        pd.read_csv(folder / "seed_aggregate.csv"),
        check_dtype=False,
        atol=1e-10,
    )
    replays = []
    for entry in protocol["models"]:
        path = ROOT / entry["path"]
        assert sha(path) == entry["sha256"]
        training = json.loads(
            (path.parent / "training.json").read_text(encoding="utf-8")
        )
        assert training["resume"] is None and training["seed"] == entry["seed"]
        model = PPO.load(path, device="cpu")
        env = DetourArmEnv()
        row = evaluate_case(env, cases[0], "ppo", model)
        saved = frame[
            (frame.controller == f"ppo_seed{entry['seed']}_standard")
            & (frame.case_id == cases[0]["id"])
        ].iloc[0]
        assert row["success"] == saved.success and row["collision"] == saved.collision
        assert np.isclose(
            row["min_goal_distance_m"], saved.min_goal_distance_m, atol=1e-10
        )
        env.close()
        replays.append(
            {
                "seed": entry["seed"],
                "case_id": cases[0]["id"],
                "reloaded_episode_equal": True,
            }
        )
    movie = verify_video(ROOT / "videos/final.mp4")
    saved = json.loads((ROOT / "videos/final.json").read_text(encoding="utf-8"))
    assert movie["frames"] == saved["frames"] and 60 <= movie["duration_s"] <= 180
    for mode in ("careful", "fast"):
        clip = json.loads(
            (ROOT / f"videos/detour_{mode}.json").read_text(encoding="utf-8")
        )
        assert (
            clip["interpreter"] == "ollama"
            and clip["model_sha256"] == protocol["models"][0]["sha256"]
        )
        trace = pd.read_csv(ROOT / f"videos/detour_{mode}.trajectory.csv")
        speed, delta = (0.525, 0.04) if mode == "careful" else (1.2, 0.16)
        for _, group in trace.groupby("case_id", sort=False):
            commands = group[["command1_rad_s", "command2_rad_s"]].to_numpy()
            assert np.max(np.abs(commands)) <= speed + 1e-7
            changes = np.diff(np.vstack([np.zeros((1, 2)), commands]), axis=0)
            assert np.max(np.abs(changes)) <= delta + 1e-7
    pdf = json.loads(
        (ROOT / "results/pdf-verification.json").read_text(encoding="utf-8")
    )
    assert pdf["pages"] == 3 and pdf["visually_reviewed_pages"] == [1, 2, 3]
    assert pdf["sha256"] == sha(ROOT / "output/pdf/physical_ai_report.pdf")
    demo_runtime = json.loads(
        (ROOT / "results/detour/demo_runtime.json").read_text(encoding="utf-8")
    )
    assert demo_runtime["ollama_stopped_after_recording"]
    reader = imageio.get_reader(str(ROOT / "videos/final.mp4"))
    sheet = Image.new("RGB", (1440, 680), (12, 20, 34))
    draw = ImageDraw.Draw(sheet)
    times = np.linspace(2, movie["duration_s"] - 2, 6)
    try:
        for i, t in enumerate(times):
            x, y = (i % 3) * 480, (i // 3) * 340
            img = Image.fromarray(reader.get_data(round(t * 25))).resize((480, 320))
            sheet.paste(img, (x, y + 20))
            draw.text((x + 8, y + 3), f"{t:.1f}s", fill="white")
    finally:
        reader.close()
    sheet.save(ROOT / "results/final-contact-sheet.jpg", quality=93)
    artifacts = [
        "README.md",
        "report-final.md",
        "requirements-lock.txt",
        "requirements-pdf.txt",
        "assets/arm.xml",
        "config.yaml",
        "output/pdf/physical_ai_report.pdf",
        "results/detour/episode_results.csv",
        "results/detour/summary.csv",
        "results/detour/seed_aggregate.csv",
        "results/detour/protocol.json",
        "videos/final.mp4",
    ] + [v["path"] for v in protocol["models"]]
    result = {
        "tests_passed": tests,
        "benchmark_episodes": len(frame),
        "cases_per_model": 100,
        "independent_training_seeds": 3,
        "model_replays": replays,
        "video": movie,
        "train_validation_test_disjoint": True,
        "cloud_gpu_hours": 0,
        "cloud_gpu_cost_usd": 0,
        "pdf_verification": pdf,
        "mode_command_bounds_verified_from_trajectories": True,
        "artifacts": [
            {"path": p, "bytes": (ROOT / p).stat().st_size, "sha256": sha(ROOT / p)}
            for p in artifacts
        ],
        "source_sha256": {
            p.relative_to(ROOT).as_posix(): sha(p)
            for directory in ("src", "tests", "tools")
            for p in (ROOT / directory).glob("*.py")
        },
    }
    (ROOT / "results/final-verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "tests_passed": tests,
                "benchmark_episodes": len(frame),
                "models_replayed": len(replays),
                "video_seconds": movie["duration_s"],
                "artifacts_verified": len(artifacts),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
