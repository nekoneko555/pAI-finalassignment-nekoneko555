"""Check saved evidence and create a compact artifact manifest."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import summarize  # noqa: E402
from src.render_video import verify_video  # noqa: E402


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main():
    tree = ET.parse(ROOT / "results/pytest.xml")
    suites = list(tree.getroot().iter("testsuite"))
    tests = sum(int(s.attrib.get("tests", 0)) for s in suites)
    errors = sum(
        int(s.attrib.get("errors", 0)) + int(s.attrib.get("failures", 0))
        for s in suites
    )
    assert tests >= 31 and errors == 0
    # Host names are unnecessary in a public evidence file.
    for s in suites:
        s.attrib.pop("hostname", None)
    tree.write(ROOT / "results/pytest.xml", encoding="utf-8", xml_declaration=True)
    df = pd.read_csv(ROOT / "results/episode_results.csv")
    summary = pd.read_csv(ROOT / "results/summary.csv")
    pd.testing.assert_frame_equal(summarize(df), summary, check_dtype=False, atol=1e-10)
    assert df.groupby("controller").case_id.nunique().eq(100).all()
    assert df.groupby("controller").size().eq(100).all()
    metadata = load("results/evaluation.json")
    assert (
        metadata["model_sha256"]
        == hashlib.sha256(
            (ROOT / "checkpoints/best_model.zip").read_bytes()
        ).hexdigest()
    )
    assert (
        metadata["cases_sha256"]
        == hashlib.sha256((ROOT / "heldout_eval_cases.json").read_bytes()).hexdigest()
    )
    assert metadata["llm_used"]
    assert all(r["source"] == "ollama" for r in metadata["llm_interpretations"])
    instructions = load("results/instruction_ollama.json")
    assert instructions["summary"]["actual_backend_responses"] == 24
    assert len(instructions["rows"]) == 24
    runtime = load("results/runtime.json")
    assert runtime["runtime_stopped_after_verification"]
    assert all(m["size_vram"] == 0 for m in runtime["llm_runtime"]["models"])
    videos = []
    for name in ["baseline", "random", "careful", "fast", "comparison"]:
        path = ROOT / "videos" / f"{name}.mp4"
        actual = verify_video(path)
        saved = load(f"videos/{name}.json")
        assert all(
            actual[k] == saved[k] for k in ("bytes", "frames", "fps", "duration_s")
        )
        videos.append(actual)
    assert 60 <= videos[-1]["duration_s"] <= 180
    assert load("videos/careful.json")["interpreter"] == "ollama"
    assert load("videos/fast.json")["interpreter"] == "ollama"
    reader = imageio.get_reader(str(ROOT / "videos/comparison.mp4"))
    sheet = Image.new("RGB", (1440, 680), (12, 20, 34))
    draw = ImageDraw.Draw(sheet)
    times = [0, 8, 22, 45, 83, videos[-1]["duration_s"] - 2]
    try:
        for i, second in enumerate(times):
            frame = Image.fromarray(reader.get_data(round(second * 25))).resize(
                (480, 320)
            )
            x, y = (i % 3) * 480, (i // 3) * 340
            sheet.paste(frame, (x, y + 20))
            draw.text((x + 8, y + 3), f"{second:.1f}s", fill="white")
    finally:
        reader.close()
    sheet.save(ROOT / "results/video-contact-sheet.jpg", quality=92)
    files = [
        "README.md",
        "report.md",
        "requirements-lock.txt",
        "assets/arm.xml",
        "checkpoints/best_model.zip",
        "checkpoints/final_model.zip",
        "heldout_eval_cases.json",
        "results/episode_results.csv",
        "results/summary.csv",
        "results/comparison.png",
        "results/learning_curve.png",
        "videos/comparison.mp4",
    ]
    artifacts = []
    for name in files:
        path = ROOT / name
        assert path.is_file() and path.stat().st_size > 0
        artifacts.append(
            {
                "file": name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "day1_complete": True,
        "pytest_tests": tests,
        "pytest_errors": errors,
        "evaluation_episodes_per_controller": 100,
        "evaluation_controllers": len(summary),
        "llm_real_responses": 24,
        "llm_model": instructions["summary"]["model"],
        "cloud_gpu_hours": 0,
        "cloud_gpu_cost_usd": 0,
        "videos": videos,
        "python_version": sys.version.split()[0],
        "packages": {
            p: importlib.metadata.version(p)
            for p in ["mujoco", "gymnasium", "stable-baselines3", "torch", "ollama"]
        },
        "artifacts": artifacts,
    }
    (ROOT / "results/verification.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "day1_complete": True,
                "pytest_tests": tests,
                "controllers": len(summary),
                "comparison_video_s": videos[-1]["duration_s"],
                "artifacts_verified": len(artifacts),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
