"""Check frozen experiment hashes, raw metrics, replay, actual grasps and media."""

import hashlib
import json
import ast
from io import StringIO
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from so101_experiment import episode  # noqa: E402
from src.so101_task import SO101PickPlaceEnv, cases_for  # noqa: E402


def main():
    torch.set_num_threads(1)
    folder = ROOT / "results/so101/final"
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf8"))
    for key in ["source_sha256", "data_sha256"]:
        for path, digest in protocol[key].items():
            current = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            if current != digest:
                # Preserve the original experimental freeze while allowing a
                # documented rendering-only update after the experiment.
                audit = json.loads(
                    (folder / "rendering-update.json").read_text(encoding="utf8")
                )
                assert path == audit["source_path"] == "src/so101_pickplace.py"
                assert (
                    digest == audit["original_sha256"]
                    and current == audit["updated_sha256"]
                )
                original = audit["original_source"]
                assert digest in {
                    hashlib.sha256(original.encode("utf8")).hexdigest(),
                    hashlib.sha256(
                        original.replace("\n", "\r\n").encode("utf8")
                    ).hexdigest(),
                }

                def non_render_ast(source):
                    tree = ast.parse(source)
                    for node in tree.body:
                        if (
                            isinstance(node, ast.ClassDef)
                            and node.name == "PickPlaceSimulation"
                        ):
                            node.body = [
                                n
                                for n in node.body
                                if not isinstance(n, ast.FunctionDef)
                                or n.name != "render"
                            ]
                    return ast.dump(tree, include_attributes=False)

                assert non_render_ast(original) == non_render_ast(
                    (ROOT / path).read_text(encoding="utf8")
                )
    for m in protocol["models"]:
        assert (
            hashlib.sha256((ROOT / m["file"]).read_bytes()).hexdigest() == m["sha256"]
        )
    upstream = json.loads(
        (ROOT / "assets/so101/source.json").read_text(encoding="utf8")
    )
    for entry in upstream["files"]:
        assert (
            hashlib.sha256(
                (ROOT / "assets/so101/upstream" / entry["file"]).read_bytes()
            ).hexdigest()
            == entry["sha256"]
        )
    splits = [
        {c["seed"] for c in cases_for(s)} for s in ["train", "validation", "test"]
    ]
    assert all(
        splits[i].isdisjoint(splits[j]) for i in range(3) for j in range(i + 1, 3)
    )
    frame = pd.read_csv(folder / "episode_results.csv")
    summary = pd.read_csv(folder / "summary.csv").set_index("controller")
    assert len(frame) == 1300 and frame.controller.nunique() == 13
    assert not frame.duplicated(["controller", "case_id"]).any()
    assert all(frame.groupby("controller").size() == 100)
    calculated = (
        (frame.completed_phases == 8)
        & frame.lifted
        & frame.physical_grasp
        & frame.released
        & frame.object_settled
        & (~frame.invalid_state)
        & (frame.final_xy_error_m < 0.035)
        & frame.object_height_m.between(0.01, 0.035, inclusive="neither")
    )
    np.testing.assert_array_equal(calculated, frame.is_success)
    for label, group in frame.groupby("controller"):
        assert abs(group.is_success.mean() - summary.loc[label, "success_rate"]) < 1e-12
        assert (
            abs(group.elapsed_time_s.mean() - summary.loc[label, "mean_time_s"]) < 1e-12
        )
    replay = []
    for seed in [42, 123, 456]:
        model = PPO.load(
            ROOT / f"checkpoints/so101_seed{seed}/best_model.zip", device="cpu"
        )
        e = SO101PickPlaceEnv()
        try:
            r = episode(e, cases_for("test")[0], model)
        finally:
            e.close()
        saved = frame[
            (frame.controller == f"ppo_seed{seed}_standard")
            & (frame.case_id == r["case_id"])
        ].iloc[0]
        assert bool(saved.is_success) == r["is_success"]
        assert abs(saved.elapsed_time_s - r["elapsed_time_s"]) < 1e-12
        assert abs(saved.final_xy_error_m - r["final_xy_error_m"]) < 1e-10
        replay.append({"seed": seed, "case": r["case_id"], "equal": True})
    modes = []
    for mode, limit, acc in [("careful", 0.4, 0.8), ("fast", 1.2, 3.0)]:
        meta = json.loads(
            (ROOT / f"videos/so101_{mode}_ppo.json").read_text(encoding="utf8")
        )
        assert (
            meta["interpreter"]["source"] == "ollama"
            and meta["interpreter"]["mode"] == mode
        )
        assert meta["model_sha256"] == protocol["models"][0]["sha256"]
        trace = pd.read_csv(ROOT / f"videos/so101_{mode}_ppo.trajectory.csv")
        cmd = trace[[f"velocity_command{i}" for i in range(5)]].to_numpy()
        assert np.abs(cmd).max() <= limit + 1e-10
        assert (
            np.abs(np.diff(np.vstack([np.zeros(5), cmd]), axis=0)).max()
            <= acc * 0.04 + 1e-10
        )
        assert (
            (trace.object_z > 0.06) & trace.fixed_jaw_contact & trace.moving_jaw_contact
        ).any()
        assert (
            meta["outcome"]["is_success"] and meta["outcome"]["weld_constraints"] == 0
        )
        saved = frame[
            (frame.controller == f"ppo_seed42_{mode}")
            & (frame.case_id == meta["case"]["id"])
        ].iloc[0]
        assert abs(saved.elapsed_time_s - meta["outcome"]["elapsed_time_s"]) < 1e-12
        modes.append(
            {
                "mode": mode,
                "source": "ollama",
                "command_limits": True,
                "physical_grasp": True,
                "success": True,
            }
        )
    if (folder / "rendering-update.json").exists():
        audit = json.loads(
            (folder / "rendering-update.json").read_text(encoding="utf8")
        )
        for name, previous in audit["previous_demos"].items():
            meta = json.loads((ROOT / f"videos/{name}.json").read_text(encoding="utf8"))
            old = previous["metadata"]
            if "outcome" in old:
                assert meta["outcome"] == old["outcome"]
                assert meta["interpreter"] == old["interpreter"]
            else:
                assert {k: v for k, v in meta.items() if k != "video"} == {
                    k: v for k, v in old.items() if k != "video"
                }
            trace_path = ROOT / f"videos/{name}.trajectory.csv"
            if (
                hashlib.sha256(trace_path.read_bytes()).hexdigest()
                != previous["trajectory_sha256"]
            ):
                assert name == "so101_pickplace_ik"
                original_csv = previous["original_trajectory_csv"]
                assert (
                    hashlib.sha256(original_csv.encode("utf8")).hexdigest()
                    == previous["trajectory_sha256"]
                )
                original_frame = pd.read_csv(StringIO(original_csv))
                current_frame = pd.read_csv(trace_path)
                pd.testing.assert_frame_equal(
                    original_frame,
                    current_frame[original_frame.columns],
                    check_exact=True,
                )
                assert (
                    not current_frame[[f"ppo_residual{i}" for i in range(5)]]
                    .to_numpy()
                    .any()
                )
    xmlpath = ROOT / "results/so101/pytest.xml"
    xml = ET.parse(xmlpath)
    for suite in xml.getroot().iter("testsuite"):
        suite.attrib.pop("hostname", None)
        assert (
            int(suite.attrib.get("failures", 0)) == 0
            and int(suite.attrib.get("errors", 0)) == 0
        )
    tests = sum(int(s.attrib["tests"]) for s in xml.getroot().iter("testsuite"))
    assert tests == 42
    xml.write(xmlpath, encoding="utf-8", xml_declaration=True)
    pdf = json.loads(
        (ROOT / "results/so101/pdf-verification.json").read_text(encoding="utf8")
    )
    assert pdf["pages"] == 3 and pdf["visually_reviewed_pages"] == [1, 2, 3]
    assert (
        pdf["sha256"]
        == hashlib.sha256(
            (ROOT / "output/pdf/so101_report.pdf").read_bytes()
        ).hexdigest()
    )
    runtime = json.loads(
        (ROOT / "results/so101/llm_runtime.json").read_text(encoding="utf8")
    )
    assert runtime["server_stopped_after_recording"] and runtime["real_inferences"] == 2
    assert all(m["size_vram"] == 0 for m in runtime["running_models"]["models"])
    video = json.loads((ROOT / "videos/so101_final.json").read_text(encoding="utf8"))
    assert 60 <= video["duration_s"] <= 180 and video["playback_speed"] == 1
    reader = imageio.get_reader(str(ROOT / "videos/so101_final.mp4"))
    positions = np.linspace(2, video["duration_s"] - 2, 9)
    sheet = Image.new("RGB", (1440, 1020), (12, 20, 34))
    for i, pos in enumerate(positions):
        im = Image.fromarray(reader.get_data(round(pos * 25))).resize((480, 320))
        sheet.paste(im, ((i % 3) * 480, (i // 3) * 340 + 20))
        ImageDraw.Draw(sheet).text(
            ((i % 3) * 480 + 4, (i // 3) * 340 + 4), f"{pos:.1f}s", fill="white"
        )
    reader.close()
    sheet.save(ROOT / "results/so101/final-contact-sheet.jpg")
    important = [
        "README-SO101.md",
        "report-so101.md",
        "videos/so101_final.mp4",
        "output/pdf/so101_report.pdf",
        "results/so101/final/episode_results.csv",
        "results/so101/final/protocol.json",
    ]
    submission = None
    if (ROOT / "videos/so101_submission.json").exists():
        submission = json.loads(
            (ROOT / "videos/so101_submission.json").read_text(encoding="utf8")
        )
        assert submission["size"] == [1920, 1080] and submission["fps"] == 30
        assert (
            submission["full_decode_passed"] and 60 <= submission["duration_s"] <= 180
        )
        assert (
            submission["sha256"]
            == hashlib.sha256(
                (ROOT / "videos/so101_submission.mp4").read_bytes()
            ).hexdigest()
        )
        assert [r["segment"] for r in submission["segments"]] == ["op", "main", "ed"]
        assert sum(r["frames"] for r in submission["segments"]) == submission["frames"]
        for segment in submission["segments"]:
            source = ROOT / "videos" / segment["input"]["file"]
            assert (
                hashlib.sha256(source.read_bytes()).hexdigest()
                == segment["input"]["sha256"]
            )
        important.extend(
            ["videos/so101_submission.mp4", "videos/so101_submission.json"]
        )
    result = {
        "tests": tests,
        "episodes": len(frame),
        "controllers": 13,
        "cases_per_controller": 100,
        "independent_seeds": 3,
        "train_validation_test_disjoint": True,
        "upstream_files_verified": len(upstream["files"]),
        "replays": replay,
        "real_llm_demos": modes,
        "pdf": pdf,
        "llm_runtime_stopped": True,
        "rendering_update": "debug sites hidden; original trajectory values verified unchanged",
        "video": {k: v for k, v in video.items() if k != "segments"},
        "submission_video": submission,
        "cloud_gpu_hours": 0,
        "cloud_gpu_cost_usd": 0,
        "artifacts": [
            {
                "path": p,
                "sha256": hashlib.sha256((ROOT / p).read_bytes()).hexdigest(),
                "bytes": (ROOT / p).stat().st_size,
            }
            for p in important
        ],
    }
    (ROOT / "results/so101/verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
