"""Package source and verified outputs, excluding runtimes, caches and credentials."""

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = []
    for name in (
        "README.md",
        "report.md",
        "report-final.md",
        "requirements.txt",
        "requirements-pdf.txt",
        "requirements-lock.txt",
        "config.yaml",
        "heldout_eval_cases.json",
        "obstacle_eval_cases.json",
        ".gitignore",
    ):
        files.append(ROOT / name)
    for folder in ("src", "tests", "tools", "assets", "data", "results"):
        files.extend(
            p
            for p in (ROOT / folder).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        )
    files.extend(p for p in (ROOT / "videos").glob("*") if p.is_file())
    for name in (
        "best_model.zip",
        "final_model.zip",
        "training.json",
        "validation.json",
        "training_episodes.csv",
    ):
        p = ROOT / "checkpoints" / name
        if p.exists():
            files.append(p)
    for folder in ("fixed", "detour_seed42", "detour_seed123", "detour_seed456"):
        files.extend(
            p for p in (ROOT / "checkpoints" / folder).glob("*") if p.is_file()
        )
    files.append(ROOT / "output/pdf/physical_ai_report.pdf")
    target = ROOT / "output/physical-ai-delivery.zip"
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(
        target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as z:
        for path in sorted(set(files)):
            name = "physical-ai-instruction-arm/" + path.relative_to(ROOT).as_posix()
            assert not any(
                v in Path(name).parts
                for v in (".cache", ".venv", ".git", ".env", "tmp")
            )
            z.write(path, name)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert "physical-ai-instruction-arm/videos/final.mp4" in z.namelist()
        count = len(z.namelist())
    manifest = {
        "file": target.name,
        "bytes": target.stat().st_size,
        "entries": count,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "excluded": [
            "Python runtime",
            "Ollama runtime and weights",
            "credentials",
            "Git internals",
            "temporary files",
            "TensorBoard event files",
        ],
    }
    (ROOT / "output/delivery-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
