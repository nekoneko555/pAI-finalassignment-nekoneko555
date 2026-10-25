"""Package the SO-101 edition without changing the previous planar delivery ZIP."""

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = []
    for name in [
        "README.md",
        "README-SO101.md",
        "report-so101.md",
        "config.yaml",
        "requirements.txt",
        "requirements-lock.txt",
        "requirements-pdf.txt",
        ".gitignore",
    ]:
        files.append(ROOT / name)
    for folder in ["src", "tests", "tools", "assets"]:
        files.extend(
            p
            for p in (ROOT / folder).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        )
    files.extend((ROOT / "data").glob("*.json"))
    for folder in ["results/so101/final"]:
        files.extend(p for p in (ROOT / folder).rglob("*") if p.is_file())
    for name in [
        "demo_instructions.json",
        "llm_runtime.json",
        "verification.json",
        "pytest.xml",
        "pdf-verification.json",
        "report-data.json",
        "final-contact-sheet.jpg",
        "submission-contact-sheet.jpg",
        "lifted-preview.jpg",
    ]:
        files.append(ROOT / "results/so101" / name)
    for seed in [42, 123, 456]:
        files.extend(
            p for p in (ROOT / f"checkpoints/so101_seed{seed}").glob("*") if p.is_file()
        )
    files.extend(p for p in (ROOT / "videos").glob("so101_*") if p.is_file())
    files += [ROOT / "output/pdf/so101_report.pdf", ROOT / "output/so101-plan.md"]
    target = ROOT / "output/physical-ai-so101-delivery.zip"
    with zipfile.ZipFile(
        target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as z:
        for p in sorted(set(files)):
            assert p.is_file(), p
            relative = p.relative_to(ROOT).as_posix()
            assert not any(
                s in p.relative_to(ROOT).parts
                for s in [".cache", ".venv", ".git", ".env", "tmp"]
            )
            if relative == "README.md":
                # The extracted edition opens with its own reproducible task.
                z.writestr(
                    "physical-ai-so101/README.md",
                    (ROOT / "README-SO101.md").read_bytes(),
                )
            else:
                z.write(p, "physical-ai-so101/" + relative)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        count = len(z.namelist())
        personal = []
        for name in z.namelist():
            if name.endswith(
                (".py", ".md", ".json", ".xml", ".csv", ".yaml", ".ps1", ".txt")
            ):
                text = z.read(name).decode("utf8", errors="replace")
                if any(
                    s in text
                    for s in [
                        Path.home().name,
                        str(Path.home()),
                        Path.home().as_posix(),
                    ]
                ):
                    personal.append(name)
        assert not personal, personal
    result = {
        "file": target.name,
        "bytes": target.stat().st_size,
        "entries": count,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "zip_crc_passed": True,
        "known_personal_strings_found": False,
    }
    (ROOT / "output/so101-delivery-manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
