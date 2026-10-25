"""Optional local CPU-only extraction of the official portable Ollama release.

No system install, registry changes, or paid services. CLI invocation downloads
the official archive; only the executable and CPU runtime are extracted.
"""

import hashlib
import argparse
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v0.40.9")
    args = parser.parse_args()
    target = ROOT / ".cache/ollama"
    target.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(
        f"https://api.github.com/repos/ollama/ollama/releases/tags/{args.version}",
        timeout=30,
    ) as r:
        release = json.load(r)
    asset = next(
        a for a in release["assets"] if a["name"] == "ollama-windows-amd64.zip"
    )
    archive = target / "official.zip"
    if not archive.exists() or archive.stat().st_size != asset["size"]:
        print(
            f"Downloading {release['tag_name']}: {asset['size'] / 1e6:.1f} MB",
            flush=True,
        )
        with (
            urllib.request.urlopen(asset["browser_download_url"], timeout=60) as src,
            archive.open("wb") as dst,
        ):
            shutil.copyfileobj(src, dst, length=1024 * 1024)
    digest = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
    expected = asset.get("digest")
    if expected and expected != "sha256:" + digest:
        raise RuntimeError("Official asset checksum mismatch")
    extracted = []
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            name = entry.filename.replace("\\", "/")
            lowered = name.lower()
            if entry.is_dir() or any(
                x in lowered for x in ("cuda", "vulkan", "rocm", "mlx")
            ):
                continue
            if not (lowered == "ollama.exe" or lowered.startswith("lib/ollama/")):
                continue
            out = (target / name).resolve()
            if not out.is_relative_to(target.resolve()):
                raise RuntimeError("Invalid archive path")
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(entry) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(name)
    metadata = {
        "version": release["tag_name"],
        "source": asset["browser_download_url"],
        "archive_sha256": digest,
        "checksum_verified": bool(expected),
        "extracted": extracted,
    }
    (target / "install.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    # Delete just the verified archive file; keep the extracted runtime.
    archive.unlink()
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
