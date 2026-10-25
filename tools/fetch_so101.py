"""Fetch a pinned upstream model and check each Git blob before storing it."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "cda5236f98383f6fbaa4820c6b18dba892cd087f"
REPO = "google-deepmind/mujoco_menagerie"
PREFIX = "robotstudio_so101/"


def main():
    dest = ROOT / "assets/so101/upstream"
    tree = json.load(
        urllib.request.urlopen(
            f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1",
            timeout=60,
        )
    )
    entries = [
        x for x in tree["tree"] if x["path"].startswith(PREFIX) and x["type"] == "blob"
    ]

    def fetch(entry):
        relative = entry["path"][len(PREFIX) :]
        path = dest / relative
        data = (
            path.read_bytes()
            if path.exists()
            else urllib.request.urlopen(
                f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{entry['path']}",
                timeout=120,
            ).read()
        )
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if blob != entry["sha"]:
            raise ValueError(f"Upstream blob mismatch: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {
            "file": relative,
            "git_blob": blob,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(fetch, entries))
    record = {
        "repository": f"https://github.com/{REPO}",
        "commit": COMMIT,
        "upstream_directory": PREFIX.rstrip("/"),
        "license": "Apache-2.0",
        "upstream_files_modified": False,
        "files": files,
    }
    (dest.parent / "source.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "commit": COMMIT,
                "files": len(files),
                "bytes": sum(x["bytes"] for x in files),
            }
        )
    )


if __name__ == "__main__":
    main()
