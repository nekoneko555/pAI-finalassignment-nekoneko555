"""Append supplied OP/ED at real-time speed; normalize without cropping/stretching."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import imageio.v2 as imageio
import imageio_ffmpeg
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def metadata(path):
    reader = imageio.get_reader(str(path))
    try:
        info = reader.get_meta_data()
        count = reader.count_frames()
    finally:
        reader.close()
    probe = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", str(path)],
        capture_output=True,
        encoding="utf8",
        errors="replace",
    )
    return {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "frames": count,
        "fps": info["fps"],
        "duration_s": info["duration"],
        "size": list(info["size"]),
        "has_audio": "Audio:" in probe.stderr,
    }


def checked(command):
    result = subprocess.run(
        command, capture_output=True, encoding="utf8", errors="replace"
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opening", type=Path, default=ROOT / "videos/so101_op.mp4")
    parser.add_argument("--ending", type=Path, default=ROOT / "videos/so101_ed.mp4")
    parser.add_argument("--main", type=Path, default=ROOT / "videos/so101_final.mp4")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "videos/so101_submission.mp4"
    )
    args = parser.parse_args()
    sources = []
    for label, source in [
        ("op", args.opening),
        ("main", args.main),
        ("ed", args.ending),
    ]:
        if label != "main":
            destination = ROOT / f"videos/so101_{label}.mp4"
            if source.resolve() != destination.resolve():
                shutil.copyfile(source, destination)
            source = destination
        sources.append((label, source, metadata(source)))
    # All three supplied inputs have no audio. Do not silently discard an
    # unexpected future soundtrack if this script is used with different media.
    if any(info["has_audio"] for _, _, info in sources):
        raise ValueError(
            "An input contains audio; preserve its soundtrack with an audio-aware assembly."
        )
    temporary = ROOT / "tmp/video-bookends/normalized"
    temporary.mkdir(parents=True, exist_ok=True)
    records = []
    start_frame = 0
    for label, source, info in sources:
        normalized = temporary / f"{label}.mp4"
        checked(
            [
                FFMPEG,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-an",
                "-map_metadata",
                "-1",
                "-vf",
                "scale=1920:1080:force_original_aspect_ratio=decrease:force_divisible_by=2,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x0c1422,setsar=1,fps=30",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                "4",
                "-video_track_timescale",
                "30000",
                str(normalized),
            ]
        )
        rendered = metadata(normalized)
        records.append(
            {
                "segment": label,
                "input": info,
                "start_frame": start_frame,
                "start_s": start_frame / 30,
                "frames": rendered["frames"],
                "duration_s": rendered["frames"] / 30,
            }
        )
        start_frame += rendered["frames"]
        print(label, rendered["frames"], "frames", flush=True)
    listing = temporary / "concat.txt"
    listing.write_text(
        "".join(f"file '{label}.mp4'\n" for label, _, _ in sources), encoding="utf8"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checked(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c",
            "copy",
            "-map_metadata",
            "-1",
            "-movflags",
            "+faststart",
            str(args.output),
        ]
    )
    output = metadata(args.output)
    assert output["frames"] == start_frame and output["size"] == [1920, 1080]
    assert (
        abs(output["duration_s"] - sum(info["duration_s"] for _, _, info in sources))
        < 0.15
    )
    assert 60 <= output["duration_s"] <= 180
    checked(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-xerror",
            "-i",
            str(args.output),
            "-f",
            "null",
            "-",
        ]
    )
    reader = imageio.get_reader(str(args.output))
    total = output["frames"]
    op_end, main_end = records[1]["start_frame"], records[2]["start_frame"]
    frames = [
        15,
        op_end // 2,
        op_end - 1,
        op_end,
        op_end + 30,
        main_end - 1,
        main_end,
        main_end + 30,
        total - 1,
    ]
    sheet = Image.new("RGB", (1440, 870), (12, 20, 34))
    try:
        for i, index in enumerate(frames):
            frame = Image.fromarray(reader.get_data(index)).resize((480, 270))
            x, y = (i % 3) * 480, (i // 3) * 290
            sheet.paste(frame, (x, y + 20))
            ImageDraw.Draw(sheet).text(
                (x + 5, y + 3), f"{index / 30:.3f}s", fill="white"
            )
    finally:
        reader.close()
    sheet.save(ROOT / "results/so101/submission-contact-sheet.jpg")
    output.update(
        segments=records,
        playback_speed=1,
        full_decode_passed=True,
        fit="preserve aspect ratio; pillarbox main content",
        audio="none in all source files",
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
