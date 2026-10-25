"""Assemble a 1-3 minute film from complete measured physical pick/place clips."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from src.rendering import caption, title_frame  # noqa: E402
from src.render_video import verify_video  # noqa: E402


def main():
    out = ROOT / "videos/so101_final.mp4"
    data = json.loads(
        (ROOT / "results/so101/report-data.json").read_text(encoding="utf8")
    )
    agg = pd.DataFrame(data["aggregate"]).set_index("mode")
    summary = pd.DataFrame(data["summary"]).set_index("controller")
    segments = []
    count = 0
    with imageio.get_writer(
        str(out), fps=25, codec="libx264", quality=8, macro_block_size=1
    ) as writer:

        def still(frame, seconds, kind, text):
            nonlocal count
            segments.append(
                {
                    "type": kind,
                    "start_s": count / 25,
                    "duration_s": seconds,
                    "text": text,
                }
            )
            for _ in range(round(seconds * 25)):
                writer.append_data(frame)
            count += round(seconds * 25)

        def card(lines, seconds=6):
            still(title_frame(lines), seconds, "card", lines)

        def clip(name):
            nonlocal count
            start = count
            reader = imageio.get_reader(str(ROOT / f"videos/{name}.mp4"))
            try:
                for frame in reader:
                    writer.append_data(frame)
                    count += 1
            finally:
                reader.close()
            segments.append(
                {
                    "type": "clip",
                    "file": name + ".mp4",
                    "start_s": start / 25,
                    "duration_s": (count - start) / 25,
                }
            )

        def chart(path, lines, seconds=9):
            canvas = Image.new("RGB", (960, 640), (12, 20, 34))
            with Image.open(ROOT / path) as raw:
                im = ImageOps.contain(raw.convert("RGB"), (930, 450))
                canvas.paste(im, ((960 - im.width) // 2, 135))
            still(
                caption(
                    np.asarray(canvas),
                    lines,
                    footer="Measured chart | simulation stopped",
                ),
                seconds,
                "chart",
                lines,
            )

        card(
            [
                "日本語で動作条件を変えるSO-101",
                "A地点でつかむ → 持ち上げる → B地点へ運ぶ → 離す",
                "実在するアームのMuJoCoモデル / 実機の映像ではない",
                "物体は自由剛体 / 接触と摩擦で保持 / 固定拘束なし",
                "動作映像は1倍速 / 説明カードではシミュレーション停止",
            ],
            8,
        )
        card(
            [
                "制御の役割分担",
                "IK: 経由姿勢を計算 / ルール: 作業の順番と指の開閉",
                "PPO: 5関節の追従速度を最大±35%補正",
                "LLM: 日本語からcareful / fastを選択",
                "把持全体をゼロから学習した方式ではない",
            ],
            7,
        )
        card(
            [
                "まず学習なしの基準動作",
                "A=橙の領域 / B=緑の領域 / 赤い立方体を運搬",
                "両指で持ち上げ、Bで離して静止するまで確認",
            ],
            5,
        )
        clip("so101_ik_reference")
        chart(
            "results/so101/final/learning_curve.png",
            [
                "実測: PPO学習中の検証30配置 / 独立した3学習seed",
                "成功率が最大、同率なら時間が短いモデルを選択",
            ],
            8,
        )
        card(
            [
                "同じ物体・配置・PPO重みで日本語だけを変更",
                "精密部品なので慎重にA地点からB地点へ運んで",
                "ローカルLLMの実推論 → careful",
                "速度指令上限0.4rad/s / 指令加速度上限0.8rad/s²",
            ],
            6,
        )
        clip("so101_careful_ppo")
        card(
            [
                "次は速さを優先する指示",
                "急いでA地点からB地点へ運んで",
                "ローカルLLMの実推論 → fast",
                "速度指令上限1.2rad/s / 指令加速度上限3.0rad/s²",
            ],
            6,
        )
        clip("so101_fast_ppo")
        chart(
            "results/so101/final/comparison.png",
            [
                "実測: 未使用の100配置 / 同じ配置を全方式へ使用",
                "PPOは3学習seedの平均 / 動画の成功例だけで集計しない",
            ],
            10,
        )
        card(
            [
                "結果をどう読むか",
                f"IK基準の成功率: {summary.loc['ik', 'success_rate']:.1%}",
                f"PPO標準の成功率: {agg.loc['standard', 'mean_success_rate']:.1%}",
                f"平均時間: IK {summary.loc['ik', 'mean_time_s']:.2f}秒 / PPO {agg.loc['standard', 'mean_time_s']:.2f}秒",
                "PPOが把持を初めて可能にした、とは主張しない",
            ],
            7,
        )
        card(
            [
                "確認できた範囲と限界",
                "既知の立方体・狭いA/B範囲・既知の座標での検証",
                "画像認識・任意物体・実機転移は未実装",
                "慎重モードという名前は安全保証を意味しない",
                "全試行CSV・重み・テスト・再現手順を同梱 / クラウドGPU未使用",
            ],
            8,
        )
    meta = verify_video(out)
    assert 60 <= meta["duration_s"] <= 180, meta
    meta.update(
        segments=segments,
        playback_speed=1,
        scene="SO-101 physical pick/place",
        demo_case="data/so101_test.json first case",
        demo_training_seed=42,
    )
    out.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(json.dumps({k: v for k, v in meta.items() if k != "segments"}, indent=2))


if __name__ == "__main__":
    main()
