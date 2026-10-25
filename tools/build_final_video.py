"""Assemble a captioned final film from complete measured episodes and data cards."""

import json
from pathlib import Path
import sys
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rendering import title_frame, caption  # noqa: E402
from src.render_video import verify_video  # noqa: E402


def main():
    data = json.loads((ROOT / "results/final-summary.json").read_text(encoding="utf-8"))
    output = ROOT / "videos/final.mp4"
    segments = []
    frame_count = 0
    with imageio.get_writer(
        str(output), fps=25, codec="libx264", quality=8, macro_block_size=1
    ) as writer:

        def card(lines, seconds):
            nonlocal frame_count
            frame = title_frame(lines)
            for _ in range(round(seconds * 25)):
                writer.append_data(frame)
            frame_count += round(seconds * 25)
            segments.append(
                {
                    "type": "card",
                    "start_s": frame_count / 25 - seconds,
                    "duration_s": seconds,
                    "text": lines,
                }
            )

        def clip(name):
            nonlocal frame_count
            start = frame_count
            reader = imageio.get_reader(str(ROOT / f"videos/{name}.mp4"))
            try:
                for frame in reader:
                    writer.append_data(frame)
                    frame_count += 1
            finally:
                reader.close()
            segments.append(
                {
                    "type": "clip",
                    "file": f"{name}.mp4",
                    "start_s": start / 25,
                    "duration_s": (frame_count - start) / 25,
                }
            )

        def chart(path, seconds, lines):
            nonlocal frame_count
            base = Image.new("RGB", (960, 640), (12, 20, 34))
            with Image.open(ROOT / path) as raw:
                img = ImageOps.contain(raw.convert("RGB"), (920, 460))
                base.paste(img, ((960 - img.width) // 2, 135))
            frame = caption(
                np.asarray(base), lines, footer="Measured chart / simulation stopped"
            )
            for _ in range(round(seconds * 25)):
                writer.append_data(frame)
            segments.append(
                {
                    "type": "chart",
                    "file": path,
                    "start_s": frame_count / 25,
                    "duration_s": seconds,
                }
            )
            frame_count += round(seconds * 25)

        card(
            [
                "Instruction-Aware Safe Motion",
                "自然言語で動作条件を変えるロボットアーム",
                "工場の搬送を模した2関節の手先移動 / 把持は未実装",
                "MuJoCo + PPO + ローカルLLM + ルールによる指令制限",
                "動作映像は実時間1倍速 / 説明画面ではシミュレーション停止",
            ],
            8,
        )
        card(
            [
                "比較する課題",
                "障害物に向かう直接追従と、肘を畳む迂回を比較",
                "学習512配置 / 検証30配置 / 最終評価100配置",
                "3つの学習seedを独立初期化 / 最終評価前に重みを固定",
                "この映像は評価ケースの先頭を使用 / 全試行はCSVに保存",
            ],
            6,
        )
        clip("detour_ik")
        clip("detour_before")
        chart(
            "results/detour/learning_curve.png",
            10,
            [
                "実測: 検証30配置での学習経過 / seed 42, 123, 456",
                "既知の迂回経路上で開始姿勢を変えるカリキュラム",
                "経路はPPOの観測・推論時の制御には渡さない",
            ],
        )
        clip("detour_standard")
        card(
            [
                "日本語で実行条件を変更",
                "LLMは1指示につき1回だけcareful / fastを解釈",
                "PPOの同じ重みへ、ルールによる速度・指令変化量制限を適用",
                "速度指令上限: careful 0.525 / fast 1.200 rad/s",
                "指令加速度上限: careful 1.0 / fast 4.0 rad/s²",
                f"日本語分類: {data['llm_correct']}/24一致 / キーワード: {data['keyword_correct']}/24",
                "24件の手作りデータでの結果 / 一般的な性能保証ではない",
            ],
            6,
        )
        clip("detour_careful")
        clip("detour_fast")
        chart(
            "results/detour/comparison.png",
            12,
            [
                "実測: 最終評価100配置 / 同じ配置を各学習seedで使用",
                "PPOの誤差棒は学習seed 3つの標準偏差",
                "Oracleは経路を知る特別な参照制御 / 学習済み方策と区別",
            ],
        )
        card(
            [
                "分かった限界",
                "慎重モードの接触率は11.7% / 標準モードは8.3%",
                "速度を下げるだけでは接触を防げなかった",
                "観測・制御の応答が変わる影響は追加検証が必要",
            ],
            6,
        )
        card(
            [
                "限界と再現性",
                "平面2関節・障害物1個・固定の初期姿勢の課題",
                "任意配置への一般化・実機安全・把持は未検証",
                f"pytest {data['tests']}件通過 / CSV・重み・再現手順を同梱",
                "クラウドGPUは未使用 / 稼働0時間・料金0 USD",
            ],
            8,
        )
    result = verify_video(output)
    assert 60 <= result["duration_s"] <= 180, result
    result.update(
        segments=segments,
        playback_speed=1,
        source_cases="data/detour_test.json",
        demo_training_seed=42,
        demo_seed_selected_before_final_test=True,
    )
    output.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "segments"}, indent=2))


if __name__ == "__main__":
    main()
