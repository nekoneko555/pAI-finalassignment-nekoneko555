"""Three-page submission report from the frozen measured experiment summary.

Run with the bundled document Python runtime (reportlab + pypdf installed).
No identity, local absolute paths, or model weights are embedded in the report.
"""

import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab import rl_config
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon

ROOT = Path(__file__).resolve().parents[1]
NAVY = colors.HexColor("#14283f")
TEAL = colors.HexColor("#087e8b")
LIGHT = colors.HexColor("#eff5f8")


def main():
    # Stable metadata allows a byte-for-byte rebuild from unchanged measurements.
    rl_config.invariant = 1
    data = json.loads((ROOT / "results/final-summary.json").read_text(encoding="utf-8"))
    font_path = Path("C:/Windows/Fonts/meiryo.ttc")
    if font_path.is_file():
        pdfmetrics.registerFont(TTFont("JP", str(font_path)))
    else:
        pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))
    font = "JP" if font_path.is_file() else "HeiseiKakuGo-W5"
    styles = {
        "title": ParagraphStyle(
            "title",
            fontName=font,
            fontSize=22,
            leading=29,
            textColor=NAVY,
            spaceAfter=12,
        ),
        "h": ParagraphStyle(
            "h",
            fontName=font,
            fontSize=13,
            leading=19,
            textColor=TEAL,
            spaceBefore=10,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            fontName=font,
            fontSize=9.1,
            leading=14,
            spaceAfter=6,
            wordWrap="CJK",
        ),
        "small": ParagraphStyle(
            "small",
            fontName=font,
            fontSize=7.7,
            leading=11,
            spaceAfter=5,
            wordWrap="CJK",
        ),
        "code": ParagraphStyle(
            "code", fontName="Courier", fontSize=7.5, leading=11, spaceAfter=4
        ),
    }

    def p(text, kind="body"):
        return Paragraph(text, styles[kind])

    def table(rows, widths):
        cells = [
            [
                p(
                    f'<font color="#ffffff">{escape(str(c))}</font>'
                    if i == 0
                    else escape(str(c)),
                    "small",
                )
                for c in row
            ]
            for i, row in enumerate(rows)
        ]
        t = Table(cells, colWidths=widths, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, colors.white]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.8, TEAL),
                ]
            )
        )
        return t

    def heading(text):
        return p(text, "h")

    def picture(path, width):
        from PIL import Image as PILImage

        with PILImage.open(ROOT / path) as im:
            w, h = im.size
        return Image(str(ROOT / path), width=width, height=width * h / w)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(TEAL)
        canvas.line(36, 31, A4[0] - 36, 31)
        canvas.setFont(font, 7)
        canvas.setFillColor(NAVY)
        canvas.drawString(36, 20, "Instruction-Aware Safe Motion | 2026-10-23")
        canvas.drawRightString(A4[0] - 36, 20, f"{doc.page} / 3")
        canvas.restoreState()

    story = []
    story += [
        p("Instruction-Aware<br/>Safe Motion", "title"),
        p("自然言語で動作条件を変更する強化学習ロボットアーム", "h"),
        p("Physical AI応用講座 最終課題 / 開発期間: 2026-10-23 - 2026-10-26", "small"),
        heading("1. 目的と独自の工夫"),
        p(
            "工場内の搬送を模した水平2関節アームの手先移動を扱う。日本語の指示をLLMがcareful / fastへ分類し、PPOが出す角速度指令にモード別の制限を適用する。物体の把持は実装していない。"
        ),
        p(
            "独自に作成したMuJoCo環境と、指示解釈・学習方策・指令制限を接続した。さらに、直接の目標追従では接触するが、肘を畳む経路なら通れる課題を作成し、学習前後を同一条件で測定した。"
        ),
    ]
    d = Drawing(520, 126)
    boxes = [
        (0, 78, "日本語指示"),
        (135, 78, "Ollama / JSON"),
        (270, 78, "careful / fast"),
        (0, 10, "MuJoCo / 観測"),
        (135, 10, "PPO / 角速度"),
        (270, 10, "指令制限"),
        (405, 10, "MuJoCoへ"),
    ]
    for x, y, label in boxes:
        w = 112 if x < 405 else 105
        d.add(Rect(x, y, w, 35, rx=4, fillColor=LIGHT, strokeColor=TEAL))
        d.add(
            String(
                x + w / 2,
                y + 14,
                label,
                fontName=font,
                fontSize=8.5,
                textAnchor="middle",
                fillColor=NAVY,
            )
        )
    for x, y in [(112, 95), (247, 95), (112, 27), (247, 27), (382, 27)]:
        d.add(Line(x, y, x + 20, y, strokeColor=TEAL))
        d.add(
            Polygon(
                [x + 20, y, x + 15, y + 3, x + 15, y - 3],
                fillColor=TEAL,
                strokeColor=TEAL,
            )
        )
    d.add(Line(326, 78, 326, 45, strokeColor=TEAL))
    d.add(Polygon([326, 45, 323, 50, 329, 50], fillColor=TEAL, strokeColor=TEAL))
    story += [
        d,
        p(
            "LLMは指示ごとに1回だけ呼ぶ。PPOは移動を学習し、速度・指令変化量の制限は非学習のルールで行う。迂回経路は可解性確認と学習時の開始姿勢にのみ用い、推論時には方策へ渡さない。",
            "small",
        ),
        heading("2. ロボットと学習設定"),
    ]
    story += [
        table(
            [
                ["項目", "設定"],
                [
                    "物理モデル",
                    "水平2関節、リンク0.35m / 0.30m、MuJoCo速度アクチュエータ",
                ],
                ["観測 / 行動", "14次元状態 / 2関節の正規化目標角速度"],
                ["刻み / 成功条件", "物理0.01秒×4 / 8秒以内に距離0.05m未満・接触なし"],
                ["PPO", "MLP [64,64]、lr 3e-4、gamma 0.99、n_steps 1024、batch 64"],
                ["追加学習", "独立初期化 seed 42 / 123 / 456、各303,104ステップ、CPU"],
                ["報酬", "2×距離改善 - 0.01 - 0.3×距離、成功+10、接触-10、異常-5"],
                [
                    "モード制限",
                    "careful: 0.525 rad/s・指令加速度1.0 rad/s² / fast: 1.200・4.0",
                ],
            ],
            [95, 425],
        ),
        Spacer(1, 7),
        p(
            "学習512配置・検証30配置・最終評価100配置を別seedで生成した。検証結果だけで各学習seedの最良重みを選択し、最終評価を見て再調整していない。",
            "small",
        ),
        PageBreak(),
    ]
    story += [
        heading("3. 実測比較: 直接追従では通れない100配置"),
        p(
            "各方式で同じ開始姿勢・目標・障害物・評価seedを使用した。PPOは3つの学習seedの平均±標本標準偏差を示す。同じ100配置を繰り返しているため、300個の独立した配置とは数えない。"
        ),
    ]
    rows = [["方式", "成功率", "接触率", "平均手先速度"]]
    for row in data["display_rows"]:
        rows.append([row["label"], row["success"], row["collision"], row["speed"]])
    story += [
        table(rows, [195, 110, 110, 105]),
        Spacer(1, 6),
        p(
            "経路オラクルは可解性確認に使った経路を知る特別な参照制御であり、学習済みPPOとは区別する。完了時間は成功した試行だけで集計し、成功0件は未定義とした。",
            "small",
        ),
        picture("results/detour/comparison.png", 460),
        picture("results/detour/learning_curve.png", 460),
        p(
            "上: 最終評価。誤差棒は学習seed間の標準偏差。下: 検証30配置での学習経過。全試行の値はCSVへ保存し、動画の成功例だけから成功率を作っていない。",
            "small",
        ),
        PageBreak(),
    ]
    story += [
        heading("4. 自然言語と動作条件の変更"),
        p(
            f"ローカルLLM {escape(data['llm_model'])} を使用した。日本語24指示の期待ラベル一致は{data['llm_correct']}/24、キーワード方式は{data['keyword_correct']}/24。手作りの小規模データであり、一般的な分類性能の保証ではない。"
        ),
        p(
            "JSON schemaでmodeとreasonだけを許可し、不正出力・通信失敗時はcarefulへ戻す。LLMが自由な関節値やコードを作る構成にはしていない。慎重モードは速度と指令の変化を抑えるが、成功率や接触率が必ず改善するわけではない。"
        ),
        heading("5. 結果の解釈と限界"),
        p(escape(data["interpretation"])),
        p(
            "扱った課題は、固定の初期姿勢から肘を畳む経路がある配置に限る。把持、実機転移、未知形状、多数障害物、物体の破損防止は未検証。指令制限は実機の安全保証や物理的加速度の保証ではない。"
        ),
        heading("6. 再現方法と成果物"),
        p(
            "Python 3.11の仮想環境へrequirements-lock.txtを導入し、以下をリポジトリ直下で実行する。学習にはCPUを使用した。",
            "small",
        ),
        p("python -m pytest -q", "code"),
        p("python tools/run_detour_seeds.py --seeds 42 123 456", "code"),
        p("python -m src.benchmark_detour `", "code"),
        p("  --models checkpoints/detour_seed42/best_model.zip `", "code"),
        p("  checkpoints/detour_seed123/best_model.zip `", "code"),
        p("  checkpoints/detour_seed456/best_model.zip `", "code"),
        p("  --seeds 42 123 456 --output results/detour-recheck", "code"),
        p(
            "完全な引数、Ollamaの開始・停止、動画生成、同梱モデルによる再評価はREADMEの続きの手順に記載した。再評価時は新しい出力先を使い、保存済み結果を上書きしない。",
            "small",
        ),
        table(
            [
                ["成果物", "場所"],
                ["動画", "videos/final.mp4"],
                ["PPO重み", "checkpoints/detour_seed{42,123,456}/best_model.zip"],
                [
                    "全試行 / seed集計",
                    "results/detour/episode_results.csv / seed_summary.csv",
                ],
                [
                    "実験条件・hash",
                    "results/detour/protocol.json / results/final-verification.json",
                ],
            ],
            [115, 405],
        ),
        Spacer(1, 8),
        p(
            f"テスト: {data['tests']}件通過。追加学習の実測時間合計: {data['training_seconds']:.1f}秒。クラウドGPU稼働0時間・料金0 USD（ローカル電気代・通信料は未計測）。L40 $0.82/hは指示書の参考価格であり、利用実績ではない。",
            "small",
        ),
        heading("7. 使用技術・出典"),
        p(
            "MuJoCo / Gymnasium / Stable-Baselines3 PPO / PyTorch / NumPy / pandas / Matplotlib / Pydantic / Ollama / ImageIO / FFmpeg / ReportLab。新規の2リンクXMLと環境・評価コードを実装した。Qwen2.5:1.5Bの重みは配布物に含めない。",
            "small",
        ),
        p(
            '<link href="https://mujoco.readthedocs.io/en/latest/python.html" color="#087e8b">MuJoCo公式Python</link> / <link href="https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/" color="#087e8b">Gymnasium公式</link> / <link href="https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html" color="#087e8b">SB3 PPO公式</link> / <link href="https://ollama.com/blog/structured-outputs" color="#087e8b">Ollama構造化出力</link> / <link href="https://ollama.com/library/qwen2.5:1.5b" color="#087e8b">Qwen2.5:1.5B</link>',
            "small",
        ),
    ]
    output = ROOT / "output/pdf/physical_ai_report.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=34,
        bottomMargin=42,
        title="Instruction-Aware Safe Motion",
        author="",
        subject="Measured Physical AI project report",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    from pypdf import PdfReader

    reader = PdfReader(output)
    assert len(reader.pages) == 3, f"Expected 3 pages, got {len(reader.pages)}"
    assert all(page.extract_text().strip() for page in reader.pages)
    print(
        f"Saved {output.name}: {len(reader.pages)} pages, {output.stat().st_size} bytes"
    )


if __name__ == "__main__":
    main()
