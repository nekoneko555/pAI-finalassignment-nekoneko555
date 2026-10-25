"""Three-page report of actual SO-101 pick/place and residual PPO measurements."""

import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Table,
    TableStyle,
    Image,
    PageBreak,
)
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def main():
    rl_config.invariant = 1
    data = json.loads(
        (ROOT / "results/so101/report-data.json").read_text(encoding="utf8")
    )
    summary = {r["controller"]: r for r in data["summary"]}
    agg = {r["mode"]: r for r in data["aggregate"]}
    pdfmetrics.registerFont(TTFont("SOJP", "C:/Windows/Fonts/meiryo.ttc"))
    navy = colors.HexColor("#14283f")
    teal = colors.HexColor("#087e8b")
    styles = {
        k: ParagraphStyle(
            k,
            fontName="SOJP",
            fontSize=size,
            leading=lead,
            textColor=navy,
            wordWrap="CJK",
            spaceAfter=6,
        )
        for k, size, lead in [
            ("body", 9, 14),
            ("small", 7.6, 11),
            ("title", 22, 29),
            ("h", 13, 19),
        ]
    }
    styles["h"].textColor = teal
    styles["h"].spaceBefore = 8

    def p(text, style="body"):
        return Paragraph(text, styles[style])

    def table(rows, widths):
        cells = [
            [
                p(
                    ('<font color="white">' + escape(str(c)) + "</font>")
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
                    ("BACKGROUND", (0, 0), (-1, 0), navy),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.HexColor("#eff5f8"), colors.white],
                    ),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        return t

    def img(relative, width, height=None):
        from PIL import Image as PIL

        with PIL.open(ROOT / relative) as raw:
            w, h = raw.size
        return Image(str(ROOT / relative), width=width, height=height or width * h / w)

    def footer(canvas, doc):
        canvas.setFont("SOJP", 7)
        canvas.setFillColor(navy)
        canvas.drawString(
            36, 20, "Language-Conditioned SO-101 Pick & Place | 2026-10-23"
        )
        canvas.drawRightString(A4[0] - 36, 20, f"{doc.page} / 3")

    story = [
        p("日本語で動作条件を変える<br/>SO-101の把持・運搬", "title"),
        p("Physical AI応用講座 最終課題 / MuJoCoシミュレーション", "small"),
        p("1. 目的と独自の工夫", "h"),
        p(
            "机のA地点の立方体へ近づき、指でつかんで持ち上げ、B地点へ運んで離す。同じ物体・配置・PPO重みに対し、日本語指示で動作条件を変更する。実在するSO-101のモデルを用い、実際の接触と摩擦による把持を検証した。"
        ),
        img("results/so101/lifted-preview.jpg", 520),
        p(
            "図: 実際のMuJoCo状態から描画した把持・運搬中の様子。物体は自由剛体であり、座標の直接変更や溶接拘束を使って保持していない。実機写真ではない。",
            "small",
        ),
        p("2. AIと設計した制御の役割", "h"),
        table(
            [
                ["要素", "役割"],
                ["IK + 状態機械", "8段階の経由姿勢と指の開閉を決める"],
                ["PPO（学習部分）", "5関節の基準追従速度を最大±35%補正する残差制御"],
                [
                    "Qwen2.5:1.5B",
                    "日本語をcareful / fastへ分類。指示ごとに1回、ループ外で実行",
                ],
                [
                    "モード制限",
                    "careful: 0.4rad/s・0.8rad/s² / fast: 1.2・3.0（指令の上限）",
                ],
            ],
            [112, 408],
        ),
        p(
            "把持全体をゼロから学習した方式ではない。standardとfastは同じ定数であり、2種類の速度方策を学習したわけではない。",
            "small",
        ),
        PageBreak(),
        p("3. 学習と比較実験", "h"),
        p(
            f"学習128配置・検証30配置・最終評価100配置を別seedで生成。学習seed 42 / 123 / 456を独立初期化し、合計{data['training_timesteps']:,}ステップをCPUで学習した。検証成功率が最大、同率なら平均時間が短い重みを選択し、最終評価前にhashを固定した。"
        ),
        p(
            "PPOはMLP64×64、学習率3e-4、n_steps512、batch128、4逐次ベクトル環境。観測43次元、行動5次元。報酬は関節誤差改善・時間コスト・段階進行・最終成功を使用。"
        ),
        img("results/so101/final/comparison.png", 520),
        table(
            [
                ["方式", "成功率", "平均時間", "平均手先速度"],
                [
                    "IK基準",
                    f"{summary['ik']['success_rate']:.1%}",
                    f"{summary['ik']['mean_time_s']:.3f}s",
                    f"{summary['ik']['mean_tip_speed_m_s']:.4f}m/s",
                ],
                [
                    "未学習PPO補正",
                    f"{summary['untrained_residual']['success_rate']:.1%}",
                    f"{summary['untrained_residual']['mean_time_s']:.3f}s",
                    f"{summary['untrained_residual']['mean_tip_speed_m_s']:.4f}m/s",
                ],
                [
                    "PPO標準",
                    f"{agg['standard']['mean_success_rate']:.1%} ± {agg['standard']['sd_success_rate']:.1%}",
                    f"{agg['standard']['mean_time_s']:.3f}s",
                    f"{agg['standard']['mean_tip_speed_m_s']:.4f}m/s",
                ],
                [
                    "PPO慎重",
                    f"{agg['careful']['mean_success_rate']:.1%} ± {agg['careful']['sd_success_rate']:.1%}",
                    f"{agg['careful']['mean_time_s']:.3f}s",
                    f"{agg['careful']['mean_tip_speed_m_s']:.4f}m/s",
                ],
            ],
            [160, 130, 100, 130],
        ),
        p(
            "各方式で同じ100配置を使用。PPOは3学習seedの平均、±は学習seed間の標本標準偏差。同じ100配置の繰り返しなので300個の独立配置とは数えない。平均時間・速度は待機と退避、失敗試行も含む。",
            "small",
        ),
        img("results/so101/final/learning_curve.png", 520),
        p(
            "IKだけでも成立する課題である。PPO導入を把持成功や性能向上の根拠とせず、基準との差は実測で判断する。",
            "small",
        ),
        PageBreak(),
        p("4. 物理的把持と成功条件", "h"),
        p(
            "35g・一辺36mmの立方体を自由剛体として扱う。両指に接触したまま物体中心が6cmを超えて持ち上がり、全8段階を完了し、B地点の半径3.5cm以内で解放され、物体中心高さ1〜3.5cm・並進速度0.03m/s未満で静止した場合を成功とする。初期の物体中心高さは1.8cm。単にBへ近づいただけでは成功としない。"
        ),
        p("5. 自然言語による動作条件の変更", "h"),
        p(
            "『精密部品なので慎重にA地点からB地点へ運んで』『急いでA地点からB地点へ運んで』の2指示をローカルLLMへ実際に入力し、同じ評価先頭ケース・同じPPO重みで撮影した。JSON schemaでmode / reasonだけを許可する。不正出力や通信失敗はcarefulへ戻し、実推論と区別して記録する。2件のデモから一般的な言語理解性能は主張しない。"
        ),
        p("6. 限界と解釈", "h"),
        p(
            "固定開始姿勢、既知の立方体、狭いA/B配置範囲、既知座標での検証に限る。カメラ画像からの認識、未知位置探索、任意形状、実機転移は未実装。手首の可動範囲から持ち上げ高さにも制限がある。速度指令の制限は実測の物理速度・加速度や実機安全を保証せず、慎重という名称も安全保証を意味しない。"
        ),
        p("7. 再現・出典・提出物", "h"),
        table(
            [
                ["成果物", "場所"],
                ["最終動画", "videos/so101_final.mp4"],
                ["コード・再現手順", "README-SO101.md / src/so101_task.py"],
                ["PPO重み", "checkpoints/so101_seed{42,123,456}/best_model.zip"],
                [
                    "全試行・実験条件",
                    "results/so101/final/episode_results.csv / protocol.json",
                ],
                [
                    "実LLM・動作記録",
                    "results/so101/demo_instructions.json / videos/so101_*.trajectory.csv",
                ],
            ],
            [112, 408],
        ),
        p(
            f"テスト{data['tests_passed']}件通過。学習時間の実測合計{data['training_seconds']:.1f}秒。クラウドGPUは未使用・0時間・0USD。依存はrequirements-lock.txt。手順はREADME-SO101.md。公開・投稿・OmniCampus提出は未実施。",
            "small",
        ),
        p(
            'モデル: <link href="https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101" color="#087e8b">MuJoCo Menagerie / The Robot Studio SO-101</link>。コミットcda5236f98383f6fbaa4820c6b18dba892cd087f。Apache-2.0ライセンスと元ファイルを同梱し、物体・机・制御・評価を別コードで追加。MuJoCo、Gymnasium、SB3 PPO、PyTorch、Ollama、Qwen、ImageIO、ReportLab等を使用。',
            "small",
        ),
    ]
    out = ROOT / "output/pdf/so101_report.pdf"
    doc = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=32,
        bottomMargin=40,
        title="Language-Conditioned SO-101 Pick & Place",
        author="",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    assert len(PdfReader(out).pages) == 3
    print(out.name, out.stat().st_size)


if __name__ == "__main__":
    main()
