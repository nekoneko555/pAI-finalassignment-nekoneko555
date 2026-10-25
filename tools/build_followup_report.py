"""Generate follow-up prose/plots from measured CSVs without changing Day 1 evidence."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def markdown_table(frame):
    rows = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        rows.append(
            "| "
            + " | ".join(
                "未定義"
                if pd.isna(v)
                else f"{v:.4f}"
                if isinstance(v, float)
                else str(v)
                for v in row
            )
            + " |"
        )
    return "\n".join(rows)


def main():
    summary = pd.read_csv(ROOT / "results/detour/summary.csv")
    aggregate = pd.read_csv(ROOT / "results/detour/seed_aggregate.csv")
    seeds = pd.read_csv(ROOT / "results/detour/seed_summary.csv")
    protocol = read_json("results/detour/protocol.json")
    training = [
        read_json(f"checkpoints/detour_seed{seed}/training.json")
        for seed in (42, 123, 456)
    ]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
    for seed in (42, 123, 456):
        history = read_json(f"checkpoints/detour_seed{seed}/validation.json")
        for ax, key in zip(axs, ["success_rate", "collision_rate"]):
            ax.plot(
                [r["timesteps"] for r in history],
                [r[key] for r in history],
                "o-",
                markersize=3,
                label=f"seed {seed}",
            )
    for ax, title in zip(
        axs,
        ["Validation success rate (30 cases)", "Validation collision rate (30 cases)"],
    ):
        ax.set_title(title)
        ax.set_xlabel("PPO timesteps")
        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "results/detour/learning_curve.png", dpi=170)
    plt.close(fig)
    suites = list(ET.parse(ROOT / "results/pytest.xml").getroot().iter("testsuite"))
    tests = sum(int(s.attrib.get("tests", 0)) for s in suites)
    errors = sum(
        int(s.attrib.get("errors", 0)) + int(s.attrib.get("failures", 0))
        for s in suites
    )
    assert errors == 0
    llm = read_json("results/instruction_ollama.json")
    keyword = read_json("results/instruction_keyword.json")
    display = []
    for key, label in [("ik", "直接IK"), ("previous_ppo", "前回PPO（障害物なし学習）")]:
        row = summary[summary.controller == key].iloc[0]
        display.append(
            {
                "label": label,
                "success": f"{row.success_rate:.1%}",
                "collision": f"{row.collision_rate:.1%}",
                "speed": f"{row.mean_speed:.3f} m/s",
            }
        )
    for mode, label in [
        ("standard", "PPO・標準（3 seed）"),
        ("careful", "PPO・慎重（3 seed）"),
        ("fast", "PPO・高速（3 seed）"),
    ]:
        row = aggregate[aggregate["mode"] == mode].iloc[0]
        display.append(
            {
                "label": label,
                "success": f"{100 * row.mean_success_rate:.1f} ± {100 * row.sd_success_rate:.1f}%",
                "collision": f"{100 * row.mean_collision_rate:.1f} ± {100 * row.sd_collision_rate:.1f}%",
                "speed": f"{row.mean_speed:.3f} m/s",
            }
        )
    row = summary[summary.controller == "oracle"].iloc[0]
    display.append(
        {
            "label": "経路オラクル（特別な参照）",
            "success": f"{row.success_rate:.1%}",
            "collision": f"{row.collision_rate:.1%}",
            "speed": f"{row.mean_speed:.3f} m/s",
        }
    )
    standard = aggregate[aggregate["mode"] == "standard"].iloc[0]
    old = summary[summary.controller == "previous_ppo"].iloc[0]
    careful = aggregate[aggregate["mode"] == "careful"].iloc[0]
    interpretation = (
        f"標準PPOの成功率は学習seed平均{standard.mean_success_rate:.1%}（SD {standard.sd_success_rate:.1%}）、"
        f"接触率は{standard.mean_collision_rate:.1%}（SD {standard.sd_collision_rate:.1%}）。"
        f"前回PPOの成功率{old.success_rate:.1%}、接触率{old.collision_rate:.1%}と比較した。"
        f"慎重モードの平均手先速度は{careful.mean_speed:.3f}m/s、標準は{standard.mean_speed:.3f}m/sだった。"
        f"慎重モードの接触率は{careful.mean_collision_rate:.1%}で、標準より高かった。"
        "同じ8秒の制限内で比較した結果で、速度を下げるだけでは接触を防げなかった。"
    )
    final = {
        "tests": tests,
        "display_rows": display,
        "training_seconds": sum(r["wall_time_s"] for r in training),
        "training_steps": sum(r["actual_timesteps"] for r in training),
        "llm_model": llm["summary"]["model"],
        "llm_correct": sum(r["correct"] for r in llm["rows"]),
        "keyword_correct": sum(r["correct"] for r in keyword["rows"]),
        "interpretation": interpretation,
        "training_seeds": [42, 123, 456],
        "cases_per_seed": protocol["episodes_per_model"],
        "cloud_gpu_hours": 0,
        "cloud_gpu_cost_usd": 0,
    }
    (ROOT / "results/final-summary.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    selected = []
    for seed in (42, 123, 456):
        hist = read_json(f"checkpoints/detour_seed{seed}/validation.json")
        best = max(
            hist, key=lambda r: (r["success_rate"], -r["mean_min_goal_distance_m"])
        )
        selected.append(
            {
                "seed": seed,
                "selected_timesteps": best["timesteps"],
                "validation_success": best["success_rate"],
            }
        )
    lines = [
        "# 続きの実装・最終実測レポート",
        "",
        "記録日: 2026-10-23（JST）。前回のDay 1成果はそのまま保存した。",
        "",
        "## 追加した内容",
        "",
        "直接IKでは接触し、肘を畳む既知の経路なら通れる障害物配置を実装した。学習512配置、検証30配置、最終評価100配置は別seedや。",
        "経路を0.01 rad以下の刻みで調べ、両リンクのクリアランスが18mmより大きい配置を採用した。MuJoCoでの参照制御でも可解性を確認した。",
        "学習時は既知経路上のゴール近傍から開始し、20万ステップ基準のスケジュールで開始姿勢を広げた。約17万ステップで元の開始姿勢だけに切り替える。",
        "経路は開始姿勢の教材と可解性の確認に使う。PPOへの観測14次元に経路・目標関節角・経路進行度は含めず、推論時にも経路を使わない。",
        "",
        "報酬は前回の式に距離コスト -0.3×手先距離を追加。5cmの成功境界付近で止まる傾向を抑える狙いや。この設定は最終評価前に固定した。",
        "3つの学習seedはランダム初期重みから独立に開始し、前回モデルを初期値として使っていない。各303,104ステップ、4環境の逐次ベクトル実行、CPUや。",
        "",
        "## モデル選択と試験条件",
        "",
        markdown_table(pd.DataFrame(selected)),
        "",
        "各seedで検証成功率が最大、同率なら最小距離が小さい重みを選択した。100配置の最終評価を見る前に重みと条件を固定し、hashをprotocol.jsonへ保存した。",
        "最終評価後の再学習・ハイパーパラメータ変更は行っていない。",
        "",
        "## 全方式の実測結果",
        "",
        markdown_table(summary),
        "",
        "## 学習seedごとの結果",
        "",
        markdown_table(seeds),
        "",
        "## 学習seed間の集計",
        "",
        markdown_table(aggregate),
        "",
        "同じ100配置を各モデルで評価したので、300配置が独立とは解釈しない。SDは学習seed 3つの間の標本標準偏差や。3回だけなので不確実性の推定には限界がある。",
        "完了時間は成功した試行だけの平均で、成功0件は未定義。速度は試行ごとの平均をさらに平均した値や。",
        "",
        interpretation,
        "報酬変更と開始姿勢のカリキュラムを組み合わせた結果で、各要素の寄与は個別に切り分けていない。",
        "",
        "![最終比較](results/detour/comparison.png)",
        "",
        "![検証経過](results/detour/learning_curve.png)",
        "",
        "## LLMと役割分担",
        "",
        f"Qwen2.5:1.5Bで日本語24指示中{final['llm_correct']}件一致、キーワード方式は{final['keyword_correct']}件一致。前回と同じ保存済みの実推論結果を使い、追加の分類性能として水増ししていない。",
        "今回の100配置比較では、分類誤りと制御性能を分けるためモードを明示指定した。実LLMで指示を解釈して新しいPPOを動かす様子は追加動画で確認する。",
        "PPOが生成するのは角速度で、careful/fastの速度・指令変化量制限はルールベース。2種類の速度方策を学習したわけではない。",
        "",
        "## 実測時間・テスト・費用",
        "",
        f"追加学習は合計{final['training_steps']:,}ステップ、{final['training_seconds']:.2f}秒。pytestは{tests}件通過。保存・復元後の決定的行動一致を各seedで検査した。",
        "可変半径に合わせてMuJoCoの接触判定用境界も更新し、半径変更で実際の接触が切り替わるテストを追加した。",
        "クラウドGPUは未使用、稼働0時間・料金0 USD。ローカルの電気代・通信料は未計測。L40 $0.82/hは参考料金であり利用実績ではない。",
        "",
        "## 限界と提出前の作業",
        "",
        "平面2関節・手先移動だけで、把持や実機での動作は未実装。障害物1個、初期姿勢固定、可解な配置の一群だけの評価や。任意の配置へ一般化したとは主張せえへん。",
        "速度制限だけで接触回避や安全を保証しない。特に慎重モードでも8秒制限は共通なので、所要時間と成功率の関係を併記する。",
        "提出用PDFと動画・コードの準備を行う。OmniCampusアカウントIDに応じたファイル名・動画タイトルの反映、GitHubの公開とYouTube投稿はまだ行っていない。",
        "",
        "## 再現",
        "",
        "```powershell",
        "python -m pytest -q",
        "python tools/run_detour_seeds.py --seeds 42 123 456",
        "python -m src.benchmark_detour --models checkpoints/detour_seed42/best_model.zip checkpoints/detour_seed123/best_model.zip checkpoints/detour_seed456/best_model.zip --seeds 42 123 456 --output results/detour-recheck",
        "```",
        "",
        "依存関係はrequirements-lock.txt、従来課題の構成と再現手順はREADME、前回結果はreport.mdを参照する。APIの出典はREADMEに記載した。",
    ]
    if (ROOT / "videos/final.json").exists():
        video = read_json("videos/final.json")
        lines += [
            "",
            "## 作成済みの提出物",
            "",
            f"最終動画 `videos/final.mp4` は{video['duration_s']:.2f}秒、{video['frames']}フレーム、25fps。実動作は1倍速、説明カードは停止画面や。",
            "提出用PDFは `output/pdf/physical_ai_report.pdf` の3ページ。全ページを画像化して表示を確認した。",
            "コード・データ・モデル・動画・PDFを `output/physical-ai-delivery.zip` にまとめる。最新の整合性検査は `results/final-verification.json`。",
            "追加動画は評価ケースの先頭を固定し、LLM解釈を保存したうえで使っている。カメラ調整による撮り直しでは同じ解釈を再利用し、新しい分類評価として数えていない。",
        ]
    (ROOT / "report-final.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
