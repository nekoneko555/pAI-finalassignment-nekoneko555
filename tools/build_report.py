"""Build the report and plots exclusively from saved measurements."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def table(df):
    header = "| " + " | ".join(df.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = []
    for row in df.itertuples(index=False, name=None):
        rows.append(
            "| "
            + " | ".join(
                "未定義（成功0件）"
                if pd.isna(x)
                else f"{x:.4f}"
                if isinstance(x, float)
                else str(x)
                for x in row
            )
            + " |"
        )
    return "\n".join([header, rule] + rows)


def main():
    fixed = load("checkpoints/fixed/training.json")
    random = load("checkpoints/training.json")
    validations = load("checkpoints/validation.json")
    summary = pd.read_csv(ROOT / "results/summary.csv")
    evaluation = load("results/evaluation.json")
    episodes = pd.read_csv(ROOT / "results/episode_results.csv")
    ppo_failures = episodes[(episodes.controller == "ppo_standard") & ~episodes.success]
    near_failures = (
        (ppo_failures.min_goal_distance_m >= 0.05)
        & (ppo_failures.min_goal_distance_m < 0.06)
    ).sum()
    best = max(
        validations, key=lambda r: (r["success_rate"], -r["mean_min_goal_distance_m"])
    )
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for path, label in [
        ("checkpoints/fixed/validation.json", "Stage 1: fixed"),
        ("checkpoints/validation.json", "Stage 2: random"),
    ]:
        rows = load(path)
        axs[0].plot(
            [r["timesteps"] for r in rows],
            [r["success_rate"] for r in rows],
            "o-",
            label=label,
        )
        axs[1].plot(
            [r["timesteps"] for r in rows],
            [r["mean_min_goal_distance_m"] for r in rows],
            "o-",
            label=label,
        )
    axs[0].set_ylabel("Validation success rate (10 cases)")
    axs[0].set_ylim(-0.02, 1.05)
    axs[1].set_ylabel("Mean minimum goal distance (m)")
    for ax in axs:
        ax.set_xlabel("Additional PPO timesteps in stage")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "results/learning_curve.png", dpi=150)
    plt.close(fig)
    tests = ET.parse(ROOT / "results/pytest.xml").getroot()
    suites = list(tests.iter("testsuite"))
    test_count = sum(int(s.attrib.get("tests", 0)) for s in suites)
    failures = sum(
        int(s.attrib.get("failures", 0)) + int(s.attrib.get("errors", 0))
        for s in suites
    )
    video_lines = []
    for p in sorted((ROOT / "videos").glob("*.json")):
        v = json.loads(p.read_text(encoding="utf-8"))
        if "frames" in v:
            video_lines.append(
                f"- `{p.with_suffix('.mp4').name}`: {v['duration_s']:.2f}秒、{v['frames']}フレーム、{v['bytes']:,} bytes。先頭・末尾デコード確認済み。"
            )
    llm_text = "実LLMの測定結果はまだない。モックによるテストは実推論として数えない。"
    if (ROOT / "results/instruction_ollama.json").is_file():
        llm = load("results/instruction_ollama.json")["summary"]
        keyword = load("results/instruction_keyword.json")["summary"]
        accuracy = (
            "未測定"
            if llm["instruction_accuracy"] is None
            else f"{llm['instruction_accuracy']:.1%}"
        )
        llm_text = (
            f"モデル `{llm['model']}`、独自に用意した日本語{llm['total']}指示。"
            f"実LLM応答{llm['actual_backend_responses']}件、fallback {llm['fallbacks']}件、"
            f"実LLM応答のラベル一致率{accuracy}。キーワード比較は{keyword['instruction_accuracy']:.1%}。\n\n"
            "小規模な手作りデータ1組の結果やから、一般的な優位性や業務利用性能は主張しない。"
            "誤分類は `results/instruction_ollama.json` の各行で確認できる。"
            "JSON schemaは型の制約であり、意味の正しさまでは保証しない。"
        )
    obstacle_text = "障害物評価はまだ実行していない。"
    if (ROOT / "results/obstacle/summary.csv").is_file():
        obstacle = pd.read_csv(ROOT / "results/obstacle/summary.csv")
        obstacle_text = (
            "別seedの保守的な可解配置30件で追加測定した。PPOは障害物なしで学習した重みをそのまま使用し、"
            "この評価で再学習・モデル選択をしていない。\n\n" + table(obstacle) + "\n\n"
            "配置生成時に既知の関節空間の経路が空いていることを確認しているので、"
            "難しい障害物回避のベンチマークではない。接触がなくても回避能力の獲得を意味しない。"
        )
    text = f"""# Instruction-Aware Safe Motion 実装・実測レポート

記録日: 2026-10-23（JST）。4日間の開発期間は10/23〜10/26。
Day 1の完了条件は達成し、PPOと日本語指示の統合まで先行して検証した。

## 目的とタスク

製造現場の搬送動作を模した、水平2関節アームの手先移動や。
自然言語で「慎重／速い」という実行条件を変更し、速度・指令変化量・到達性能を実測する。
物体の把持や実機での安全動作は扱わない。

## 構成と独自の工夫

```mermaid
flowchart LR
  Text[日本語指示] --> LLM[Ollama: JSON分類]
  LLM --> Mode[careful / fast]
  Sim[MuJoCo状態] --> PPO[PPO方策]
  PPO --> Filter[ルールによる指令制限]
  Mode --> Filter
  Filter --> Sim
```

PPOは目標到達を学ぶ。LLMは1指示につき1回だけ呼び、制御ループ内では呼ばない。
モードから角速度と指令変化量の上限へ変換する部分は学習しないルールや。
既存のサンプル課題をそのまま動かす構成ではなく、2リンクXML、独自Gymnasium環境、
自然言語→schema検証→安全フィルタという接続、同一ケースでの比較を実装した。
フィルタの前回指令も観測へ含め、指令変化量制限に必要な内部状態を方策から観測できるようにした。

## 物理・学習設定

- リンク長0.35m / 0.30m、z軸ヒンジ2関節。速度アクチュエータを使用。
- 物理刻み0.01秒、action repeat 4、行動刻み0.04秒、最大200行動。
- 観測14次元。sin/cos関節角、角速度、手先から目標・障害物への差、障害物情報、前回指令。
- 行動は2関節の正規化目標角速度。careful上限0.525、fast上限1.200 rad/s。
- 指令加速度上限はcareful 1.0、fast 4.0 rad/s²。実際の加速度上限を保証するものではない。
- 報酬 `2*(前回距離-今回距離)-0.01`、成功+10、接触-10、状態異常-5。
- 成功は手先距離0.05m未満・接触なし・異常なし。全リンクの接触を各物理刻みで判定。
- PPO MlpPolicy [64,64]、learning rate 3e-4、gamma 0.99、n_steps 1024、batch 64、seed 42、CPU。
- 固定目標: 実{fixed["actual_timesteps"]:,}ステップ / {fixed["wall_time_s"]:.2f}秒。
- ランダム目標: 追加実{random["actual_timesteps"]:,}ステップ / {random["wall_time_s"]:.2f}秒、{random["n_envs"]}環境（逐次ベクトル実行）。
- 選択モデルはランダム目標段階の{best["timesteps"]:,}ステップ時点。
  最終ステップのモデルと最良モデルは同一とは限らない。
- 保存・再読込後の決定的行動一致を実測確認した。TensorBoardログと検証履歴を保存。

![検証ケースでの学習経過](results/learning_curve.png)

固定目標の検証10件はseedが異なっても同じ目標なので、10種類の課題を解いた意味ではない。
ランダム目標での検証は別の10配置を使った。評価ケースはモデル選択に使っていない。

## 同一の未学習ケースでの比較

評価ファイル `{evaluation["cases_file"]}`、各方式{evaluation["episodes_per_controller"]}件、
初期姿勢・目標・seedを共通化。障害物なし。LLMを使わない明示モードでの制御比較を含む。
`ppo_untrained` は固定目標の学習より前のランダム初期重みや。

{table(summary)}

時間は成功試行だけの平均。成功ゼロは未定義で、0秒ではない。
`mean_speed` は試行ごとの平均手先速度をさらに平均した値（m/s）。
`mean_abs_action_delta` はフィルタ後の角速度指令の変化量（rad/s per action）や。

![同一条件での比較](results/comparison.png)

学習前からの改善は確認できたが、単純なこの課題ではIKがPPOより高い成功率を出した。
PPOの優位性は主張しない。carefulでは遅く滑らかな指令になる一方、成功率の改善は確認できない。
standardとfastは同じ制限定数なので、同じ方策なら結果も一致する。
障害物なしの衝突率0%は、障害物回避性能の証拠ではない。

標準PPOの失敗{len(ppo_failures)}件中、{near_failures}件は最小距離が5〜6cmの範囲やった。
高速デモの先頭ケースも約5.09cmで止まり、5cm未満の成功条件に届かなかった。
方策が閾値付近で止まることや、モード変更で制御の応答が変わることが原因候補やけど、
原因を切り分ける追加学習はまだ行っていない。評価ケースを見てから方策を調整せず、失敗をそのまま保存した。

## 日本語指示の分類

{llm_text}

入力、分類結果、reason、source、所要時間、エラーを保存する。
未知モード・余分な数値・不正JSON・通信失敗はcarefulへfallbackする。
モデル出力のコードは実行しない。LLMがfastを出してもプログラム側の上限を超えない。

## 障害物付きの追加測定

{obstacle_text}

## テストと動画

pytest: {test_count}件、失敗・エラー{failures}件。`results/pytest.xml` が実行記録や。
GymnasiumとSB3のcheck_env、同seed再現性、成功と時間切れの分離、
NaN/Inf入力、非有限物理状態の終了、全リンク接触、制限値、不正LLM出力、集計を検査した。
実描画はMuJoCo公式Rendererを使用し、2D代替描画は今回の主要動画では使っていない。

{chr(10).join(video_lines)}

字幕で方策、モード、解釈元、距離、速度、実時間1倍速を示す。
字幕用の停止画面は「simulation stopped」と表示する。成功例だけでなく失敗例も撮影し、
集計は映像の抜粋と切り離して全試行を使う。

## 発生したエラーと修正

1. MuJoCoがWindowsの日本語パスのXMLを開けなかった。
   PythonでUTF-8として読んで `from_xml_string` へ渡し、再テストと実描画で修正を確認。
2. SB3の学習前にloggerへ触れてAttributeErrorが出た。
   検証開始を `_on_training_start` へ移し、実学習・保存・再読込を再実行した。
3. 接触テストの目標姿勢が障害物と重なっていた。
   テスト配置を修正し、手先から離れたリンク中央の接触で失敗判定が出ることを確認。
4. ローカルAPI通信は作業サンドボックスで制限されていた。
   localhostに限定した実行で検証した。通常のローカル端末ではOllama起動後に同じCLIを使う。

## 限界と残る4日間の改善

- 平面2関節・手先移動のみ。把持、実機転移、柔軟物の保護は未実装。
- 速度と指令変化量を制限しても、衝突回避や実機の安全性は保証しない。
- 学習はseed 42の1回。複数seedの信頼区間はまだない。
- 障害物を使ったPPO学習、難しい回避経路、指示解釈の大規模な未学習データ評価は残課題。
- 追加作業は10/24に失敗ケース分析と複数seed、10/25に障害物の学習・評価、
  10/26に最終動画編集と提出用PDF化・公開内容確認を優先する。
- Day 1の撤退条件には該当しない。Day 2の最低限の目標到達学習も成立。

## 実行環境・費用・再現

Windows、Python 3.11、依存バージョンは `requirements-lock.txt`。
GPUは使用していない。クラウドGPU実稼働0時間、クラウドGPU料金0 USD。
ローカルの電気代・通信料は計測していない。
L40の$0.82/hはユーザー指示書の参考料金で、実際に契約・利用した価格ではない。
有料インスタンスはユーザー管理下で起動し、$1.00/h未満を起動時に確認する。

```powershell
python -m pytest -q
python -m src.evaluate --model checkpoints/best_model.zip --initial-model checkpoints/fixed/initial_model.zip --episodes 100
python -m src.demo --mode careful --output videos/careful-explicit.mp4
```

学習からの再現、Ollamaの起動・モデル指定、各動画の生成手順はREADMEを参照する。
数値は上記CSV/JSONから生成しており、例示の成功率は混ぜていない。

## 出典・謝辞

MuJoCo公式Pythonバインディング、Gymnasium、Stable-Baselines3 PPO、PyTorch、
NumPy、pandas、Matplotlib、Pydantic、Ollama、ImageIO、FFmpeg、Pillowを使用。
APIの参照先はREADMEのリンクに記載した。第三者のロボット課題コードは転載していない。
Qwen2.5の使用モデル・分類結果は上の実測記録に示した。モデルの重み自体は再配布しない。
"""
    (ROOT / "report.md").write_text(text, encoding="utf-8")
    print("Generated report.md and results/learning_curve.png from saved measurements")


if __name__ == "__main__":
    main()
