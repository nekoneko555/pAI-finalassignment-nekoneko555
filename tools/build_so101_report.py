"""Build measured plots and report without mixing in the planar-arm results."""

import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402


def main():
    folder = ROOT / "results/so101/final"
    summary = pd.read_csv(folder / "summary.csv")
    aggregate = pd.read_csv(folder / "seed_aggregate.csv")
    training = [
        json.loads(
            (ROOT / f"checkpoints/so101_seed{s}/training.json").read_text(
                encoding="utf8"
            )
        )
        for s in [42, 123, 456]
    ]
    labels = ["IK", "Untrained\nresidual", "PPO\nstandard", "PPO\ncareful"]
    success = [
        summary.set_index("controller").loc[n, "success_rate"]
        for n in ["ik", "untrained_residual"]
    ] + aggregate.set_index("mode").loc[
        ["standard", "careful"], "mean_success_rate"
    ].tolist()
    duration = [
        summary.set_index("controller").loc[n, "mean_time_s"]
        for n in ["ik", "untrained_residual"]
    ] + aggregate.set_index("mode").loc[["standard", "careful"], "mean_time_s"].tolist()
    speed = [
        summary.set_index("controller").loc[n, "mean_tip_speed_m_s"]
        for n in ["ik", "untrained_residual"]
    ] + aggregate.set_index("mode").loc[
        ["standard", "careful"], "mean_tip_speed_m_s"
    ].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    for ax, values, title in zip(
        axes,
        [np.array(success) * 100, duration, speed],
        [
            "Physical pick/place success (%)",
            "Time incl. dwell and retreat (s)",
            "Mean hand speed (m/s)",
        ],
    ):
        ax.bar(labels, values, color=["#8a9ab0", "#b4bfce", "#087e8b", "#3bab94"])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0, 110)
    fig.suptitle(
        "SO-101: 100 shared held-out A/B placements; PPO means over 3 training seeds",
        fontsize=12,
    )
    fig.savefig(folder / "comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
    for seed in [42, 123, 456]:
        rows = pd.read_json(ROOT / f"checkpoints/so101_seed{seed}/validation.json")
        axes[0].plot(
            rows.timesteps, rows.success_rate * 100, "o-", label=f"seed {seed}"
        )
        axes[1].plot(rows.timesteps, rows.mean_time_s, "o-", label=f"seed {seed}")
    for ax in axes:
        ax.set_xlabel("Training timesteps")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("Validation success (%)")
    axes[0].set_ylim(0, 105)
    axes[1].set_ylabel("Validation mean completion time (s)")
    fig.suptitle("30 validation placements; best checkpoint by success then time")
    fig.savefig(folder / "learning_curve.png", dpi=160)
    plt.close(fig)
    table = "|方式|成功率|全試行平均時間|平均手先速度|\n|---|---:|---:|---:|\n"
    for r in summary.itertuples():
        table += f"|{r.controller}|{r.success_rate:.1%}|{r.mean_time_s:.3f}s|{r.mean_tip_speed_m_s:.4f}m/s|\n"
    totals = {
        "tests_passed": sum(
            int(s.attrib["tests"])
            for s in ET.parse(ROOT / "results/so101/pytest.xml")
            .getroot()
            .iter("testsuite")
        ),
        "training_timesteps": sum(t["actual_timesteps"] for t in training),
        "training_seconds": sum(t["wall_time_s"] for t in training),
        "summary": summary.to_dict(orient="records"),
        "aggregate": aggregate.to_dict(orient="records"),
        "training": training,
    }
    (ROOT / "results/so101/report-data.json").write_text(
        json.dumps(totals, indent=2), encoding="utf8"
    )
    text = f"""# 日本語で動作条件を変えるSO-101の物理把持・運搬

2026-10-23。SO-101のMuJoCoシミュレーションであり、実機の検証ではない。

## 目的と独自の工夫

A地点に置いた35g・一辺36mmの立方体へ近づき、指でつかんで持ち上げ、B地点へ運んで離す。同じ物体・配置・学習済み重みで、日本語指示に応じて関節の速度・指令変化量制限を切り替える。実在するSO-101のメッシュ・関節・サーボ・重力・接触モデルを用いる。

## 学習した部分と設計した部分

- モデル: MuJoCo Menagerieのrobotstudio_so101、コミットcda5236f98383f6fbaa4820c6b18dba892cd087f。Apache-2.0のライセンスと元ファイルを保存し、動的物体や作業机は別コードで追加した。
- IKが8つの経由姿勢を計算し、状態機械が近接・下降・把持・持ち上げ・運搬・下降・解放・退避を進める。グリッパーの開閉はルール制御。
- PPOは5関節の基準追従速度を最大±35%補正する残差制御。観測43次元に目標関節誤差、サーボ遅れ、関節状態、指令状態、作業段階、物体相対位置、モード制限を含む。把持全体をゼロから学習した方式ではない。
- 日本語をローカルQwen2.5:1.5Bがcareful/fastへ分類。1指示に1回だけ実行し、制御ループ外で設定する。不正出力・通信失敗はcarefulへフォールバックし、実LLM成功と区別する。
- carefulは関節速度指令0.4rad/s・指令加速度0.8rad/s²、standard/fastは1.2・3.0。同じ重みを使う。物理的な速度・加速度や実機安全を保証する制約ではない。

## 物理的把持と成功条件

物体はMuJoCoの自由剛体。移動中に物体の座標を書き換えず、溶接拘束も使わない。両指の接触を伴う状態で物体中心が6cmを超えることを記録する。8段階を終え、物体をB地点の半径3.5cm以内へ下ろし、指を開いて接触を離し、物体中心が高さ1〜3.5cmかつ速度0.03m/s未満で静止した場合を成功とする。物体の初期中心高さは1.8cm。

## 実験設計

学習128配置、検証30配置、最終評価100配置のseedを分離した。独立に初期化した学習seedは42/123/456。各{training[0]["actual_timesteps"]:,}ステップ、合計{totals["training_timesteps"]:,}ステップ、CPU学習の実測時間合計{totals["training_seconds"]:.2f}秒。PPOはMLP64×64、lr3e-4、n_steps512、batch128、4つの逐次ベクトル環境。保存と復元後の決定的行動一致を確認した。

検証成功率が最大、同率なら完了時間が短い重みを選ぶ。最終評価前にモデルのhashと条件をprotocol.jsonへ固定した。動画の学習seed42と最終評価先頭のケースを使用し、最終評価を見てデモ用モデルを選ばない。

全方式に同じ100配置を使う。同じ配置を3つの学習seedで繰り返すため、300個の独立した配置とは扱わない。standardとfastの定数は同じであり、別の高速方策を学習したわけではない。

## 全試行の実測結果

{table}

![比較](results/so101/final/comparison.png)

![検証経過](results/so101/final/learning_curve.png)

## 結果の解釈

IKだけでも成立する課題であり、学習なしの基準制御も提示する。PPOを導入したこと自体を性能向上の証拠にしない。PPO・モード間の差は上の実測値で判断する。全試行平均時間は把持や解放の待機・退避を含み、失敗試行も含む。手先速度も待機を含む全ステップ平均。成功率100%であっても、この限定した100配置で失敗がなかったという意味であり、任意の配置での成功を保証しない。

## 限界

固定した開始姿勢、既知寸法の立方体、狭いA/B範囲、既知の物体座標だけの検証。カメラ画像からの認識、任意形状、未知位置の探索、実機転移は実装していない。手首可動範囲から、物体の持ち上げ高さも限定される。慎重モードという名前は安全保証を意味しない。

## 出典と再現

https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101

モデルとライセンスはassets/so101/upstream、固定版と各ファイルhashはassets/so101/source.json。MuJoCo、Gymnasium、Stable-Baselines3、PyTorch、Ollama、Qwen、ImageIO等を使用。依存はrequirements-lock.txt。手順はREADME-SO101.mdを参照。

テストは{totals["tests_passed"]}件通過。クラウドGPUは未使用、0時間・0USD。ローカル電気代は未計測。GitHub公開・YouTube投稿・OmniCampus提出はまだ行っていない。
"""
    (ROOT / "report-so101.md").write_text(text, encoding="utf8")
    print(
        json.dumps(
            {k: v for k, v in totals.items() if k.startswith("training_")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
