> **SO-101版を追加:** A地点の物体をつかみ、B地点へ運んで離すシミュレーションは [README-SO101.md](README-SO101.md) を参照。以下は保存している従来の2関節版の説明や。

# Instruction-Aware Safe Motion

**続きの実装:** 障害物を迂回するPPOをseed 42 / 123 / 456で独立に学習した。
最新の結果は `report-final.md`、提出用PDFは `output/pdf/physical_ai_report.pdf`、
最終動画は `videos/final.mp4`。前回のDay 1成果とモデルは保存している。

自然言語で動作条件を変える、MuJoCo上の水平2関節アームやで。
把持は実装せず、工場の搬送を模した**手先移動**を扱う。

## まず動かす

このリポジトリ直下で実行する。Windows / Python 3.11で実行確認済み。
LinuxではPython 3.11の仮想環境を作り、ヘッドレス描画時は `export MUJOCO_GL=egl` を設定する。

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-lock.txt
.\.venv\Scripts\Activate.ps1
python -m pytest -q
python -m src.render_video --episodes 20 --output videos/baseline.mp4
```

PowerShellの実行ポリシーでactivateできへん場合は、以降の `python` を
`.\.venv\Scripts\python.exe` に置き換えればええ。Linuxでは `.venv/bin/python` を使う。
初回セットアップにuvを使わへん場合は `python -m venv .venv` と
仮想環境の `python -m pip install -r requirements-lock.txt` でも構わへん。

## 構成と責任分担

```text
日本語指示 → llm_parser.py (1指示に1回) → careful / fast
                                            ↓
MuJoCo状態 → 観測 → PPO方策 → 正規化角速度 → controller.py → 速度アクチュエータ
       ↑                                                   ↓
       └────────────────── MuJoCo物理計算 ──────────────────┘
```

| ファイル | 役割 / 読むと分かること |
|---|---|
| `assets/arm.xml` | リンク長、ヒンジ関節、接触マスク、速度アクチュエータ |
| `src/env.py` | Gymnasiumのreset/step、観測・報酬・終了条件 |
| `src/controller.py` | 学習しない速度・指令変化量の制限 |
| `src/baseline.py` | 逆運動学と関節角誤差から速度を作る非RL比較 |
| `src/train.py` | PPO学習、検証ケースによる最良モデル選択、保存・再読込 |
| `src/evaluate.py` | 同じ未学習ケースで全制御器を比較、CSVと図を保存 |
| `src/llm_parser.py` | Ollama構造化出力、Pydantic検証、carefulへのfallback |
| `src/demo.py` | 指示を1回解釈して動画を撮る |
| `src/render_video.py` | MuJoCo映像に字幕を重ねてH.264を保存・再検査 |
| `src/scenarios.py` | 学習・モデル選択と切り離した固定評価ケースを保存 |
| `config.yaml` | 環境、PPO、クラウド費用上限の設定 |
| `tests/` | 環境API、再現性、接触、制御制限、異常LLM出力、集計の検査 |

## 制御を理解するための順番

1. `baseline.py` の `target - qpos` は「どちらへどれだけ関節を回したいか」や。
   これを角速度へ変換して、最初に学習なしで物理系と動画を検証する。
2. PPOは、その計算を**状態から行動を出すニューラルネット**で置き換える。
   学習ステップ数と成功率を分けて見て、保存された重みを再ロードして評価する。
3. LLMは関節値を作らず、動作条件を2種類から選ぶ。
   `careful/fast` の差は後段のルールによる速度上限と指令変化量制限や。
   **2種類の速度方策を学習した、とは説明せえへん。**

観測は14次元のfloat32や。
`sin(q1,q2), cos(q1,q2), qvel(2), goal-tip(2), obstacle-tip(2), radius, exists, previous_command(2)`
の順。速度指令を徐々に変えるフィルタには履歴が必要やから、前回指令も観測へ入れてる。
行動は2次元の `[-1,1]`。長さ0.35m + 0.30m、物理刻み0.01秒、4回繰返しで1行動0.04秒や。

報酬は `2*(前回距離-今回距離)-0.01` に、成功+10、接触-10、状態異常-5を加える。
成功は目標から0.05m未満かつ接触・異常なし。最大200行動。
成功・接触・異常は `terminated`、通常の時間切れは `truncated` と分けてる。
全リンクと障害物の接触をMuJoCoの各物理刻みで調べる。

| モード | 角速度指令の上限 | 指令変化量の上限（1行動） |
|---|---:|---:|
| careful | 0.525 rad/s | 0.040 rad/s |
| fast / standard | 1.200 rad/s | 0.160 rad/s |

ハード上限は1.5 rad/s、指令の加速度上限4 rad/s²。実際の速度も監視し、超過時は異常終了する。
これは衝突回避を保証するフィルタではない。実機用の安全装置や物理的な加速度保証でもない。

## 学習・比較

```powershell
# P0が動いてから、1000 → 10000 → 100000 の順で確認
python -m src.train --timesteps 1000 --seed 42 --output checkpoints/smoke
python -m src.train --timesteps 10000 --seed 42 --output checkpoints/fixed
python -m src.train --timesteps 100000 --seed 42 --task random --n-envs 4 --resume checkpoints/fixed/best_model.zip --output checkpoints

# heldout_eval_cases.json は同梱済み。学習や最良モデル選択に使わへん。
python -m src.evaluate --model checkpoints/best_model.zip --initial-model checkpoints/fixed/initial_model.zip --episodes 100
```

4環境は既定で `DummyVecEnv` による逐次実行。プロセス並列にする場合だけ `--subproc` を追加する。
PPO + MLPはCPU、PyTorchの演算スレッドは1。学習の最後はロールアウト単位に丸められるので、
指定1000ステップに対して実際は1024、指定10000に対して10240になる場合がある。
`training.json` の `actual_timesteps` と `wall_time_s` を実測値として使う。
各学習コマンドの出力先を分けると、重み・検証履歴の上書きを防げるで。
`--resume` は重み・optimizerを読み込んで追加学習し、追加分のステップ数を新しく数える。

`best_model.zip` は**検証用10ケース（seed 708000から）**で選ぶ。
評価用100ケース（seed 20261023から）は別ファイル。
`results/episode_results.csv` に全試行、`summary.csv` に集計、`comparison.png` に実測図を保存する。
完了時間は**成功した試行だけ**の平均。成功ゼロなら空欄で、0秒と扱わへん。
`mean_abs_action_delta` は正規化行動ではなく、フィルタ後の角速度指令差の絶対値を
各関節・各行動で平均した値（rad/s per action）や。

`random / ik / ppo_untrained / ppo_standard / ppo_careful / ppo_fast` を比較する。
このCLIのモード選択は明示指定で、実LLMを使った比較とは区別して記録する。
`standard` と `fast` の定数は同じなので、同一方策なら結果も同じになる。
障害物なしの衝突率0%から障害物回避性能は主張できへん。

障害物付きケースは以下で別に生成・評価できる。現状の生成器は、既知の関節空間経路の
全リンクが障害物から離れている保守的な配置だけを採用する。難しい回避能力の評価には不足する。

```powershell
python -m src.scenarios --task obstacle --count 30 --seed 20261101 --output obstacle_eval_cases.json
python -m src.evaluate --cases obstacle_eval_cases.json --episodes 30 --output results/obstacle
```

## 日本語入力とデモ

Ollamaのサービスとモデルの準備が必要や。起動してへん場合やJSON検証失敗時は `careful` にする。
fallbackやキーワード処理を「LLM推論成功」として数えへん。

```powershell
ollama serve
# 別の端末で、モデルのサイズとライセンスを確認してから
ollama pull qwen2.5:3b
python -m src.llm_parser --evaluate --output results/instruction_ollama.json
python -m src.llm_parser --evaluate --backend keyword --output results/instruction_keyword.json
python -m src.demo --text '精密部品だから慎重に移動して' --output videos/careful.mp4
python -m src.demo --text '急いで素早く移動して' --output videos/fast.mp4
```

3Bモデルは約1.9GB・Qwen RESEARCH LICENSE、1.5Bは約986MB・Apache-2.0。
小さいモデルに切り替えるなら `ollama pull qwen2.5:1.5b` を実行し、
CLIで `--llm-model qwen2.5:1.5b`（demo）/ `--model qwen2.5:1.5b`（llm_parser）を指定する。
24指示の一致率とfallback件数を別々に保存する。テスト中のFakeClientはAPI契約の検査用で、実推論ではない。

今回の実測には、メモリ制約からQwen2.5:1.5Bを使った。Ollama 0.40.9の公式ポータブル版を
`.cache/ollama`、モデルを `.cache/ollama-models` に置いている。システムのPATHは変更してへん。
この作業環境で同じモデルを再利用する場合は次の手順や。新規環境なら最初のコマンドで
公式ZIPを取得し、CPU用ファイルだけ展開する（GitHubから約1.6GBのダウンロードが必要）。

```powershell
# 新規環境だけ必要
python tools/prepare_ollama.py
# ローカル専用のサーバーを開始
powershell -ExecutionPolicy Bypass -File tools/ollama_local.ps1 Start
.\.cache\ollama\ollama.exe pull qwen2.5:1.5b
python -m src.llm_parser --evaluate --model qwen2.5:1.5b --output results/instruction_ollama.json
python -m src.demo --text '精密部品だから慎重に移動して' --llm-model qwen2.5:1.5b --output videos/careful.mp4
python -m src.demo --text '急いで素早く移動して' --llm-model qwen2.5:1.5b --output videos/fast.mp4
python -m src.evaluate --episodes 100 --use-llm --llm-model qwen2.5:1.5b
# 作業後に、このプロジェクトが起動したサーバーだけを止める
powershell -ExecutionPolicy Bypass -File tools/ollama_local.ps1 Stop
```

作業サンドボックスではlocalhostへの通信にも許可が必要な場合がある。
`source=fallback` のときは通信エラーを確認し、分類成功と取り違えへんようにする。
CPU推論を再現できるよう、APIには `num_gpu=0, num_ctx=2048, num_thread=4` を渡している。

```powershell
# LLMを使わず、動作条件だけを比較する方法
python -m src.demo --mode careful --output videos/careful-explicit.mp4
python -m src.demo --mode fast --output videos/fast-explicit.mp4
# 自然言語の単純キーワード比較（字幕とログにもkeywordと出る）
python -m src.demo --text 'ゆっくり動いて' --parser keyword --output videos/keyword.mp4
python -m src.render_video --comparison --inputs videos/baseline.mp4 videos/careful.mp4 videos/fast.mp4 --output videos/comparison.mp4
```

字幕の `interpreter=ollama / fallback / keyword / explicit_mode` を見れば、どうモードを決めたか分かる。
合成動画は入力を順番につなぐ。1〜3分の条件は完成ファイルの `duration_s` を確認する。
すべての動画の隣に、フレーム数・尺・実試行の結果をJSONで保存する。
保存直後にMP4を再読込し、フレーム数と先頭・末尾のデコードを検査する。

## エラーの調べ方

- **XMLを開けへん**: Windowsの日本語パス対策としてPythonでUTF-8読込後 `from_xml_string` に渡している。
- **描画できへん**: Linuxなら `MUJOCO_GL=egl`、CPU描画用設定が必要な環境ではMuJoCo公式資料を確認する。
  緊急時は各動画コマンドに `--renderer 2d`。物理計算はMuJoCoのままで、描画だけ実状態の2D表示になる。
- **学習が改善せえへん**: `validation.json` の成功率と最小距離を見る。まずfixedで確認し、
  次にrandomへ進む。検証ケースで調整して、heldoutの成績を見ながら報酬を調整しない。
- **LLMに接続できへん**: `ollama list` と `python -m src.llm_parser --text '急いで'` を確認する。
  `source=fallback` はLLM未成功や。モデル名・起動ポートを直すか、明示モードで実験を続ける。

## 4日間の作業と費用

| 日程（JST） | 優先事項 | 証拠 |
|---|---|---|
| 10/23 Day 1 | MuJoCo、環境検査、非RL動画 | pytest、videos/baseline.mp4 |
| 10/24 Day 2 | PPOと保存復元、未学習ケース評価 | checkpoints、CSV、TensorBoard |
| 10/25 Day 3 | LLMの実推論、動作条件変更、障害物評価 | 指示結果JSON、careful/fast動画 |
| 10/26 Day 4 | 1〜3分の最終動画、レポート、再現確認 | comparison.mp4、report.md |

Day 1失敗時は障害物・LLMを止めて単純到達へ絞る。
Day 2でPPOが成立しなければ固定目標の学習と原因分析へ集中する。
Day 3でLLMが動かなければ明示モードを使い、LLM未完成と記す。

クラウドはまだ起動してへん。L40を第一候補とし、指示書の参考表示は $0.82/h。
**起動時に表示単価が $1.00/h 未満か再確認**し、GPU以外の料金も記録する。
このリポジトリには有料インスタンスの自動作成処理はない。
利用する場合はユーザーの管理下で起動し、開始・終了時刻と停止確認を残す。

## 出典・使用ライブラリ

新規の2リンクXML、環境、フィルタ、IK、実験・動画コードを実装した。既存のロボット課題を丸写ししていない。
利用APIは [MuJoCo公式Python](https://mujoco.readthedocs.io/en/latest/python.html)、
[Gymnasiumカスタム環境](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)、
[SB3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)、
[Ollama構造化出力](https://ollama.com/blog/structured-outputs) を参照した。
モデルは [Qwen2.5:3B](https://ollama.com/library/qwen2.5:3b) と
[Qwenライセンス](https://ollama.com/library/qwen2.5:3b/blobs/b5c0e5cf74cf) を確認する。
実際の依存バージョンは `requirements-lock.txt` に保存する。

動画・重みは大きいためGitではignoreしている。提出時には実ファイルを別添するか、GitHub Releases等に置く。
公開・提出・YouTubeアップロードはまだ行っていない。最終的な実測値と残課題は `report.md` を参照する。

## 保存された成果と整合性確認

- `videos/comparison.mp4`: 実測の学習曲線・失敗例・IK・実LLMによるモード切替・全体指標を含む115.08秒の動画。
- `checkpoints/best_model.zip`: 検証ケースで選んだPPO。最終ステップの重みは `final_model.zip`。
- `results/episode_results.csv`: 8方式×100試行。`results/obstacle/` に別の30試行比較も保存。
- `results/instruction_ollama.json`: 実LLMの24指示評価。キーワード比較は `instruction_keyword.json`。
- `results/runtime.json`: Ollamaの版、モデルdigest、CPU実行（VRAM使用0）、停止確認。
- `results/verification.json`: テスト数、動画の尺・デコード、CSV再集計、モデルhashの確認記録。

レポートと比較動画を作り直すときは次の順番や。`build_report.py` は保存済みの実測結果だけを読み、
学習曲線を描く。最後の再実行で合成動画の情報もレポートへ反映する。

```powershell
python tools/build_report.py
python -m src.render_video --comparison --inputs videos/random.mp4 videos/baseline.mp4 videos/careful.mp4 videos/fast.mp4 --output videos/comparison.mp4
python tools/build_report.py
python tools/verify_project.py
```

## 続き: 障害物回避と3 seedの比較

前回の `obstacle` は、直接の関節角補間でも通れる配置だけを採用していた。
今回は `src/detour.py` に、**直接追従では接触し、肘を畳むと通れる課題**を追加した。
リンク全体の距離を計算し、0.01 rad以下の刻みで迂回経路を確認する。
`oracle` はこの経路を知る参照制御で、PPOとは分けて扱う。

学習を成立させるため、最初は既知の経路上のゴール付近から始め、開始姿勢を徐々に広げた。
約17万ステップで本来の初期姿勢だけに切り替わる、20万ステップ基準のカリキュラムや。
**経路は開始姿勢を選ぶ教材。観測やPPO推論時の制御には渡さへん。**
状態14次元と行動の意味は前回と同じで、報酬に `-0.3 * distance` を追加した。
ゴール手前で止まるより、最後まで近づく行動を促す狙いや。

| 分割 | 配置数 | seed範囲 | 用途 |
|---|---:|---|---|
| data/detour_train.json | 512 | 510000から | 学習と開始姿勢のカリキュラム |
| data/detour_validation.json | 30 | 630000から | 最良チェックポイントの選択 |
| data/detour_test.json | 100 | 910000から | 凍結した重みの最終比較 |

seed 42 / 123 / 456のPPOを、各々ランダム初期重みから303,104ステップ学習した。
この3つは前回PPOからの追加学習ではなく、独立した初期化や。
最良のチェックポイントだけを検証データで選び、最終100配置は学習・モデル選択に使わへん。
同じ100配置を3モデルで使うので、300配置が独立とは数えず、学習seed間の平均±標本標準偏差を示す。

```powershell
# 同梱データ・モデルで、学習せずにすぐ動かす
python -m src.demo --mode careful --model checkpoints/detour_seed42/best_model.zip --cases data/detour_test.json --episodes 3 --output videos/detour-recheck.mp4

# 学習から再現する。完了済みのtraining.jsonがあるseedは保持してスキップする。
python tools/run_detour_seeds.py --seeds 42 123 456

# 生データと集計を再評価する。新しい出力先を指定して元の結果を保持する。
python -m src.benchmark_detour --models checkpoints/detour_seed42/best_model.zip checkpoints/detour_seed123/best_model.zip checkpoints/detour_seed456/best_model.zip --seeds 42 123 456 --output results/detour-recheck
```

最初から学習し直す場合は、既存の完了モデルを保存して別の出力先に `src.train` を実行する。
例えば `python -m src.train --task detour --timesteps 300000 --curriculum-steps 200000 --validation-every 25000 --seed 42 --n-envs 4 --output checkpoints/reproduction42` や。
`run_detour_seeds.py` は再開用ではないため、中断した出力先の学習は初期重みからやり直す。

最終評価の `results/detour/episode_results.csv` は13方式×100試行。
ランダム、直接IK、経路オラクル、前回PPO、3学習seed×標準・慎重・高速を含む。
モードは明示指定して制御性能を測り、実LLMでの指示解釈はデモ動画で別に確認する。
`protocol.json` に配置・モデルのhashを保存し、`seed_summary.csv` に各seedの結果、
`seed_aggregate.csv` にseed間の集計を保存する。

```powershell
# 実LLM付きの新しいPPO動画を再生成
powershell -ExecutionPolicy Bypass -File tools/ollama_local.ps1 Start
python tools/record_detour_clips.py
powershell -ExecutionPolicy Bypass -File tools/ollama_local.ps1 Stop
python tools/build_followup_report.py
python tools/build_final_video.py
python tools/verify_final.py
python tools/package_delivery.py
```

PDF生成は `tools/export_pdf.py`。ReportLab / pypdf / Pillowを使う。
Codex同梱の文書用Pythonランタイムで実行した。普通の仮想環境から再生成する場合は、
`uv pip install --python .venv/Scripts/python.exe -r requirements-pdf.txt` でPDF用依存だけ追加する。
学習用の固定依存は `requirements-lock.txt` に保存済みで、PDF依存は別用途や。
PDFを作ったあと `tools/verify_final.py` と `tools/package_delivery.py` を実行する。

最新の整合性記録は `results/final-verification.json`。`results/verification.json` は前回の検証記録や。
配布ZIPは `output/physical-ai-delivery.zip` に、コード・データ・モデル・動画・PDFを含む。
Python/Ollama本体、LLMの重み、秘密情報、Git内部、キャッシュは同梱しない。

提出前には、OmniCampusアカウント名またはIDに応じてPDFのファイル名と動画タイトルを設定する。
GitHub公開・YouTube投稿・フォーム送信はまだ行っていない。
