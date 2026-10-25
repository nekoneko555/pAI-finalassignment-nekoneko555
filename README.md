# SO-101: 日本語で動作条件を変える物理把持・運搬

SO-101のMuJoCoシミュレーションで、A地点の物体へ近づき、つかみ、持ち上げ、B地点へ運んで離す。実機や映像認識の実験ではない。

**YouTube提出用の動画は `videos/so101_submission.mp4`。** 提供されたOP・EDを本編の前後へ追加した約1分42秒のフルHD版（1920×1080、30fps）。縦横比を保ち、本編の左右へ余白を付けている。元の3動画には音声がない。本編単体は `videos/so101_final.mp4` として保存している。

配布物に同梱したOP・EDから再結合するには `python tools/add_video_bookends.py` を実行する。接続位置・入力hash・全編デコード検査は `videos/so101_submission.json` に記録する。

## レポートとコードの入口

- [提出レポート PDF](output/pdf/so101_report.pdf)
- [編集可能なWord版](output/docx/so101_report.docx)
- [物理シミュレーションと基本制御](src/so101_pickplace.py)
- [PPOの観測 行動 報酬](src/so101_task.py)
- [学習と比較評価](tools/so101_experiment.py)
- [日本語指示を使った撮影](tools/record_so101.py)
- [SO-101のテスト](tests/test_so101.py)

コードに加え、SO-101のモデルとライセンス、学習済みPPO重み、評価結果、動画を同梱する。Ollama本体とQwenのモデルは含まない。
Ollamaの初回準備は `python tools/prepare_ollama.py` を実行し、以降は下記の手順で起動する。

## 役割分担

```text
日本語 → ローカルQwen → careful/fast → 速度・指令変化量の制限
物体とBの既知座標 → IK経由姿勢 → PPO残差による5関節の速度補正 → サーボ
作業段階のルール → 指の開閉 → MuJoCoの接触と摩擦で物体を保持
```

PPOの補正は基準速度の±35%。IKの経由姿勢と指の開閉を使うハイブリッド制御であり、把持全体をゼロから学習した方式ではない。物体は自由剛体で、移動中に座標を書き換えず、溶接拘束も使わない。

carefulの速度指令上限は0.4rad/s、指令加速度上限は0.8rad/s²。standard/fastは1.2・3.0。同じ重みを使い、standardとfastは同じ定数。実機安全や物理的速度・加速度の保証ではない。

## セットアップと最短の確認

Python 3.11を使用し、リポジトリ直下で実行する。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
# 配布ZIPにはモデルを同梱。ソースのみの場合は公式固定版を取得する。
.\.venv\Scripts\python.exe tools/fetch_so101.py
.\.venv\Scripts\python.exe -m pytest tests/test_so101.py -q
.\.venv\Scripts\python.exe -m src.so101_pickplace --output videos/so101_pickplace_ik.mp4
```

最後のコマンドは学習なしの基準制御。物理的把持が成立するかを、LLMやPPOと切り離して確認できる。`--no-video`なら描画せずに検証する。`--mode careful`で低い速度条件を使える。

## 学習・評価の再現

```powershell
.\.venv\Scripts\python.exe tools/so101_experiment.py prepare
# 同名の完了済み学習は上書きしない。再学習は別seedを指定する。
.\.venv\Scripts\python.exe tools/so101_experiment.py train --seeds 42 123 456 --steps 131072
.\.venv\Scripts\python.exe tools/so101_experiment.py evaluate --seeds 42 123 456 --output results/so101/recheck
```

配布モデルでの再評価はevaluateだけでよい。出力先を変え、保存済み最終評価を上書きしない。学習128配置、検証30配置、最終評価100配置を分離する。チェックポイントは検証成功率、その次に平均時間で選択する。

観測43次元は関節の目標誤差・サーボ遅れ・速度・指令・sin/cos、8段階の作業状態、物体相対位置、モード上限。行動5次元は関節速度の補正。報酬は関節誤差改善、時間コスト、段階進行、最終成功から構成する。正確な式はsrc/so101_task.py。

## 実LLMと動画

ローカルOllamaを起動し、qwen2.5:1.5bを使う。既存の環境にモデルがあれば再取得は不要。

```powershell
.\tools\ollama_local.ps1 Start
# 初回のみ必要
$env:OLLAMA_HOST='127.0.0.1:11434'
.\.cache\ollama\ollama.exe pull qwen2.5:1.5b
.\.venv\Scripts\python.exe tools/record_so101.py
.\tools\ollama_local.ps1 Stop
```

record_so101は実LLMの返答が想定モードに一致した場合だけ撮影する。通信失敗による既定値を実LLM成功として扱わない。日本語指示、解析結果、使用モデルhash、成功判定、毎ステップの物体位置と両指接触をMP4と同名のJSON/CSVへ記録する。

保存済みの実LLM解釈でカメラ等を撮り直す場合だけ`--reuse-interpretations`を使用する。この再利用を新たな分類評価には数えない。

## 成功条件と制約

8つの作業段階を終え、両指に接触したまま物体中心が6cmを超え、Bの半径3.5cm以内へ置き、指を開いて接触を離し、物体が静止していることを確認する。浮かせたままBへ近づくだけでは成功にしない。

35g・一辺36mmの立方体、固定開始姿勢、狭いA/B配置範囲、既知の座標だけを検証する。実機転移、任意物体、画像認識、未知環境への一般化は未検証。

## 出典とモデル

SO-101の形状と力学モデルは[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101)を使用し、コミット`cda5236f98383f6fbaa4820c6b18dba892cd087f`に固定。元ファイルは改変せずassets/so101/upstreamへ保存した。Apache-2.0のLICENSEを同梱。元モデルはThe Robot StudioのSO-101記述に由来する。出典・ファイルhashはassets/so101/source.json。

作業机・物体・色付きA/B領域・制御・学習・比較・日本語の条件変更・動画は今回のコードで追加した。把持実装中にはIKの姿勢制約と指の接近位置を修正し、失敗を確認してから再実行した。初期失敗の数値を最終成功として混ぜない。

元の平面2関節版はREADME-PLANAR.mdの従来手順およびreport-final.mdに残す。SO-101の評価に平面版の成功率を流用しない。クラウドGPUは使っていない。

## 調整で分かったこと

- **届く位置と、つかめる姿勢は別。** 5関節のアームへ位置3成分と姿勢3成分をすべて指定すると、解がないことがある。今回は指を下へ向けることとロールを制約し、残りの向きを自由にしてIKを解いた。まず関節範囲と手先位置の残差を数値で調べると原因を切り分けられる。
- **指の片側は固定されている。** 開いた指の見た目の中央へ寄せると、下降時に固定指が物体を押し、閉じても持ち上がらなかった。指先・物体の座標、接触するbody、接触距離を出力し、接近位置を調整した。成功は物体高さと両指の接触で検証する。
- **学習の効果は基準と比べる。** IKだけの運搬を先に作ることで、失敗が物理的把持・経由姿勢・学習方策・言語解釈のどこから来るか調べやすくなる。PPOを追加したこと自体を改善とは扱わず、同じ未使用配置で時間と成功率を比較する。
