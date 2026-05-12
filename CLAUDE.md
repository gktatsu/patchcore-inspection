# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

本リポジトリは Roth et al. (2021) "Towards Total Recall in Industrial Anomaly Detection" ([arXiv:2106.08265](https://arxiv.org/abs/2106.08265)) の PatchCore 実装です。ImageNet 事前学習済み Backbone からパッチ特徴量を抽出し、Coreset サンプリングで圧縮したメモリバンクと最近傍探索（faiss）で異常スコアを算出します。

## Claude への作業指示

### コミット規約

- **最小単位**でコミットする（1コミット1変更）
- メッセージは**日本語**で書く
- フォーマット：

```
<prefix>: <変更の概要（1行）>

<変更の詳細（任意、箇条書き可）>
```

使用するprefix：

| prefix | 用途 |
|--------|------|
| `feat` | 新機能の追加 |
| `fix` | バグ修正 |
| `refactor` | 動作を変えないリファクタリング |
| `perf` | パフォーマンス改善 |
| `test` | テストの追加・修正 |
| `docs` | ドキュメント・コメントの変更 |
| `chore` | ビルド・設定・依存関係の変更 |
| `style` | コードフォーマットの変更（動作に影響なし） |
| `log` | ログ・デバッグ出力の変更 |
| `git` | .gitignore・git設定の変更 |

### 作業方針

- `./dataset/` および `/disk/data_area4/` への書き込みは**必ずユーザーに確認**する（設定で保護済み）
- `git push` / `git reset --hard` / `git rebase` / `git merge` / `git checkout` は**必ずユーザーに確認**する
- コード変更後は `python -m py_compile <file>` で構文チェックを行う
- テストがある場合は変更後に `PYTHONPATH=./src python3 -m pytest -v` を実行する
- **コミット前に必ずBashで動作確認を行う**（構文チェック or テスト実行 or --help確認）
- 動作確認が取れたもののみコミットする（未確認のコードはコミットしない）
- 実験の実行（`run_patchcore.py` 等）は長時間かかるため、ユーザーの明示的な指示なく自動実行しない
- エラーが出た場合、修正案を提示してからユーザーに確認を取る

### 不要なことはしない

- タスクに直接関係しないリファクタリングは行わない
- 将来の拡張を見越した抽象化は行わない
- コードにコメントを書く場合は「なぜ」が非自明な場合のみ（1行）

## 環境セットアップ

```bash
# venv を使った実行（推奨）
source .venv/bin/activate

# 環境確認
.venv/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available(), '/ devices:', torch.cuda.device_count())"
.venv/bin/python -c "import faiss; print('faiss OK, GPU support:', hasattr(faiss, 'StandardGpuResources'))"
```

## テスト

```bash
# 全テスト
PYTHONPATH=./src python3 -m pytest -v
# または
./local_run_test.sh
```

## パイプライン全体の流れ

### 1. 学習（PatchCore モデルの構築）

```bash
# venv スクリプト経由（推奨）
bash bin/run_patchcore_venv.sh

# 直接実行
PYTHONPATH=./src .venv/bin/python bin/run_patchcore.py \
  --gpu 1 --seed 0 --save_patchcore_model \
  --log_group IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 \
  --log_project MyDataset_Results \
  ./results \
  patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu \
    --pretrain_embed_dimension 1024 --target_embed_dimension 1024 \
    --anomaly_scorer_num_nn 1 --patchsize 3 \
  sampler -p 0.1 approx_greedy_coreset \
  dataset -d anomaly_dataset --resize 256 --imagesize 224 \
    mvtec ./dataset
```

### 2. 評価（学習済みモデルの評価）

```bash
PYTHONPATH=./src .venv/bin/python bin/load_and_evaluate_patchcore.py \
  --gpu 0 --seed 0 <save_folder> \
  patch_core_loader -p <model_path> --faiss_on_gpu \
  dataset -d anomaly_dataset --resize 256 --imagesize 224 \
    mvtec ./dataset
```

### 結果ファイル

実験結果は `./results/<log_project>/<log_group>/results.csv` に保存される：

| カラム | 説明 |
|--------|------|
| `instance_auroc` | 画像レベル異常検知 AUROC |
| `full_pixel_auroc` | ピクセルレベル異常局所化 AUROC（全ピクセル） |
| `anomaly_pixel_auroc` | ピクセルレベル異常局所化 AUROC（異常ピクセルのみ） |
| `instance_f1` | 画像レベル F1 スコア |
| `train_time` | 学習時間（秒） |
| `fps` | 推論速度 |

## コードアーキテクチャ

### `src/patchcore/`

- **`patchcore.py`** : メインの PatchCore クラス。特徴抽出 → パッチ埋め込み → メモリバンク構築 → 異常スコアリングを管理。
- **`backbones.py`** : WideResNet50 等の事前学習済み Backbone。`-b` オプションで指定するモデル名に対応。
- **`common.py`** : NetworkFeatureAggregator。指定レイヤ（`-le` オプション）からパッチ特徴量を抽出する。
- **`sampler.py`** : Coreset サンプリング実装（approx_greedy_coreset）。`-p` オプションでサブサンプリング率を指定。
- **`metrics.py`** : AUROC・AUPRC・F1 等の評価指標計算。
- **`datasets/mvtec.py`** : MVTec AD 形式のデータセットローダー。カスタムデータセットを追加する場合はここを参考にする。
- **`utils.py`** : ストレージ管理・デバイス設定等のユーティリティ。

### `bin/`

- **`run_patchcore.py`** : 学習エントリポイント。click の chain group で `patch_core`・`sampler`・`dataset` サブコマンドを連結する。
- **`run_patchcore_venv.sh`** : venv 環境でコンテナなしに実行するシェルスクリプト。
- **`load_and_evaluate_patchcore.py`** : 保存済みモデルをロードして評価するエントリポイント。

### `scripts/`

- **`extract_rar.py`** : RAR 形式の異常データセットを解凍するスクリプト。
- **`experiment_done_hook.py`** : 実験完了時に results.csv のサマリーを表示する PostToolUse フック（自動実行）。

## データ配置

```
./dataset/                                       ローカルの学習・評価データ（保護対象）
./results/                                       実験結果・学習済みモデル
/disk/data_area4/t-hori/2025/patchcore/          外部ストレージのデータ（保護対象）
  anomaly_dataset/
    ground_truth/abnormal/                       異常画像のグラウンドトゥルース
```

MVTec AD 形式のディレクトリ構造：

```
./dataset/<カテゴリ>/
  train/good/            正常サンプル（学習用）
  test/good/             正常サンプル（評価用）
  test/<欠陥名>/         異常サンプル（評価用）
  ground_truth/<欠陥名>/ マスク画像（PNG、異常ピクセルが白）
```
