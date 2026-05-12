# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

本リポジトリは論文 [Pose Priors from Language Models (arXiv:2405.03689)](https://arxiv.org/abs/2405.03689) の実装です。GPT-4V で画像から接触制約テーブルを生成し、それを PyTorch 損失関数に変換して SMPL-X ボディモデルの 3D 姿勢最適化を行うパイプラインです。

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

### 作業方針

- `datasets/` への書き込みは**必ずユーザーに確認**する（設定で保護済み）
- `git push` / `git reset --hard` / `git rebase` / `git merge` / `git checkout` は**必ずユーザーに確認**する
- コード変更後は `python -m py_compile <file>` で構文チェックを行う
- テストがある場合は変更後に `python -m unittest discover tests/` を実行する
- **コミット前に必ずBashで動作確認を行う**（構文チェック or テスト実行 or --help確認）
- 動作確認が取れたもののみコミットする（未確認のコードはコミットしない）
- 実験の実行（`main.py`）は長時間かかるため、ユーザーの明示的な指示なく自動実行しない
- エラーが出た場合、修正案を提示してからユーザーに確認を取る

### 不要なことはしない

- タスクに直接関係しないリファクタリングは行わない
- 将来の拡張を見越した抽象化は行わない
- コードにコメントを書く場合は「なぜ」が非自明な場合のみ（1行）

## 環境セットアップ

```bash
tar -xvzf prosepose_env.tar.gz
source prosepose_env/bin/activate
# または
source .venv/bin/activate
```

## テスト

pytest は未インストール。unittest を使う：

```bash
# 全テスト
python -m unittest discover tests/

# 単一モジュール
python -m unittest tests.test_convert_contact_constraints
```

## パイプライン全体の流れ

### 1. 画像 → GPT 制約テーブル生成
```bash
python gpt_table.py --vision --keys-path datasets/processed/Hi4D/val_keys.json \
    --output-path gpt4v_hi4d_val/table_20samples.json --num-samples 20 \
    --prompt-choice 2 --images-dir datasets/processed/Hi4D/cropped_images \
    --gpt-model-name gpt-4-vision-preview --ext jpg
```

### 2. テーブル → カスタム損失関数 JSON に変換
```bash
python convert_tables_to_programs.py \
    --input-path gpt4v_hi4d_val/table_20samples.json \
    --output-path gpt4v_hi4d_val/programs_from_prompt2.json --suffix 0
```

### 3. 最適化実行（従来方式）
```bash
python -m llib.methods.hhcs_optimization.main \
    --exp-cfg llib/methods/hhcs_optimization/configs/prosepose.yaml \
    --start-idx 0 \
    --exp-opts logging.base_folder=hi4d_val/prosepose_gpt4v/optimization \
               logging.run=bev_gpt4v_prompt2_20samples \
               datasets.train_names=[] datasets.train_composition=[] \
               datasets.val_names="['hi4d']" datasets.test_names=[]
```

### 3'. 最適化実行（LossSpec 方式）
```bash
python -m llib.methods.hhcs_optimization.main \
    --exp-cfg llib/methods/hhcs_optimization/configs/prosepose.yaml \
    --loss-spec loss_specs/sample.json \
    --vertex-list-map loss_specs/vertex_list_map_sample.json \
    --start-idx 0
```

### 4. 評価
```bash
python -m llib.methods.hhcs_optimization.evaluation.hi4d_eval \
    --exp-cfg llib/methods/hhcs_optimization/evaluation/hi4d_eval.yaml \
    --predictions-folder PATH_TO_METHOD_DIR --eval-split val --print_result \
    --backoff-predictions-folder PATH_TO_HEURISTIC_02_DIR
```

### 5. 結果比較
```bash
python scripts/compare_metrics.py prev_metrics.json curr_metrics.json \
    --label1 "実験A" --label2 "実験B"
```

### 6. ロス履歴の可視化
```bash
python scripts/visualize_loss_history.py \
    --loss-history results/img.loss_history.json --log-scale
```

## コードアーキテクチャ

### 最適化パイプライン（`llib/methods/hhcs_optimization/`）

- **`main.py`** : エントリポイント。設定読み込み → データセット構築 → サンプルごとに `process_item()` を呼び出す。出力は `.pkl`（パラメータ）と `.png`（可視化）。`DEBUG_IMG_NAMES` リストに画像名を入れると特定サンプルのみデバッグ処理できる。BEV 推定が異常（h2 の z < 1.0）な場合は自動スキップ。
- **`fit_module.py`** (`HHCSOpti`) : 3 ステージ最適化を管理。`fit()` は multi-start（Y 軸 180° フリップの組み合わせ試行）を行い最良候補を選択。内部で `_fit_single()` を呼ぶ：
  - Stage 0 : `global_orient` + `transl` のみ最適化（接触のみ）
  - Stage 1 : 全パラメータ（betas / scale 含む）最適化
  - Stage 2 : `body_pose` + `global_orient` + `transl`（betas 固定で contact 強化）
- **`loss_module.py`** (`HHCOptiLoss`) : 各ステージの損失重みを `set_weights(stage)` で切り替える。損失は `keypoint2d`、`pose_prior`、`shape_prior`、`custom`（GPT 由来）、`hhc_contact_general` の 5 種。

### 設定システム（OmegaConf + dataclass）

- **`llib/defaults/main.py`** : 全設定のデフォルト値（structured config）。
- `--exp-cfg` でベース yaml を指定し、`--exp-opts` で個別項目をオーバーライドする方式。
- 各設定キーは `cfg.datasets.*`、`cfg.model.optimization.*`、`cfg.logging.*` の階層構造。

### データセット層（`llib/data/`）

- **`single_optimization.py`** (`SingleOptiDataset`) : 最適化用の基底データセット。`load_data()` でデータセット名に応じた前処理クラス（FlickrCI3D / CHI3D / Hi4D / MOYO）を動的にロード。
- カスタム損失の読み込みは 2 方式：
  - **従来方式** : `dataset_cfg.custom_losses_path` から JSON を読み込み（GPT 出力を Python コードに変換したもの）
  - **LossSpec 方式** : `dataset_cfg.loss_spec_path` + `vertex_list_map_path` から `loss_loader.py` 経由でロード

### LossSpec 方式（`loss_loader.py`）

`loss_specs/sample.json` の `items[].constraints` を `touch` タイプの最小距離損失に変換する。`vertex_list_map`（部位名 → 頂点インデックスリスト）と組み合わせて使用。従来方式と互いに排他（`--loss-spec` 指定の有無で分岐）。

### ボディモデル（SMPL-X）

`essentials/` 配下にモデルファイルが必要。`llib/bodymodels/build.py` で構築。バッチサイズ 1 で人物 h1/h2 を別々にインスタンス化し、パラメータを `nn.Parameter` として最適化する。

### 可視化出力

最適化後の `.png` には行ごとに「Initial（BEV）」「After Stage 0」「Optimized」「Ground Truth」の 4 パターン＋各視点（front/45°/90°/135°/bird eye）を並べる。接触制約テーブルの画像も末尾に追加される（`scripts/visualize_loss_history.py` の `_draw_contact_table` を利用）。

`scripts/compare_metrics.py` は日本語フォントが必要なため **Docker 内では文字化けする**。ローカル環境で実行すること。

## W&B ロギング

```bash
--exp-opts logging.logger=wandb logging.project_name=ProsePose_Hi4D
```

API キーは `.wandb/api.txt`（1 行目のみ）。TensorBoard に戻す場合は `logging.logger=tensorboard`。

## データセット配置

```
datasets/
  original/   FlickrCI3D_Signatures / CHI3D / Hi4D / moyo_cam05_centerframe
  processed/  同上（前処理済み pkl, keypoints など）
```

前処理は `prepare_flickrci3d_images.py` / `prepare_chi3d_images.py` / `prepare_hi4d_images.py` / `prepare_moyo_images.py`。

Ubuntu で nltk の SSL エラーが出た場合：
```bash
sudo update-ca-certificates --fresh
export SSL_CERT_DIR=/etc/ssl/certs/
```
