#!/usr/bin/env python3
"""
継続学習実験: PatchCore
各ラウンドで異なる N_PER_ROUND 枚を学習し、コアセットをマージしていく。
"""
import os
import sys
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "src"))

import patchcore.backbones
import patchcore.common
import patchcore.datasets.mvtec as mvtec
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler

# ---------- 設定 ----------
DATASET_PATH = os.path.join(SCRIPT_DIR, "..", "dataset")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "..", "results", "ContinualLearning")
N_PER_ROUND = 1000
PERCENTAGE = 0.01
N_ROUNDS = 10
MEMORY_LIMIT_MB = 32_000  # 超えたら停止
GPU = 0
SEED = 0
RESIZE = 256
IMAGESIZE = 224
BACKBONE = "wideresnet50"
LAYERS = ["layer2", "layer3"]
PRETRAIN_DIM = 1024
TARGET_DIM = 1024
PATCHSIZE = 3
N_NN = 1
BATCH_SIZE = 8
NUM_WORKERS = 4
FAISS_ON_GPU = True
# --------------------------


def get_memory_mb() -> float:
    """プロセスの RSS メモリ使用量 (MB) を返す。/proc/self/status から読む。"""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB → MB
    except OSError:
        pass
    return 0.0


def save_plot(results: list, plot_path: str) -> None:
    rounds = [r["round"] for r in results]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Continual Learning — PatchCore  N={N_PER_ROUND}, P={PERCENTAGE}")

    ax = axes[0, 0]
    ax.plot(rounds, [r["instance_auroc"] for r in results], "b-o")
    ax.set_title("Instance AUROC")
    ax.set_xlabel("Round")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1)
    ax.grid(True)

    ax = axes[0, 1]
    ax.plot(rounds, [r["train_time"] for r in results], "r-o")
    ax.set_title("Train Time per Round (s)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Seconds")
    ax.grid(True)

    ax = axes[1, 0]
    ax.plot(rounds, [r["memory_mb"] for r in results], "g-o")
    ax.set_title("Process Memory (MB)")
    ax.set_xlabel("Round")
    ax.set_ylabel("MB")
    ax.axhline(MEMORY_LIMIT_MB, color="red", linestyle="--", linewidth=0.8, label="limit")
    ax.legend(fontsize=8)
    ax.grid(True)

    ax = axes[1, 1]
    ax.plot(rounds, [r["memory_bank_size"] for r in results], "m-o")
    ax.set_title("Memory Bank Size (# vectors)")
    ax.set_xlabel("Round")
    ax.set_ylabel("# Vectors")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] {plot_path}")


def main():
    os.makedirs(RESULTS_PATH, exist_ok=True)

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- 学習データ全体をロード ---
    full_train = mvtec.MVTecDataset(
        DATASET_PATH,
        classname="anomaly_dataset",
        resize=RESIZE,
        imagesize=IMAGESIZE,
        split=mvtec.DatasetSplit.TRAIN,
        seed=SEED,
    )
    all_indices = np.arange(len(full_train))
    rng.shuffle(all_indices)

    needed = N_ROUNDS * N_PER_ROUND
    if len(all_indices) < needed:
        raise ValueError(
            f"学習画像が不足: {len(all_indices)} 枚 < 必要 {needed} 枚"
        )
    print(f"学習画像: {len(all_indices)} 枚  (使用: {needed} 枚)")

    # --- テストセット（固定） ---
    test_dataset = mvtec.MVTecDataset(
        DATASET_PATH,
        classname="anomaly_dataset",
        resize=RESIZE,
        imagesize=IMAGESIZE,
        split=mvtec.DatasetSplit.TEST,
        seed=SEED,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    print(f"テスト画像: {len(test_dataset)} 枚")

    # --- PatchCore 初期化 ---
    backbone = patchcore.backbones.load(BACKBONE)
    backbone.name = BACKBONE
    backbone.seed = None

    nn_method = patchcore.common.FaissNN(FAISS_ON_GPU, NUM_WORKERS)

    pc = patchcore.patchcore.PatchCore(device)
    pc.load(
        backbone=backbone,
        layers_to_extract_from=LAYERS,
        device=device,
        input_shape=(3, IMAGESIZE, IMAGESIZE),
        pretrain_embed_dimension=PRETRAIN_DIM,
        target_embed_dimension=TARGET_DIM,
        patchsize=PATCHSIZE,
        featuresampler=patchcore.sampler.ApproximateGreedyCoresetSampler(
            PERCENTAGE, device
        ),
        anomaly_scorer_num_nn=N_NN,
        nn_method=nn_method,
    )

    results = []
    csv_path = os.path.join(RESULTS_PATH, "results.csv")
    plot_path = os.path.join(RESULTS_PATH, "progress.png")

    with open(csv_path, "w") as f:
        f.write("round,instance_auroc,train_time,memory_mb,memory_bank_size\n")

    print(f"\n{N_ROUNDS} ラウンド開始  N={N_PER_ROUND}  P={PERCENTAGE}")
    print(f"メモリ上限: {MEMORY_LIMIT_MB} MB")
    print(f"結果: {RESULTS_PATH}\n")

    for round_idx in range(N_ROUNDS):
        print(f"{'='*52}")
        print(f"  Round {round_idx + 1} / {N_ROUNDS}")
        print(f"{'='*52}")

        # 開始前メモリチェック
        mem_before = get_memory_mb()
        print(f"  メモリ (開始前): {mem_before:.0f} MB")
        if mem_before > MEMORY_LIMIT_MB:
            print(f"  !! メモリ上限 ({MEMORY_LIMIT_MB} MB) 超過 → 停止")
            break

        # このラウンドの学習画像インデックス
        start = round_idx * N_PER_ROUND
        round_indices = all_indices[start : start + N_PER_ROUND]
        round_loader = DataLoader(
            Subset(full_train, round_indices.tolist()),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

        t0 = time.time()
        try:
            # パッチ特徴量の抽出
            pc.forward_modules.eval()
            new_features_list = []
            with torch.no_grad():
                for batch in round_loader:
                    imgs = batch["image"].to(torch.float).to(device)
                    new_features_list.append(pc._embed(imgs))
            new_features = np.concatenate(new_features_list, axis=0).astype(np.float32)

            # コアセットサンプリング
            new_coreset = pc.featuresampler.run(new_features).astype(np.float32)
            print(f"  新規コアセット: {len(new_coreset)} ベクトル")

            # 既存メモリバンクとマージ
            if round_idx == 0:
                merged = new_coreset
            else:
                old = pc.anomaly_scorer.detection_features.astype(np.float32)
                merged = np.concatenate([old, new_coreset], axis=0)
            print(f"  マージ後バンク: {len(merged)} ベクトル")

            # アノマリースコアラーを再フィット
            pc.anomaly_scorer.fit(detection_features=[merged])

        except MemoryError:
            print(f"  !! MemoryError → Round {round_idx + 1} で停止")
            break

        train_time = time.time() - t0

        # テストセットで評価
        scores, _, labels_gt, _ = pc.predict(test_loader)
        metrics_result = patchcore.metrics.compute_imagewise_retrieval_metrics(
            scores, labels_gt
        )
        instance_auroc = metrics_result["auroc"]

        mem_after = get_memory_mb()
        bank_size = len(pc.anomaly_scorer.detection_features)

        result = {
            "round": round_idx + 1,
            "instance_auroc": instance_auroc,
            "train_time": train_time,
            "memory_mb": mem_after,
            "memory_bank_size": bank_size,
        }
        results.append(result)

        print(
            f"  AUROC={instance_auroc:.4f}  "
            f"train_time={train_time:.1f}s  "
            f"memory={mem_after:.0f}MB"
        )

        # チェックポイント保存
        ckpt_dir = os.path.join(RESULTS_PATH, f"round_{round_idx+1:02d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        pc.save_to_path(ckpt_dir)
        # detection_features を別途保存（継続実行のため）
        feat_file = os.path.join(ckpt_dir, "detection_features.pkl")
        with open(feat_file, "wb") as fh:
            pickle.dump(pc.anomaly_scorer.detection_features, fh, pickle.HIGHEST_PROTOCOL)
        print(f"  [ckpt] {ckpt_dir}")

        # CSV 追記
        with open(csv_path, "a") as f:
            f.write(
                f"{result['round']},{instance_auroc:.6f},"
                f"{train_time:.2f},{mem_after:.1f},{bank_size}\n"
            )

        # グラフ更新（即時保存）
        save_plot(results, plot_path)

    print(f"\n完了  CSV: {csv_path}  グラフ: {plot_path}")


if __name__ == "__main__":
    main()
