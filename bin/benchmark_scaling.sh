#!/bin/bash
# スケーリングベンチマーク：画像枚数とサブサンプリング率を変えて train_time / memory_mb を計測する
#
# 使い方:
#   bash bin/benchmark_scaling.sh [GPU番号]
#
# 結果: ./results/Benchmark_Results/<log_group>/results.csv
#
# 実行時間の目安（環境依存）:
#   1条件あたり数秒〜数分。全条件（14条件）で30〜60分程度。

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && cd .. && pwd)"
cd "$SCRIPT_DIR"

GPU="${1:-1}"

SITE_PACKAGES="$SCRIPT_DIR/.venv/lib/python3.10/site-packages"
PYTHON="env -u PYTHONHOME HOME=/home2/t-hori PYTHONPATH=$SCRIPT_DIR/src:$SITE_PACKAGES /usr/bin/python3.10"

# ベンチマーク対象
IMAGE_COUNTS=(100 500 1000 2000 5000 10000 20000)
PERCENTAGES=(0.1 0.01)

SUMMARY_FILE="$SCRIPT_DIR/results/Benchmark_Results/summary.csv"
mkdir -p "$(dirname "$SUMMARY_FILE")"
echo "max_train_images,percentage,train_time,memory_mb" > "$SUMMARY_FILE"

TIMEOUT_SEC=14400  # 4時間

for N in "${IMAGE_COUNTS[@]}"; do
  for P in "${PERCENTAGES[@]}"; do
    # パーセント表記（0.1→P10, 0.01→P01）
    P_TAG=$(echo "$P" | sed 's/0\./P/' | sed 's/\.//g')
    LOG_GROUP="Bench_N${N}_${P_TAG}"
    echo "=============================="
    echo "  N=$N  p=$P  log_group=$LOG_GROUP"
    echo "=============================="

    if timeout "$TIMEOUT_SEC" $PYTHON bin/run_patchcore.py \
      --gpu "$GPU" --seed 0 \
      --log_group "$LOG_GROUP" \
      --log_project Benchmark_Results \
      ./results \
      patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu \
        --pretrain_embed_dimension 1024 --target_embed_dimension 1024 \
        --anomaly_scorer_num_nn 1 --patchsize 3 \
      sampler -p "$P" approx_greedy_coreset \
      dataset -d anomaly_dataset --resize 256 --imagesize 224 \
        --max_train_images "$N" \
        mvtec ./dataset 2>&1; then

      # results.csv から train_time と memory_mb を抽出してサマリーに追記
      RESULT_CSV=$(find "$SCRIPT_DIR/results/Benchmark_Results/$LOG_GROUP" \
        -name "results.csv" 2>/dev/null | head -1)
      if [ -n "$RESULT_CSV" ]; then
        TRAIN_TIME=$(awk -F',' 'NR==2{print $11}' "$RESULT_CSV")
        MEMORY_MB=$(awk -F',' 'NR==2{print $13}' "$RESULT_CSV")
        echo "$N,$P,$TRAIN_TIME,$MEMORY_MB" >> "$SUMMARY_FILE"
        echo "  -> train_time=${TRAIN_TIME}s  memory=${MEMORY_MB}MB"
      fi
    else
      echo "  -> タイムアウト（${TIMEOUT_SEC}秒超過）: N=$N p=$P をスキップ"
      echo "$N,$P,TIMEOUT,TIMEOUT" >> "$SUMMARY_FILE"
    fi

    echo ""
  done
done

echo "=============================="
echo "ベンチマーク完了"
echo "サマリー: $SUMMARY_FILE"
echo "=============================="
cat "$SUMMARY_FILE"
