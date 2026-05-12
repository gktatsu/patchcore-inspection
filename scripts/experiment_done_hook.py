"""PostToolUse フック: PatchCore 実験完了後にサマリーを表示する。

Claude Code が Bash ツールを呼ぶたびに実行される。
run_patchcore の実行でない場合は即終了する。
"""
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def _find_latest_results():
    candidates = list(RESULTS_DIR.glob("**/results.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _print_summary(results_path: Path):
    with open(results_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    mean_row = next((r for r in rows if r["Row Names"] == "Mean"), rows[-1])
    run_dir = results_path.parent

    sep = "=" * 56
    print(f"\n{sep}")
    print(f" 実験完了: {run_dir.parent.name}/{run_dir.name}")
    print(sep)

    metrics = [
        ("instance_auroc", "画像AUROC         "),
        ("full_pixel_auroc", "ピクセルAUROC     "),
        ("anomaly_pixel_auroc", "異常ピクセルAUROC "),
        ("instance_f1", "画像F1            "),
        ("train_time", "学習時間(秒)       "),
        ("fps", "FPS               "),
    ]
    for key, label in metrics:
        if key in mean_row and mean_row[key]:
            try:
                print(f"  {label} {float(mean_row[key]):>10.4f}")
            except ValueError:
                pass
    print(sep)


def main():
    try:
        raw = sys.stdin.read()
        data = json.loads(raw) if raw.strip() else {}
    except Exception:
        data = {}

    tool_input = data.get("tool_input", {})
    command = tool_input.get("command", "") if isinstance(tool_input, dict) else ""
    if "run_patchcore" not in command:
        return

    results_path = _find_latest_results()
    if results_path is None:
        return

    _print_summary(results_path)


if __name__ == "__main__":
    main()
