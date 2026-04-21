import csv
import os


def summarize_metrics(csv_path: str):
    if not os.path.exists(csv_path):
        return

    val_metrics = {}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epoch = int(row["epoch"])
                step = int(row["step"])
                oa = float(row["val_OverallAccuracy"])
                aa = float(row["val_AverageAccuracy"])
                f1 = float(row["val_F1Score"])
                loss = float(row["val_loss"])
            except (KeyError, ValueError):
                continue

            val_metrics[epoch] = {
                "step": step,
                "OverallAcc": oa,
                "AvgAcc": aa,
                "F1": f1,
                "Loss": loss
            }

    if not val_metrics:
        print("[WARNING] No valid validation metric rows found.")
        return

    print("\n=== Validation Summary ===")
    print(f"{'Epoch':<6} {'OA':<8} {'AA':<8} {'F1':<8} {'Loss':<8}")
    for epoch, m in sorted(val_metrics.items()):
        print(f"{epoch:<6} {m['OverallAcc']:<8.3f} {m['AvgAcc']:<8.3f} {m['F1']:<8.3f} {m['Loss']:<8.3f}")

    best_epoch = max(val_metrics.items(), key=lambda kv: kv[1]["F1"])
    print(f"\nBest epoch by F1: {best_epoch[0]} (F1 = {best_epoch[1]['F1']:.3f})")
