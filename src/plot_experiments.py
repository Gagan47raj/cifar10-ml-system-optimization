
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = ROOT / "results" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = {
    "A_workers": {
        "file": ROOT / "results" / "experiment_A_repeated"
                / "batch_size_mean_std.csv",
        "x": "num_workers",
        "label": "DataLoader workers",
    },
    "B_batch_size": {
        "file": ROOT / "results" / "experiment_B_batch_size"
                / "batch_size_mean_std.csv",
        "x": "batch_size",
        "label": "Batch size",
    },
}

METRICS = [
    (
        "steady_state_mean_training_epoch_seconds",
        "Mean training time per epoch (seconds)",
        "training_epoch_time",
    ),
    (
        "steady_state_mean_training_throughput",
        "Mean training throughput (images/second)",
        "throughput",
    ),
    (
        "best_validation_accuracy",
        "Mean best validation accuracy",
        "validation_accuracy",
    ),
]

for experiment_name, config in EXPERIMENTS.items():
    csv_path = config["file"]

    if not csv_path.exists():
        print(f"Skipping {experiment_name}: missing {csv_path}")
        continue

    df = pd.read_csv(csv_path).sort_values(config["x"])

    for metric, ylabel, suffix in METRICS:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        if mean_col not in df.columns:
            print(f"Skipping {metric}: column {mean_col} not found")
            continue

        x = df[config["x"]]
        y = df[mean_col]
        error = df[std_col] if std_col in df.columns else None

        plt.figure(figsize=(8, 5))
        plt.errorbar(
            x, y, yerr=error,
            marker="o", capsize=5
        )
        plt.xlabel(config["label"])
        plt.ylabel(ylabel)
        plt.title(f"{experiment_name}: {ylabel}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output = PLOT_DIR / f"{experiment_name}_{suffix}.png"
        plt.savefig(output, dpi=200)
        plt.close()
        print(f"Saved: {output}")

print("Plot generation finished.")
