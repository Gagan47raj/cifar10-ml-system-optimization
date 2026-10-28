
import csv
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dataset import build_dataloaders
from model import SmallCNN



ROOT = Path(__file__).resolve().parents[1]

OUT = ROOT / "results" / "experiment_B_batch_size"
LOGS = OUT / "logs"
CHECKPOINTS = OUT / "checkpoints"

BATCH_SIZES = [64, 128, 256]
TRIAL_SEEDS = [42, 43, 44]
EPOCHS = 15

# Fixed to isolate the effect of batch size
NUM_WORKERS = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SPLIT_SEED = 42

LOGS.mkdir(parents=True, exist_ok=True)
CHECKPOINTS.mkdir(parents=True, exist_ok=True)



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    loss_sum = 0.0
    correct = 0
    total = 0

    sync(device)
    start = time.perf_counter()

    with (torch.enable_grad() if training else torch.no_grad()):
        for images, labels in loader:
            images = images.to(
                device, non_blocking=(device.type == "cuda")
            )
            labels = labels.to(
                device, non_blocking=(device.type == "cuda")
            )

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            n = labels.size(0)
            total += n
            loss_sum += loss.item() * n
            correct += (logits.argmax(1) == labels).sum().item()

    sync(device)
    elapsed = time.perf_counter() - start

    return {
        "loss": loss_sum / total,
        "accuracy": correct / total,
        "seconds": elapsed,
        "images_per_second": total / elapsed,
    }


def write_csv(path, rows):
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)



def run_trial(batch_size, trial_seed, device):
    seed_everything(trial_seed)

    run_name = f"batch_{batch_size}_seed_{trial_seed}"
    print(f"\n{'=' * 65}")
    print(f"Run: {run_name}")
    print(f"{'=' * 65}")

    train_loader, val_loader, _ = build_dataloaders(
        root=str(ROOT / "data"),
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        seed=SPLIT_SEED,
        split_seed=SPLIT_SEED,
        loader_seed=trial_seed,
    )

    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    rows = []
    best_val_accuracy = -1.0
    best_epoch = 0
    best_checkpoint = CHECKPOINTS / f"{run_name}_best.pt"

    for epoch in range(1, EPOCHS + 1):
        sync(device)
        response_start = time.perf_counter()

        train = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val = run_epoch(model, val_loader, criterion, device)

        sync(device)
        response_seconds = time.perf_counter() - response_start

        row = {
            "run": run_name,
            "batch_size": batch_size,
            "num_workers": NUM_WORKERS,
            "trial_seed": trial_seed,
            "epoch": epoch,
            "train_loss": train["loss"],
            "train_accuracy": train["accuracy"],
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "train_seconds": train["seconds"],
            "validation_seconds": val["seconds"],
            "epoch_response_seconds": response_seconds,
            "train_images_per_second": train["images_per_second"],
        }
        rows.append(row)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"val_acc={val['accuracy']:.4f} | "
            f"train={train['seconds']:.2f}s | "
            f"response={response_seconds:.2f}s | "
            f"throughput={train['images_per_second']:.1f}"
        )

        if val["accuracy"] > best_val_accuracy:
            best_val_accuracy = val["accuracy"]
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_accuracy": best_val_accuracy,
                    "batch_size": batch_size,
                    "num_workers": NUM_WORKERS,
                    "trial_seed": trial_seed,
                    "split_seed": SPLIT_SEED,
                },
                best_checkpoint,
            )

    write_csv(LOGS / f"{run_name}_epochs.csv", rows)

    # Exclude epoch 1 from steady-state metrics
    steady = rows[1:]

    summary = {
        "run": run_name,
        "batch_size": batch_size,
        "num_workers": NUM_WORKERS,
        "trial_seed": trial_seed,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_val_accuracy,
        "final_validation_accuracy": rows[-1]["val_accuracy"],
        "first_epoch_training_seconds": rows[0]["train_seconds"],
        "steady_state_mean_training_epoch_seconds": statistics.mean(
            r["train_seconds"] for r in steady
        ),
        "steady_state_mean_epoch_response_seconds": statistics.mean(
            r["epoch_response_seconds"] for r in steady
        ),
        "steady_state_mean_training_throughput": statistics.mean(
            r["train_images_per_second"] for r in steady
        ),
        "total_training_seconds": sum(
            r["train_seconds"] for r in rows
        ),
        "total_epoch_response_seconds": sum(
            r["epoch_response_seconds"] for r in rows
        ),
        "checkpoint": str(best_checkpoint.relative_to(ROOT)),
    }

    with (LOGS / f"{run_name}_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print("\nTrial summary:")
    print(json.dumps(summary, indent=2))
    return summary




def aggregate_trials(trial_summaries):
    metrics = [
        "steady_state_mean_training_epoch_seconds",
        "steady_state_mean_epoch_response_seconds",
        "steady_state_mean_training_throughput",
        "best_validation_accuracy",
        "final_validation_accuracy",
        "total_training_seconds",
    ]

    aggregated = []

    for batch_size in BATCH_SIZES:
        group = [
            row for row in trial_summaries
            if row["batch_size"] == batch_size
        ]

        if len(group) != len(TRIAL_SEEDS):
            raise RuntimeError(
                f"Batch size {batch_size}: expected "
                f"{len(TRIAL_SEEDS)} trials, found {len(group)}"
            )

        result = {
            "batch_size": batch_size,
            "num_workers": NUM_WORKERS,
            "trial_count": len(group),
        }

        for metric in metrics:
            values = [float(row[metric]) for row in group]
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_std"] = statistics.stdev(values)

        aggregated.append(result)

    return aggregated




def save_aggregates(aggregated, trial_summaries):
    trials_path = OUT / "all_trial_summaries.csv"
    write_csv(trials_path, trial_summaries)

    csv_path = OUT / "batch_size_mean_std.csv"
    write_csv(csv_path, aggregated)

    json_path = OUT / "batch_size_mean_std.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

    print("\nAggregate files saved:")
    print(f"  {trials_path}")
    print(f"  {csv_path}")
    print(f"  {json_path}")




def load_completed_trial(batch_size, trial_seed):
    run_name = f"batch_{batch_size}_seed_{trial_seed}"

    summary_path = LOGS / f"{run_name}_summary.json"
    epochs_path = LOGS / f"{run_name}_epochs.csv"
    checkpoint_path = CHECKPOINTS / f"{run_name}_best.pt"

    if not (
        summary_path.is_file()
        and epochs_path.is_file()
        and checkpoint_path.is_file()
    ):
        return None

    try:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        with epochs_path.open("r", newline="", encoding="utf-8") as f:
            epoch_rows = list(csv.DictReader(f))

        if len(epoch_rows) != EPOCHS:
            return None

        if summary.get("run") != run_name:
            return None

        if summary.get("batch_size") != batch_size:
            return None

        if not checkpoint_path.stat().st_size:
            return None

        return summary

    except (OSError, json.JSONDecodeError, csv.Error):
        return None


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_summaries = []

    # Interleave batch sizes across trial seeds
    for trial_seed in TRIAL_SEEDS:
        for batch_size in BATCH_SIZES:
            run_name = f"batch_{batch_size}_seed_{trial_seed}"

            saved_summary = load_completed_trial(
                batch_size, trial_seed
            )

            if saved_summary is not None:
                print(f"\nSkipping completed trial: {run_name}")
                all_summaries.append(saved_summary)
                continue

            print(f"\nRunning trial: {run_name}")
            summary = run_trial(batch_size, trial_seed, device)
            all_summaries.append(summary)

    expected_runs = {
        f"batch_{batch_size}_seed_{seed}"
        for seed in TRIAL_SEEDS
        for batch_size in BATCH_SIZES
    }
    completed_runs = {s.get("run") for s in all_summaries}

    if completed_runs != expected_runs:
        missing = sorted(expected_runs - completed_runs)
        raise RuntimeError(f"Missing trials: {missing}")

    aggregated = aggregate_trials(all_summaries)
    save_aggregates(aggregated, all_summaries)


if __name__ == "__main__":
    main()
