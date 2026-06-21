"""Train independent 13-class CNNs using all 12 pressure sensors."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn

from sensor_dataloader import DATASET_FILES, DataBundle, build_dataloaders


ARCHITECTURE_VERSION = "all_sensor_cnn_v1"


class PipeIDCNN(nn.Module):
    def __init__(self, in_channels: int = 12, num_classes: int = 13) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(in_channels, 64, 9),
            self._block(64, 128, 7),
            self._block(128, 256, 5),
            self._block(256, 384, 3),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(192, num_classes),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int, kernel_size: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size,
                padding=kernel_size // 2, bias=False
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train all-sensor CNNs to predict 13 PipeID classes."
    )
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--output-dir", default="results_cnn_13class")
    parser.add_argument(
        "--datasets", nargs="+", choices=list(DATASET_FILES),
        default=list(DATASET_FILES)
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--baseline-subtract", action="store_true")
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline"),
        default=os.getenv("WANDB_MODE", "online")
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    count = 0
    for signals, labels, _ in loader:
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(signals)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_count = labels.size(0)
        loss_sum += loss.item() * batch_count
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += batch_count
    return loss_sum / count, correct / count


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    loss_sum = 0.0
    labels_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    indices_all: list[np.ndarray] = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for signals, labels, indices in loader:
        signals = signals.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        logits = model(signals)
        loss_sum += criterion(logits, labels_device).item() * labels.size(0)
        labels_all.append(labels.numpy())
        predictions_all.append(logits.argmax(dim=1).cpu().numpy())
        indices_all.append(indices.numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    y_true = np.concatenate(labels_all)
    y_pred = np.concatenate(predictions_all)
    sample_indices = np.concatenate(indices_all)
    metrics = {
        "loss": loss_sum / len(y_true),
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "inference_seconds": elapsed,
        "milliseconds_per_sample": elapsed * 1000 / len(y_true),
    }
    return metrics, y_true, y_pred, sample_indices, elapsed


def save_evaluation(
    output_dir: Path,
    metrics: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_indices: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "test_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    labels = np.arange(13)
    names = [f"PipeID_{i}" for i in range(1, 14)]
    report_dict = classification_report(
        y_true, y_pred, labels=labels, target_names=names,
        output_dict=True, zero_division=0
    )
    report = pd.DataFrame(report_dict).transpose()
    report.to_csv(output_dir / "classification_report.csv")
    matrix = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=names, columns=names
    )
    matrix.to_csv(output_dir / "confusion_matrix.csv")
    pd.DataFrame(
        {
            "sample_index": sample_indices,
            "true_pipe_id": y_true + 1,
            "predicted_pipe_id": y_pred + 1,
            "correct": y_true == y_pred,
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)
    return report, matrix


def run_dataset(args: argparse.Namespace, dataset: str, device: torch.device) -> dict:
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B is mandatory. Install it with `pip install wandb`, then run "
            "`wandb login`, or use --wandb-mode offline."
        ) from error

    set_seed(args.seed)  # Identical initialization for every noise condition.
    output_root = Path(args.output_dir)
    dataset_dir = output_root / dataset
    bundle: DataBundle = build_dataloaders(
        args.data_dir, dataset, output_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        test_size=0.2,
        channels=list(range(12)),
        baseline_subtract=args.baseline_subtract,
    )
    config = {
        "method": "baseline",
        "dataset": dataset,
        "noise_level": dataset,
        "seed": args.seed,
        "split_file": str(output_root / "split_indices_80_20.npz"),
        "train_ratio": 0.8,
        "test_ratio": 0.2,
        "selected_channel_ids": list(range(1, 13)),
        "K": 12,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler": "CosineAnnealingLR",
        "normalization": "train_only_channel_zscore",
        "baseline_subtract": args.baseline_subtract,
        "architecture": ARCHITECTURE_VERSION,
        "git_commit": git_commit(),
    }
    run = wandb.init(
        project="SensorOptimization",
        entity=args.wandb_entity,
        group="baseline",
        job_type="train",
        name=f"baseline-{dataset}-seed{args.seed}",
        config=config,
        mode=args.wandb_mode,
        reinit=True,
    )
    try:
        # Reset after W&B initialization so every noise condition starts from
        # exactly the same model initialization and training RNG state.
        set_seed(args.seed)
        model = PipeIDCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        history: list[dict] = []
        start = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            learning_rate = optimizer.param_groups[0]["lr"]
            train_loss, train_accuracy = train_one_epoch(
                model, bundle.train_loader, criterion, optimizer, device
            )
            scheduler.step()
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "learning_rate": learning_rate,
            }
            history.append(row)
            run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/accuracy": train_accuracy,
                    "learning_rate": learning_rate,
                },
                step=epoch,
            )
            print(
                f"[{dataset}] epoch {epoch:03d}/{args.epochs}: "
                f"loss={train_loss:.6f}, accuracy={train_accuracy:.4f}, "
                f"lr={learning_rate:.3e}"
            )
        training_seconds = time.perf_counter() - start
        pd.DataFrame(history).to_csv(dataset_dir / "train_history.csv", index=False)

        # The held-out test set is evaluated exactly once, after the final epoch.
        metrics, y_true, y_pred, sample_indices, _ = evaluate(
            model, bundle.test_loader, criterion, device
        )
        metrics["training_seconds"] = training_seconds
        report, matrix = save_evaluation(
            dataset_dir, metrics, y_true, y_pred, sample_indices
        )
        checkpoint_path = dataset_dir / "final_model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "architecture": ARCHITECTURE_VERSION,
                "num_classes": 13,
                "selected_channel_ids": list(range(1, 13)),
                "channel_mean": bundle.channel_mean,
                "channel_std": bundle.channel_std,
                "baseline_subtract": args.baseline_subtract,
                "epoch": args.epochs,
                "seed": args.seed,
            },
            checkpoint_path,
        )

        per_pipe = report.loc[[f"PipeID_{i}" for i in range(1, 14)]]
        run.log(
            {
                "test/loss": metrics["loss"],
                "test/accuracy": metrics["accuracy"],
                "test/macro_f1": metrics["macro_f1"],
                "test/per_pipe_precision": wandb.Table(
                    dataframe=per_pipe[["precision"]].reset_index()
                ),
                "test/per_pipe_recall": wandb.Table(
                    dataframe=per_pipe[["recall"]].reset_index()
                ),
                "test/per_pipe_f1": wandb.Table(
                    dataframe=per_pipe[["f1-score"]].reset_index()
                ),
                "test/confusion_matrix": wandb.Table(
                    dataframe=matrix.reset_index()
                ),
                "timing/training_seconds": training_seconds,
                "timing/inference_seconds": metrics["inference_seconds"],
            },
            step=args.epochs + 1,
        )
        artifact = wandb.Artifact(
            f"baseline-{dataset}-seed{args.seed}", type="experiment-results"
        )
        for path in (
            checkpoint_path,
            dataset_dir / "train_history.csv",
            dataset_dir / "test_metrics.json",
            dataset_dir / "classification_report.csv",
            dataset_dir / "confusion_matrix.csv",
            dataset_dir / "test_predictions.csv",
            dataset_dir / "normalization_stats.npz",
            output_root / "split_indices_80_20.npz",
        ):
            artifact.add_file(str(path))
        run.log_artifact(artifact)
        return {"dataset": dataset, **metrics}
    finally:
        run.finish()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    summaries = [run_dataset(args, dataset, device) for dataset in args.datasets]
    summary_path = Path(args.output_dir) / "summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"Completed {len(summaries)} run(s). Summary: {summary_path}")


if __name__ == "__main__":
    main()
