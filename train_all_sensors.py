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


DEFAULT_ARCHITECTURE = "v1"
ARCHITECTURE_VERSIONS = {
    "v1": "all_sensor_cnn_v1_gap",
    "v2": "all_sensor_cnn_v2_temporal8",
    "v3": "all_sensor_resnet_v3_gapmax",
    "v4": "all_sensor_cnn_v4_wide_gap",
    "v5": "all_sensor_patch_transformer_v5",
}


class PipeIDCNNV1(nn.Module):
    """Original stable all-sensor CNN baseline."""

    def __init__(self, in_channels: int = 12, num_classes: int = 13) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(in_channels, 64, 9),
            self._block(64, 128, 7),
            self._block(128, 256, 5),
            self._block(256, 384, 3),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
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
        return self.classifier(self.pool(self.features(inputs)))


class PipeIDCNNV2(PipeIDCNNV1):
    """Temporal-pooling pilot model kept for reproducibility."""

    def __init__(self, in_channels: int = 12, num_classes: int = 13) -> None:
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            self._block(in_channels, 64, 9),
            self._block(64, 128, 7),
            self._block(128, 256, 5),
            self._block(256, 384, 3),
        )
        self.temporal_pool = nn.AdaptiveAvgPool1d(8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(384 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
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
        features = self.features(inputs)
        return self.classifier(self.temporal_pool(features))


class PipeIDCNNV4(PipeIDCNNV1):
    """Wider v1-style CNN.

    This is intentionally conservative: it keeps the v1 block layout, max
    pooling schedule, global average pooling, and small classifier head. The
    only material change is larger channel capacity.
    """

    def __init__(self, in_channels: int = 12, num_classes: int = 13) -> None:
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            self._block(in_channels, 96, 9),
            self._block(96, 192, 7),
            self._block(192, 384, 5),
            self._block(384, 512, 3),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )


class ResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.main = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, bias=False
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels, out_channels, kernel_size,
                padding=padding, bias=False
            ),
            nn.BatchNorm1d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(inputs) + self.shortcut(inputs))


class PipeIDResNetV3(nn.Module):
    """Residual 1D CNN with average+max global pooling.

    This version keeps the stable global-pooling behavior of v1, but improves
    gradient flow and feature capacity through residual blocks. It avoids the
    large flattened temporal head used by v2, which was harder to optimize in
    the pilot run.
    """

    def __init__(self, in_channels: int = 12, num_classes: int = 13) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(
            ResidualBlock1D(64, 64, kernel_size=9),
            ResidualBlock1D(64, 128, kernel_size=7, stride=2, dropout=0.05),
            ResidualBlock1D(128, 128, kernel_size=7, dropout=0.05),
            ResidualBlock1D(128, 256, kernel_size=5, stride=2, dropout=0.10),
            ResidualBlock1D(256, 256, kernel_size=5, dropout=0.10),
            ResidualBlock1D(256, 384, kernel_size=3, stride=2, dropout=0.10),
            ResidualBlock1D(384, 384, kernel_size=3, dropout=0.10),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(384 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(self.stem(inputs))
        pooled = torch.cat(
            [
                self.avg_pool(features).flatten(1),
                self.max_pool(features).flatten(1),
            ],
            dim=1,
        )
        return self.classifier(pooled)


class PipeIDPatchTransformerV5(nn.Module):
    """Conv-patch Transformer for long 12-channel transients.

    The raw sequence has 3334 time points, which is too long for a vanilla
    time-step Transformer in this small-data setting. This model first uses a
    strided Conv1d tokenizer to make about 209 temporal tokens, then applies a
    compact Transformer encoder and a CLS-token classification head.
    """

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 13,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.15,
        max_tokens: int = 256,
    ) -> None:
        super().__init__()
        self.tokenizer = nn.Sequential(
            nn.Conv1d(
                in_channels, d_model, kernel_size=25,
                stride=16, padding=12, bias=False
            ),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(
                d_model, d_model, kernel_size=7,
                padding=3, groups=d_model, bias=False
            ),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_tokens + 1, d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(inputs).transpose(1, 2)
        token_count = tokens.size(1)
        if token_count + 1 > self.position_embedding.size(1):
            raise ValueError(
                f"Too many tokens ({token_count}); increase max_tokens in v5"
            )
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.position_embedding[:, : token_count + 1]
        encoded = self.encoder(tokens)
        return self.classifier(encoded[:, 0])


def build_model(
    architecture: str,
    in_channels: int = 12,
    num_classes: int = 13,
) -> nn.Module:
    if architecture == "v1":
        return PipeIDCNNV1(in_channels=in_channels, num_classes=num_classes)
    if architecture == "v2":
        return PipeIDCNNV2(in_channels=in_channels, num_classes=num_classes)
    if architecture == "v3":
        return PipeIDResNetV3(in_channels=in_channels, num_classes=num_classes)
    if architecture == "v4":
        return PipeIDCNNV4(in_channels=in_channels, num_classes=num_classes)
    if architecture == "v5":
        return PipeIDPatchTransformerV5(
            in_channels=in_channels, num_classes=num_classes
        )
    raise ValueError(f"Unknown architecture {architecture!r}")


def architecture_version(architecture: str) -> str:
    return ARCHITECTURE_VERSIONS[architecture]


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
    parser.add_argument(
        "--architecture",
        choices=sorted(ARCHITECTURE_VERSIONS),
        default=DEFAULT_ARCHITECTURE,
        help=(
            "v1=original GAP CNN, v2=temporal8 pilot, "
            "v3=residual GAP+max CNN, v4=wider v1-style GAP CNN, "
            "v5=conv-patch Transformer"
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--scheduler-t-max",
        type=int,
        default=None,
        help=(
            "CosineAnnealingLR T_max. Defaults to --epochs. For short smoke "
            "tests, set this to the planned full run length, e.g. 150 or 300."
        ),
    )
    parser.add_argument(
        "--no-scheduler",
        action="store_true",
        help="Keep the learning rate constant. Intended for debugging only.",
    )
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
    arch_version = architecture_version(args.architecture)
    scheduler_t_max = args.scheduler_t_max or args.epochs
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
        "scheduler": "none" if args.no_scheduler else "CosineAnnealingLR",
        "scheduler_t_max": None if args.no_scheduler else scheduler_t_max,
        "normalization": "train_only_channel_zscore",
        "baseline_subtract": args.baseline_subtract,
        "architecture": arch_version,
        "git_commit": git_commit(),
    }
    run = wandb.init(
        project="SensorOptimization",
        entity=args.wandb_entity,
        group="baseline",
        job_type="train",
        name=f"baseline-{args.architecture}-{dataset}-seed{args.seed}",
        config=config,
        mode=args.wandb_mode,
        reinit=True,
    )
    try:
        # Reset after W&B initialization so every noise condition starts from
        # exactly the same model initialization and training RNG state.
        set_seed(args.seed)
        model = build_model(args.architecture).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        scheduler = None
        if not args.no_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=scheduler_t_max
            )
        history: list[dict] = []
        start = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            learning_rate = optimizer.param_groups[0]["lr"]
            train_loss, train_accuracy = train_one_epoch(
                model, bundle.train_loader, criterion, optimizer, device
            )
            if scheduler is not None:
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
                "num_classes": 13,
                "selected_channel_ids": list(range(1, 13)),
                "channel_mean": bundle.channel_mean,
                "channel_std": bundle.channel_std,
                "baseline_subtract": args.baseline_subtract,
                "architecture": arch_version,
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
            f"baseline-{args.architecture}-{dataset}-seed{args.seed}",
            type="experiment-results",
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
    if args.scheduler_t_max is not None and args.scheduler_t_max <= 0:
        raise ValueError("scheduler-t-max must be positive")
    print(f"Architecture: {args.architecture} ({architecture_version(args.architecture)})")
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
