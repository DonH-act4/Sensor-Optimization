"""Retrain PipeID classifiers using Top-K sensors from a ranking JSON."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from sensor_dataloader import DATASET_FILES, build_dataloaders
from train_all_sensors import (
    ARCHITECTURE_VERSIONS,
    architecture_version,
    build_model,
    evaluate,
    git_commit,
    save_evaluation,
    set_seed,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a selected-sensor model from a ranking JSON."
    )
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--output-dir", default="results_selected_sensors")
    parser.add_argument("--dataset", choices=list(DATASET_FILES), required=True)
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument(
        "--architecture",
        choices=sorted(ARCHITECTURE_VERSIONS),
        default="v6",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler-t-max", type=int, default=None)
    parser.add_argument("--no-scheduler", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--eval-test-every", type=int, default=0)
    parser.add_argument("--baseline-subtract", action="store_true")
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline"),
        default=os.getenv("WANDB_MODE", "online"),
    )
    return parser.parse_args()


def load_selected_channels(path: str | Path, top_k: int) -> tuple[dict, list[int]]:
    ranking = json.loads(Path(path).read_text())
    channel_ids = ranking.get("ranking")
    if not isinstance(channel_ids, list) or not channel_ids:
        raise ValueError(f"{path}: missing non-empty 'ranking' list")
    if top_k <= 0 or top_k > len(channel_ids):
        raise ValueError(f"top-k must be between 1 and {len(channel_ids)}")
    selected = [int(channel_id) for channel_id in channel_ids[:top_k]]
    if len(set(selected)) != top_k or any(channel < 1 or channel > 12 for channel in selected):
        raise ValueError(f"Invalid selected channel IDs: {selected}")
    return ranking, selected


def save_selected_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    dataset: str,
    architecture_key: str,
    architecture: str,
    epoch: int,
    seed: int,
    selected_channel_ids: list[int],
    selection_json: str,
    ranking_method: str,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    baseline_subtract: bool,
    wandb_run_id: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "dataset": dataset,
            "architecture_key": architecture_key,
            "architecture": architecture,
            "num_classes": 13,
            "selected_channel_ids": selected_channel_ids,
            "selection_json": selection_json,
            "ranking_method": ranking_method,
            "channel_mean": channel_mean,
            "channel_std": channel_std,
            "baseline_subtract": baseline_subtract,
            "epoch": epoch,
            "seed": seed,
            "wandb_project": "SensorOptimization",
            "wandb_run_id": wandb_run_id,
            "git_commit": git_commit(),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if args.checkpoint_every < 0 or args.eval_test_every < 0:
        raise ValueError("checkpoint/eval intervals must be non-negative")

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("Install wandb before training selected sensors.") from error

    selection, selected_channel_ids = load_selected_channels(
        args.selection_json, args.top_k
    )
    method_family = selection.get("method_family", selection.get("method", "ranking"))
    method = selection.get("method", method_family)
    selected_zero_based = [channel_id - 1 for channel_id in selected_channel_ids]
    output_dir = (
        Path(args.output_dir)
        / args.dataset
        / f"{method_family}_{args.architecture}_K{args.top_k}_e{args.epochs:04d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_dataloaders(
        args.data_dir,
        args.dataset,
        Path(args.output_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        test_size=0.2,
        channels=selected_zero_based,
        baseline_subtract=args.baseline_subtract,
    )
    architecture = architecture_version(args.architecture)
    model = build_model(
        args.architecture, in_channels=args.top_k, num_classes=13
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler_t_max = args.scheduler_t_max or args.epochs
    scheduler = None
    if not args.no_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=scheduler_t_max
        )

    run = wandb.init(
        project="SensorOptimization",
        entity=args.wandb_entity,
        group=str(method_family),
        job_type="train",
        name=(
            f"selected-{method_family}-{args.dataset}-K{args.top_k}-"
            f"{args.architecture}-seed{args.seed}"
        ),
        mode=args.wandb_mode,
        config={
            "method_family": method_family,
            "method": method,
            "dataset": args.dataset,
            "selection_json": args.selection_json,
            "selected_channel_ids": selected_channel_ids,
            "K": args.top_k,
            "architecture": architecture,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "scheduler": "none" if args.no_scheduler else "CosineAnnealingLR",
            "scheduler_t_max": None if args.no_scheduler else scheduler_t_max,
            "seed": args.seed,
            "git_commit": git_commit(),
        },
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
        if args.eval_test_every > 0 and epoch % args.eval_test_every == 0:
            test_metrics, _, _, _, _ = evaluate(
                model, bundle.test_loader, criterion, device
            )
            run.log(
                {
                    "epoch": epoch,
                    "test_epoch/loss": test_metrics["loss"],
                    "test_epoch/accuracy": test_metrics["accuracy"],
                    "test_epoch/macro_f1": test_metrics["macro_f1"],
                },
                step=epoch,
            )
        print(
            f"[{args.dataset} K={args.top_k} {method_family}] "
            f"epoch {epoch:03d}/{args.epochs}: loss={train_loss:.6f}, "
            f"accuracy={train_accuracy:.4f}, lr={learning_rate:.3e}"
        )
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            save_selected_checkpoint(
                output_dir / f"checkpoint_epoch{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                dataset=args.dataset,
                architecture_key=args.architecture,
                architecture=architecture,
                epoch=epoch,
                seed=args.seed,
                selected_channel_ids=selected_channel_ids,
                selection_json=args.selection_json,
                ranking_method=str(method_family),
                channel_mean=bundle.channel_mean,
                channel_std=bundle.channel_std,
                baseline_subtract=args.baseline_subtract,
                wandb_run_id=run.id,
            )

    training_seconds = time.perf_counter() - start
    pd.DataFrame(history).to_csv(output_dir / "train_history.csv", index=False)
    metrics, y_true, y_pred, sample_indices, _ = evaluate(
        model, bundle.test_loader, criterion, device
    )
    metrics["training_seconds"] = training_seconds
    report, matrix = save_evaluation(output_dir, metrics, y_true, y_pred, sample_indices)
    save_selected_checkpoint(
        output_dir / "final_model.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataset=args.dataset,
        architecture_key=args.architecture,
        architecture=architecture,
        epoch=args.epochs,
        seed=args.seed,
        selected_channel_ids=selected_channel_ids,
        selection_json=args.selection_json,
        ranking_method=str(method_family),
        channel_mean=bundle.channel_mean,
        channel_std=bundle.channel_std,
        baseline_subtract=args.baseline_subtract,
        wandb_run_id=run.id,
    )
    (output_dir / "selected_channels.json").write_text(
        json.dumps(
            {
                "method_family": method_family,
                "method": method,
                "top_k": args.top_k,
                "selected_channel_ids": selected_channel_ids,
                "selection_json": args.selection_json,
            },
            indent=2,
        )
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
            "test/confusion_matrix": wandb.Table(dataframe=matrix.reset_index()),
            "timing/training_seconds": training_seconds,
        },
        step=args.epochs + 1,
    )
    artifact = wandb.Artifact(
        f"selected-{method_family}-{args.dataset}-K{args.top_k}-{args.architecture}",
        type="experiment-results",
    )
    for path in (
        output_dir / "final_model.pt",
        output_dir / "train_history.csv",
        output_dir / "test_metrics.json",
        output_dir / "classification_report.csv",
        output_dir / "confusion_matrix.csv",
        output_dir / "test_predictions.csv",
        output_dir / "selected_channels.json",
        Path(args.selection_json),
    ):
        artifact.add_file(str(path))
    run.log_artifact(artifact)
    run.finish()
    print(f"Completed selected-sensor training. Results: {output_dir}")


if __name__ == "__main__":
    main()
