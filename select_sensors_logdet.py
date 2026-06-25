"""Unsupervised log-determinant sensor ranking using training samples only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from sensor_dataloader import DATASET_FILES, build_dataloaders
from train_all_sensors import git_commit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank sensors by greedy Gaussian log-det information criterion."
    )
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--output-dir", default="results_logdet")
    parser.add_argument("--dataset", choices=list(DATASET_FILES), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--baseline-subtract", action="store_true")
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline"),
        default=os.getenv("WANDB_MODE", "online"),
    )
    return parser.parse_args()


def logdet_score(covariance: np.ndarray, selected: list[int], epsilon: float) -> float:
    if not selected:
        return 0.0
    sub_cov = covariance[np.ix_(selected, selected)]
    regularized = sub_cov + epsilon * np.eye(len(selected), dtype=np.float64)
    sign, logdet = np.linalg.slogdet(regularized)
    if sign <= 0:
        raise ValueError("Regularized covariance is not positive definite")
    return 0.5 * float(logdet)


def greedy_logdet_ranking(
    covariance: np.ndarray, epsilon: float
) -> tuple[list[int], list[dict]]:
    selected: list[int] = []
    remaining = set(range(covariance.shape[0]))
    current_score = 0.0
    records: list[dict] = []
    while remaining:
        best_sensor = None
        best_score = -np.inf
        for candidate in sorted(remaining):
            score = logdet_score(covariance, selected + [candidate], epsilon)
            if score > best_score:
                best_score = score
                best_sensor = candidate
        assert best_sensor is not None
        gain = best_score - current_score
        selected.append(best_sensor)
        remaining.remove(best_sensor)
        records.append(
            {
                "rank": len(selected),
                "channel_id": int(best_sensor + 1),
                "score": float(best_score),
                "marginal_gain": float(gain),
            }
        )
        current_score = best_score
    return [sensor + 1 for sensor in selected], records


def main() -> None:
    args = parse_args()
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("Install wandb before running logdet selection.") from error

    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_dataloaders(
        args.data_dir,
        args.dataset,
        Path(args.output_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        test_size=0.2,
        channels=list(range(12)),
        baseline_subtract=args.baseline_subtract,
    )

    cross = np.zeros((12, 12), dtype=np.float64)
    sums = np.zeros(12, dtype=np.float64)
    count = 0
    for signals, _, _ in bundle.train_loader:
        observations = signals.permute(0, 2, 1).reshape(-1, 12).double()
        cross += (observations.T @ observations).numpy()
        sums += observations.sum(dim=0).numpy()
        count += observations.shape[0]
        print(f"Accumulated {count} time observations")

    centered_cross = cross - np.outer(sums, sums) / count
    covariance = centered_cross / max(count - 1, 1)
    ranking, ranking_records = greedy_logdet_ranking(covariance, args.epsilon)
    result = {
        "method_family": "logdet",
        "method": "gaussian_logdet_entropy",
        "dataset": args.dataset,
        "split": "train",
        "channel_ids": list(range(1, 13)),
        "ranking": ranking,
        "ranking_records": ranking_records,
        "covariance": covariance.tolist(),
        "config": {
            "epsilon": args.epsilon,
            "observation_count": int(count),
            "normalization": "train_only_channel_zscore",
            "uses_labels": False,
            "seed": args.seed,
            "git_commit": git_commit(),
        },
    }
    output_path = output_dir / f"{args.dataset}_logdet_train.json"
    output_path.write_text(json.dumps(result, indent=2))

    run = wandb.init(
        project="SensorOptimization",
        entity=args.wandb_entity,
        group="logdet",
        job_type="sensor_selection",
        name=f"logdet-{args.dataset}-seed{args.seed}",
        mode=args.wandb_mode,
        config=result["config"] | {"dataset": args.dataset, "method": "logdet"},
    )
    artifact = wandb.Artifact(f"logdet-{args.dataset}-seed{args.seed}", type="analysis")
    artifact.add_file(str(output_path))
    run.log_artifact(artifact)
    run.finish()
    print(f"Saved logdet JSON: {output_path}")


if __name__ == "__main__":
    main()
