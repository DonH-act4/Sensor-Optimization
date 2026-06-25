"""Compute train-only channel attribution rankings for a trained PipeID model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from sensor_dataloader import DATASET_FILES, build_dataloaders
from train_all_sensors import ARCHITECTURE_VERSIONS, build_model, git_commit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute channel attribution JSON using Integrated Gradients."
    )
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--output-dir", default="results_attribution")
    parser.add_argument("--dataset", choices=list(DATASET_FILES), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--architecture", default=None)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--internal-batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--save-per-sample",
        action="store_true",
        help="Also save per-sample channel scores. Off by default to keep JSON small.",
    )
    parser.add_argument("--baseline-subtract", action="store_true")
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline"),
        default=os.getenv("WANDB_MODE", "online"),
    )
    return parser.parse_args()


def sorted_channel_records(scores: np.ndarray) -> list[dict]:
    order = np.argsort(scores)[::-1]
    return [
        {"rank": rank + 1, "channel_id": int(index + 1), "score": float(scores[index])}
        for rank, index in enumerate(order)
    ]


def main() -> None:
    args = parse_args()
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("Install wandb before running attribution analysis.") from error
    try:
        from captum.attr import IntegratedGradients
    except ImportError as error:
        raise RuntimeError("Install Captum first: pip install captum") from error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    architecture_key = (
        args.architecture
        or checkpoint.get("architecture_key")
        or checkpoint.get("architecture")
        or "v6"
    )
    if architecture_key not in ARCHITECTURE_VERSIONS:
        raise ValueError(
            "Pass --architecture because checkpoint architecture_key is missing "
            f"or not a key: {architecture_key!r}"
        )
    model = build_model(architecture_key, in_channels=12, num_classes=13).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

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
    loader = bundle.train_loader if args.split == "train" else bundle.test_loader

    run = wandb.init(
        project="SensorOptimization",
        entity=args.wandb_entity,
        group="attribution",
        job_type="analysis",
        name=f"attribution-{args.dataset}-{args.split}-{architecture_key}-seed{args.seed}",
        mode=args.wandb_mode,
        config={
            "method_family": "attribution",
            "method": "integrated_gradients",
            "dataset": args.dataset,
            "split": args.split,
            "checkpoint": args.checkpoint,
            "architecture": architecture_key,
            "n_steps": args.n_steps,
            "baseline": "zero_normalized_input",
            "target": "true_pipe_id_logit",
            "seed": args.seed,
            "git_commit": git_commit(),
        },
    )

    ig = IntegratedGradients(model)
    global_sum = np.zeros(12, dtype=np.float64)
    per_pipe_sum = np.zeros((13, 12), dtype=np.float64)
    per_pipe_count = np.zeros(13, dtype=np.int64)
    sample_records: list[dict] | None = [] if args.save_per_sample else None
    sample_count = 0

    for signals, labels, indices in loader:
        if args.max_samples is not None and sample_count >= args.max_samples:
            break
        if args.max_samples is not None:
            remaining = args.max_samples - sample_count
            signals = signals[:remaining]
            labels = labels[:remaining]
            indices = indices[:remaining]
        signals = signals.to(device)
        labels_device = labels.to(device)
        baseline = torch.zeros_like(signals)
        attributions = ig.attribute(
            signals,
            baselines=baseline,
            target=labels_device,
            n_steps=args.n_steps,
            internal_batch_size=args.internal_batch_size,
        )
        channel_scores = attributions.detach().abs().mean(dim=2).cpu().numpy()
        labels_np = labels.numpy()
        indices_np = indices.numpy()

        global_sum += channel_scores.sum(axis=0)
        for row, label, sample_index in zip(channel_scores, labels_np, indices_np):
            per_pipe_sum[label] += row
            per_pipe_count[label] += 1
            if sample_records is not None:
                sample_records.append(
                    {
                        "sample_index": int(sample_index),
                        "true_pipe_id": int(label + 1),
                        "scores": [float(value) for value in row],
                    }
                )
        sample_count += len(labels_np)
        print(f"Attributed {sample_count} samples")

    if sample_count == 0:
        raise RuntimeError("No samples were attributed")

    global_scores = global_sum / sample_count
    per_pipe_scores = np.divide(
        per_pipe_sum,
        np.maximum(per_pipe_count[:, None], 1),
        out=np.zeros_like(per_pipe_sum),
        where=per_pipe_count[:, None] > 0,
    )
    result = {
        "method_family": "attribution",
        "method": "integrated_gradients",
        "dataset": args.dataset,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "architecture": architecture_key,
        "channel_ids": list(range(1, 13)),
        "ranking": [record["channel_id"] for record in sorted_channel_records(global_scores)],
        "global_scores": sorted_channel_records(global_scores),
        "per_pipe_scores": {
            f"PipeID_{pipe_id}": sorted_channel_records(per_pipe_scores[pipe_id - 1])
            for pipe_id in range(1, 14)
        },
        "config": {
            "n_steps": args.n_steps,
            "baseline": "zero_normalized_input",
            "target": "true_pipe_id_logit",
            "sample_count": sample_count,
            "save_per_sample": args.save_per_sample,
            "seed": args.seed,
        },
    }
    if sample_records is not None:
        result["per_sample_scores"] = sample_records
    output_path = output_dir / f"{args.dataset}_{architecture_key}_ig_{args.split}.json"
    output_path.write_text(json.dumps(result, indent=2))
    artifact = wandb.Artifact(
        f"attribution-{args.dataset}-{architecture_key}-{args.split}", type="analysis"
    )
    artifact.add_file(str(output_path))
    run.log_artifact(artifact)
    run.finish()
    print(f"Saved attribution JSON: {output_path}")


if __name__ == "__main__":
    main()
