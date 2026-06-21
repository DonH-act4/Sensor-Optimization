"""Reusable data loading utilities for the 12-sensor leak dataset.

The MATLAB files are v7.3 (HDF5) files whose struct fields contain object
references.  This module reads samples lazily, preserves MATLAB channel order,
and exposes only PipeID as the learning target.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


DATASET_FILES = {
    "clean": ("Dataset_4000_Clean.mat", "Signals"),
    "snr0": ("Dataset_4000_SNR_0.mat", "NoisySignals_SNR0dB"),
    "snr1": ("Dataset_4000_SNR_1.mat", "NoisySignals_SNR1dB"),
    "snr5": ("Dataset_4000_SNR_5.mat", "NoisySignals_SNR5dB"),
}


@dataclass(frozen=True)
class DataBundle:
    train_loader: DataLoader
    test_loader: DataLoader
    train_indices: np.ndarray
    test_indices: np.ndarray
    labels: np.ndarray
    channel_mean: np.ndarray
    channel_std: np.ndarray
    dataset_path: Path
    signal_field: str


def resolve_dataset(data_dir: str | Path, dataset: str) -> tuple[Path, str]:
    if dataset not in DATASET_FILES:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {list(DATASET_FILES)}")
    filename, signal_field = DATASET_FILES[dataset]
    path = Path(data_dir) / filename
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return path, signal_field


def _flat_references(reference_dataset: h5py.Dataset) -> np.ndarray:
    # order='F' also handles the expected MATLAB 1xN layout correctly.
    return np.asarray(reference_dataset).reshape(-1, order="F")


def load_pipe_ids(path: str | Path) -> np.ndarray:
    """Read PipeID only and convert MATLAB labels 1..13 to PyTorch labels 0..12."""
    with h5py.File(path, "r") as handle:
        if "Dataset" not in handle or "LeakInfo_ID_xL_sL" not in handle["Dataset"]:
            raise KeyError(f"{path}: missing Dataset/LeakInfo_ID_xL_sL")
        refs = _flat_references(handle["Dataset/LeakInfo_ID_xL_sL"])
        labels = np.empty(len(refs), dtype=np.int64)
        for i, ref in enumerate(refs):
            labels[i] = int(np.asarray(handle[ref]).reshape(-1)[0]) - 1

    unique = np.unique(labels)
    if not np.array_equal(unique, np.arange(13)):
        raise ValueError(f"Expected PipeID classes 1..13, found {(unique + 1).tolist()}")
    return labels


def load_or_create_split(
    labels: np.ndarray,
    split_path: str | Path,
    seed: int = 42,
    test_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Create once, then reuse the exact stratified split for every noise level."""
    split_path = Path(split_path)
    if split_path.exists():
        split = np.load(split_path)
        train_indices = split["train_indices"].astype(np.int64)
        test_indices = split["test_indices"].astype(np.int64)
        if "labels" not in split or not np.array_equal(split["labels"], labels):
            raise ValueError(
                "Existing split labels do not match this file; paired sample order is invalid"
            )
        combined = np.concatenate([train_indices, test_indices])
        if len(combined) != len(labels) or not np.array_equal(
            np.sort(combined), np.arange(len(labels))
        ):
            raise ValueError(f"Existing split is incompatible with {len(labels)} samples")
        return train_indices, test_indices

    all_indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
        shuffle=True,
    )
    split_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        split_path,
        train_indices=np.sort(train_indices),
        test_indices=np.sort(test_indices),
        labels=labels,
        seed=np.int64(seed),
        test_size=np.float64(test_size),
    )
    return np.sort(train_indices), np.sort(test_indices)


def validate_mat_file(path: str | Path, signal_field: str) -> tuple[int, tuple[int, int]]:
    with h5py.File(path, "r") as handle:
        group = handle.get("Dataset")
        if group is None or signal_field not in group:
            raise KeyError(f"{path}: missing Dataset/{signal_field}")
        signal_refs = _flat_references(group[signal_field])
        label_refs = _flat_references(group["LeakInfo_ID_xL_sL"])
        if len(signal_refs) != len(label_refs):
            raise ValueError(f"{path}: signal and label counts differ")
        first_shape = tuple(np.asarray(handle[signal_refs[0]]).shape)
        if first_shape != (12, 3334):
            raise ValueError(f"{path}: expected signal [12, 3334], found {first_shape}")
        return len(signal_refs), first_shape


def compute_channel_stats(
    path: str | Path,
    signal_field: str,
    train_indices: Sequence[int],
    baseline_subtract: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one mean/std per channel using training samples only."""
    sums = np.zeros(12, dtype=np.float64)
    squared_sums = np.zeros(12, dtype=np.float64)
    count = 0
    with h5py.File(path, "r") as handle:
        refs = _flat_references(handle[f"Dataset/{signal_field}"])
        for index in train_indices:
            signal = np.asarray(handle[refs[int(index)]], dtype=np.float64)
            if baseline_subtract:
                signal = signal - signal[:, :1]
            sums += signal.sum(axis=1)
            squared_sums += np.square(signal).sum(axis=1)
            count += signal.shape[1]

    mean = sums / count
    variance = np.maximum(squared_sums / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    if np.any(std < 1e-12):
        raise ValueError("At least one sensor has near-zero training standard deviation")
    return mean.astype(np.float32), std.astype(np.float32)


class LeakPipeDataset(Dataset):
    """Lazy HDF5 dataset returning (normalized_signal, PipeID, sample_index)."""

    def __init__(
        self,
        path: str | Path,
        signal_field: str,
        indices: Sequence[int],
        labels: np.ndarray,
        channel_mean: np.ndarray,
        channel_std: np.ndarray,
        channels: Sequence[int] | None = None,
        baseline_subtract: bool = False,
    ) -> None:
        self.path = str(path)
        self.signal_field = signal_field
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.channel_mean = np.asarray(channel_mean, dtype=np.float32)[:, None]
        self.channel_std = np.asarray(channel_std, dtype=np.float32)[:, None]
        self.channels = np.arange(12) if channels is None else np.asarray(channels)
        self.baseline_subtract = baseline_subtract
        self._handle: h5py.File | None = None
        self._signal_refs: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def _open(self) -> None:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
            self._signal_refs = _flat_references(
                self._handle[f"Dataset/{self.signal_field}"]
            )

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        self._open()
        sample_index = int(self.indices[item])
        assert self._handle is not None and self._signal_refs is not None
        signal = np.asarray(
            self._handle[self._signal_refs[sample_index]], dtype=np.float32
        )
        if self.baseline_subtract:
            signal = signal - signal[:, :1]
        signal = (signal - self.channel_mean) / self.channel_std
        signal = np.ascontiguousarray(signal[self.channels])
        return (
            torch.from_numpy(signal),
            torch.tensor(self.labels[sample_index], dtype=torch.long),
            sample_index,
        )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_signal_refs"] = None
        return state

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._signal_refs = None

    def __del__(self) -> None:
        self.close()


def build_dataloaders(
    data_dir: str | Path,
    dataset: str,
    output_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    test_size: float = 0.2,
    channels: Sequence[int] | None = None,
    baseline_subtract: bool = False,
) -> DataBundle:
    """Build reusable train/test loaders with a shared persistent split."""
    path, signal_field = resolve_dataset(data_dir, dataset)
    sample_count, signal_shape = validate_mat_file(path, signal_field)
    labels = load_pipe_ids(path)
    split_path = Path(output_dir) / "split_indices_80_20.npz"
    train_indices, test_indices = load_or_create_split(
        labels, split_path, seed=seed, test_size=test_size
    )

    stats_dir = Path(output_dir) / dataset
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_path = stats_dir / "normalization_stats.npz"
    if stats_path.exists():
        stats = np.load(stats_path)
        channel_mean = stats["channel_mean"].astype(np.float32)
        channel_std = stats["channel_std"].astype(np.float32)
        saved_baseline = bool(stats["baseline_subtract"])
        if saved_baseline != baseline_subtract:
            raise ValueError("Saved normalization used a different baseline setting")
    else:
        channel_mean, channel_std = compute_channel_stats(
            path, signal_field, train_indices, baseline_subtract
        )
        np.savez(
            stats_path,
            channel_mean=channel_mean,
            channel_std=channel_std,
            baseline_subtract=np.bool_(baseline_subtract),
        )

    print(
        f"{dataset}: {sample_count} samples, signal {signal_shape}, "
        f"train={len(train_indices)}, test={len(test_indices)}"
    )
    train_dataset = LeakPipeDataset(
        path, signal_field, train_indices, labels, channel_mean, channel_std,
        channels, baseline_subtract
    )
    test_dataset = LeakPipeDataset(
        path, signal_field, test_indices, labels, channel_mean, channel_std,
        channels, baseline_subtract
    )
    generator = torch.Generator().manual_seed(seed)
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **common
    )
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return DataBundle(
        train_loader=train_loader,
        test_loader=test_loader,
        train_indices=train_indices,
        test_indices=test_indices,
        labels=labels,
        channel_mean=channel_mean,
        channel_std=channel_std,
        dataset_path=path,
        signal_field=signal_field,
    )
