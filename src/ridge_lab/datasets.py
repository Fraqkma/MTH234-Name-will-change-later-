"""Deterministic synthetic data and split contracts for the lab."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DatasetSplit:
    """One independently generated split with read-only one-dimensional arrays."""

    x: FloatArray
    y: FloatArray
    sample_ids: tuple[str, ...]


@dataclass(frozen=True)
class DatasetBundle:
    """Train, validation, and test splits plus the generation settings."""

    train: DatasetSplit
    validation: DatasetSplit
    test: DatasetSplit
    seed: int
    noise_std: float


def _positive_count(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _split(name: str, count: int, seed: np.random.SeedSequence, noise_std: float) -> DatasetSplit:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=count)
    y = np.sin(2.0 * np.pi * x) + rng.normal(0.0, noise_std, size=count)
    x.setflags(write=False)
    y.setflags(write=False)
    return DatasetSplit(x=x, y=y, sample_ids=tuple(f"{name}-{i:04d}" for i in range(count)))


def make_dataset(
    seed: int = 42,
    noise_std: float = 0.2,
    n_train: int = 24,
    n_val: int = 24,
    n_test: int = 120,
) -> DatasetBundle:
    """Generate independent samples from sin(2*pi*x) plus Gaussian noise."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(noise_std, bool) or not isinstance(noise_std, (int, float)):
        raise TypeError("noise_std must be a number")
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and non-negative")
    for name, count in (("n_train", n_train), ("n_val", n_val), ("n_test", n_test)):
        _positive_count(name, count)

    train_seed, val_seed, test_seed = np.random.SeedSequence(seed).spawn(3)
    return DatasetBundle(
        train=_split("train", n_train, train_seed, noise_std),
        validation=_split("validation", n_val, val_seed, noise_std),
        test=_split("test", n_test, test_seed, noise_std),
        seed=seed,
        noise_std=float(noise_std),
    )
