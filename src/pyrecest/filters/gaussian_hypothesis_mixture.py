"""Moment matching for weighted Gaussian state hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WeightedGaussianHypothesis:
    """One weighted Gaussian hypothesis."""

    mean: np.ndarray
    covariance: np.ndarray
    log_weight: float = 0.0
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float).reshape(-1)
        covariance = np.asarray(self.covariance, dtype=float)
        if covariance.shape != (mean.size, mean.size):
            raise ValueError("covariance must match mean dimension")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", _symmetrized(covariance))
        object.__setattr__(self, "log_weight", float(self.log_weight))
        if self.metadata is not None:
            object.__setattr__(self, "metadata", dict(self.metadata))


def moment_match_gaussian_hypotheses(
    hypotheses: (
        list[WeightedGaussianHypothesis] | tuple[WeightedGaussianHypothesis, ...]
    ),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return moment-matched mean/covariance and normalized weights."""
    if not hypotheses:
        raise ValueError("hypotheses must not be empty")
    weights = normalize_log_weights(
        [hypothesis.log_weight for hypothesis in hypotheses]
    )
    means = np.stack([hypothesis.mean for hypothesis in hypotheses], axis=0)
    if not all(mean.size == means.shape[1] for mean in means):
        raise ValueError("all hypothesis means must have the same dimension")

    mean = weights @ means
    covariance = np.zeros((mean.size, mean.size), dtype=float)
    for weight, hypothesis in zip(weights, hypotheses):
        if hypothesis.mean.size != mean.size:
            raise ValueError("all hypothesis means must have the same dimension")
        diff = hypothesis.mean - mean
        covariance += float(weight) * (hypothesis.covariance + np.outer(diff, diff))
    return mean, _symmetrized(covariance), weights


def normalize_log_weights(log_weights: list[float] | np.ndarray) -> np.ndarray:
    """Normalize log weights to probabilities in a numerically stable way."""
    values = np.asarray(log_weights, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("log_weights must not be empty")
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        return np.full(values.size, 1.0 / values.size)
    weights = np.exp(values - maximum)
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        return np.full(values.size, 1.0 / values.size)
    return weights / total


def _symmetrized(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    return 0.5 * (array + array.T)
