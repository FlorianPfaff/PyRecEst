"""Generic measurement-bias calibration utilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .time_offset import nearest_time_indices


@dataclass(frozen=True)
class BiasTrainingExamples:
    """Matched measurement/reference pairs used for bias fitting."""

    measured: np.ndarray
    reference: np.ndarray
    residual: np.ndarray
    features: np.ndarray
    time_delta_s: np.ndarray


@dataclass(frozen=True)
class SensorBiasCorrectionModel:
    """Ridge-linear model for measurement residual bias.

    The model predicts ``bias = measured - reference`` from optional feature
    vectors.  Applying the model subtracts the predicted bias from measurements.
    """

    target_dim: int
    feature_dim: int
    intercept: np.ndarray
    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    residual_std: np.ndarray
    training_count: int
    ridge_alpha: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        target_dim = int(self.target_dim)
        feature_dim = int(self.feature_dim)
        if target_dim <= 0:
            raise ValueError("target_dim must be positive")
        if feature_dim < 0:
            raise ValueError("feature_dim must be nonnegative")
        intercept = np.asarray(self.intercept, dtype=float).reshape(target_dim)
        coefficients = np.asarray(self.coefficients, dtype=float).reshape(feature_dim, target_dim)
        feature_mean = np.asarray(self.feature_mean, dtype=float).reshape(feature_dim)
        feature_scale = np.asarray(self.feature_scale, dtype=float).reshape(feature_dim)
        residual_std = np.asarray(self.residual_std, dtype=float).reshape(target_dim)
        feature_scale = np.where(np.isfinite(feature_scale) & (feature_scale > 0.0), feature_scale, 1.0)
        object.__setattr__(self, "target_dim", target_dim)
        object.__setattr__(self, "feature_dim", feature_dim)
        object.__setattr__(self, "intercept", intercept)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "feature_mean", feature_mean)
        object.__setattr__(self, "feature_scale", feature_scale)
        object.__setattr__(self, "residual_std", residual_std)
        object.__setattr__(self, "training_count", int(self.training_count))
        object.__setattr__(self, "ridge_alpha", float(self.ridge_alpha))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def predict(self, features: np.ndarray | None = None, *, n_rows: int | None = None) -> np.ndarray:
        """Predict residual bias for feature rows."""

        if self.feature_dim == 0:
            rows = 1 if n_rows is None else int(n_rows)
            return np.repeat(self.intercept.reshape(1, -1), rows, axis=0)
        if features is None:
            raise ValueError("features are required for a nonconstant bias model")
        x = _as_2d(features, "features")
        if x.shape[1] != self.feature_dim:
            raise ValueError("features have incompatible feature dimension")
        standardized = (x - self.feature_mean) / self.feature_scale
        return self.intercept.reshape(1, -1) + standardized @ self.coefficients

    def apply(self, measurements: np.ndarray, features: np.ndarray | None = None) -> np.ndarray:
        """Return measurements with predicted bias subtracted."""

        values = _as_2d(measurements, "measurements")
        if values.shape[1] != self.target_dim:
            raise ValueError("measurements have incompatible target dimension")
        bias = self.predict(features, n_rows=values.shape[0])
        corrected = values.copy()
        valid = np.isfinite(values).all(axis=1) & np.isfinite(bias).all(axis=1)
        corrected[valid] = values[valid] - bias[valid]
        return corrected

    def to_dict(self) -> dict[str, Any]:
        """Serialize the model to a JSON-compatible mapping."""

        return {
            "target_dim": int(self.target_dim),
            "feature_dim": int(self.feature_dim),
            "intercept": self.intercept.tolist(),
            "coefficients": self.coefficients.tolist(),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "residual_std": self.residual_std.tolist(),
            "training_count": int(self.training_count),
            "ridge_alpha": float(self.ridge_alpha),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SensorBiasCorrectionModel":
        """Deserialize a model produced by :meth:`to_dict`."""

        return cls(
            target_dim=int(payload["target_dim"]),
            feature_dim=int(payload["feature_dim"]),
            intercept=np.asarray(payload["intercept"], dtype=float),
            coefficients=np.asarray(payload["coefficients"], dtype=float),
            feature_mean=np.asarray(payload["feature_mean"], dtype=float),
            feature_scale=np.asarray(payload["feature_scale"], dtype=float),
            residual_std=np.asarray(payload["residual_std"], dtype=float),
            training_count=int(payload["training_count"]),
            ridge_alpha=float(payload.get("ridge_alpha", 0.0)),
            metadata=payload.get("metadata", {}),
        )


def make_bias_training_examples(
    measurement_times_s: np.ndarray,
    measurement_values: np.ndarray,
    reference_times_s: np.ndarray,
    reference_values: np.ndarray,
    *,
    feature_values: np.ndarray | None = None,
    max_time_delta_s: float = 2.0,
) -> BiasTrainingExamples:
    """Match measurements to nearest reference values and compute residual bias."""

    if max_time_delta_s < 0.0:
        raise ValueError("max_time_delta_s must be nonnegative")
    measurement_times = np.asarray(measurement_times_s, dtype=float).reshape(-1)
    measurements = _as_2d(measurement_values, "measurement_values")
    reference_times = np.asarray(reference_times_s, dtype=float).reshape(-1)
    references = _as_2d(reference_values, "reference_values")
    if measurement_times.size != measurements.shape[0]:
        raise ValueError("measurement_times_s length must match measurement_values rows")
    if reference_times.size != references.shape[0]:
        raise ValueError("reference_times_s length must match reference_values rows")
    if measurements.shape[1] != references.shape[1]:
        raise ValueError("measurement_values and reference_values must have the same target dimension")
    if reference_times.size == 0:
        return BiasTrainingExamples(
            measured=np.empty((0, measurements.shape[1])),
            reference=np.empty((0, measurements.shape[1])),
            residual=np.empty((0, measurements.shape[1])),
            features=np.empty((0, 0 if feature_values is None else _as_2d(feature_values, "feature_values").shape[1])),
            time_delta_s=np.empty(0),
        )

    order = np.argsort(reference_times)
    reference_times = reference_times[order]
    references = references[order]
    nearest = nearest_time_indices(reference_times, measurement_times)
    delta_s = np.abs(reference_times[nearest] - measurement_times)
    valid = (
        np.isfinite(measurement_times)
        & np.isfinite(measurements).all(axis=1)
        & np.isfinite(references[nearest]).all(axis=1)
        & (delta_s <= float(max_time_delta_s))
    )
    if feature_values is None:
        features = np.empty((measurements.shape[0], 0), dtype=float)
    else:
        features = _as_2d(feature_values, "feature_values")
        if features.shape[0] != measurements.shape[0]:
            raise ValueError("feature_values rows must match measurement_values rows")
        valid &= np.isfinite(features).all(axis=1)
    measured = measurements[valid]
    reference = references[nearest[valid]]
    return BiasTrainingExamples(
        measured=measured,
        reference=reference,
        residual=measured - reference,
        features=features[valid],
        time_delta_s=delta_s[valid],
    )


def fit_sensor_bias_correction_from_examples(
    examples: BiasTrainingExamples,
    *,
    ridge_alpha: float = 1.0e-2,
    min_samples: int = 4,
    metadata: Mapping[str, Any] | None = None,
) -> SensorBiasCorrectionModel:
    """Fit a ridge-linear bias model from prepared examples."""

    if ridge_alpha < 0.0:
        raise ValueError("ridge_alpha must be nonnegative")
    if min_samples < 1:
        raise ValueError("min_samples must be positive")
    y = _as_2d(examples.residual, "examples.residual")
    x = _as_2d(examples.features, "examples.features")
    valid = np.isfinite(y).all(axis=1) & np.isfinite(x).all(axis=1)
    y = y[valid]
    x = x[valid]
    if y.shape[0] == 0:
        raise ValueError("no finite bias training examples")
    if x.shape[0] < int(min_samples):
        x = np.empty((y.shape[0], 0), dtype=float)

    feature_mean = _nanmean_or_zero(x)
    feature_scale = np.nanstd(x, axis=0) if x.shape[1] else np.empty(0, dtype=float)
    feature_scale = np.where(np.isfinite(feature_scale) & (feature_scale > 1.0e-12), feature_scale, 1.0)
    standardized = (x - feature_mean) / feature_scale if x.shape[1] else x
    design = np.column_stack([np.ones(y.shape[0]), standardized])
    regularizer = np.eye(design.shape[1]) * float(ridge_alpha)
    regularizer[0, 0] = 0.0
    lhs = design.T @ design + regularizer
    rhs = design.T @ y
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs) @ rhs
    residual = y - design @ beta
    residual_std = np.std(residual, axis=0) if residual.size else np.zeros(y.shape[1])
    return SensorBiasCorrectionModel(
        target_dim=y.shape[1],
        feature_dim=x.shape[1],
        intercept=beta[0],
        coefficients=beta[1:],
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        residual_std=residual_std,
        training_count=int(y.shape[0]),
        ridge_alpha=float(ridge_alpha),
        metadata={} if metadata is None else dict(metadata),
    )


def fit_sensor_bias_correction(
    measurement_times_s: np.ndarray,
    measurement_values: np.ndarray,
    reference_times_s: np.ndarray,
    reference_values: np.ndarray,
    *,
    feature_values: np.ndarray | None = None,
    max_time_delta_s: float = 2.0,
    ridge_alpha: float = 1.0e-2,
    min_samples: int = 4,
    metadata: Mapping[str, Any] | None = None,
) -> SensorBiasCorrectionModel:
    """Fit a bias correction model directly from timestamped measurements."""

    examples = make_bias_training_examples(
        measurement_times_s,
        measurement_values,
        reference_times_s,
        reference_values,
        feature_values=feature_values,
        max_time_delta_s=max_time_delta_s,
    )
    return fit_sensor_bias_correction_from_examples(
        examples,
        ridge_alpha=ridge_alpha,
        min_samples=min_samples,
        metadata=metadata,
    )


def _as_2d(values: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if out.ndim == 1:
        return out.reshape(-1, 1)
    if out.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional")
    return out


def _nanmean_or_zero(values: np.ndarray) -> np.ndarray:
    if values.shape[1] == 0:
        return np.empty(0, dtype=float)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(values, axis=0)
    return np.where(np.isfinite(mean), mean, 0.0)


__all__ = [
    "BiasTrainingExamples",
    "SensorBiasCorrectionModel",
    "fit_sensor_bias_correction",
    "fit_sensor_bias_correction_from_examples",
    "make_bias_training_examples",
]
