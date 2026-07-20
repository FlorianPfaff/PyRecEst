"""Experimental multisensor dependent-Dirichlet-process association helpers.

The utilities in this module implement a small Bayesian-nonparametric association
layer that can be used as a proposal distribution or diagnostic component inside
larger multisensor multitarget trackers.  The model keeps a global set of target
atoms, lets each sensor provide its own likelihood block, and returns posterior
responsibilities over existing targets, a shared birth atom, and clutter.

This is intentionally not a complete RFS/GLMB/PMBM tracker.  It is a lightweight
building block for experiments in which heterogeneous sensors share latent target
identities but retain sensor-specific detection, birth, and clutter models.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import log
from typing import Any

import numpy as np

BIRTH_LABEL = "__birth__"
CLUTTER_LABEL = "__clutter__"


@dataclass(frozen=True)
class SensorAssociationBlock:
    """Sensor-specific likelihood block for one multisensor association update.

    Parameters
    ----------
    sensor_id:
        Stable identifier used in the returned posterior dictionary.
    log_likelihoods:
        Matrix with shape ``(num_measurements, num_targets)``.  Entry ``(i, j)``
        is ``log f_m(z_i | x_j)`` for sensor ``m``.  One-dimensional arrays are
        accepted only when there is a single existing target.
    detection_probabilities:
        Scalar or length-``num_targets`` vector with target-detection
        probabilities for this sensor.  These values down-weight existing-target
        assignment scores but do not model the full missed-detection event.
    birth_log_weights:
        Scalar or length-``num_measurements`` vector for the marginal birth
        evidence ``log int f_m(z_i | x) dH_birth(x)``.
    clutter_log_weights:
        Scalar or length-``num_measurements`` vector for clutter evidence, for
        example ``log(lambda_m * kappa_m(z_i))``.
    concentration:
        Local HDP/DDP concentration for this sensor.  Larger values make sensor
        assignments rely more strongly on the global target and birth weights.
    """

    sensor_id: str
    log_likelihoods: Any
    detection_probabilities: float | Sequence[float] = 1.0
    birth_log_weights: float | Sequence[float] = 0.0
    clutter_log_weights: float | Sequence[float] = 0.0
    concentration: float = 1.0


@dataclass(frozen=True)
class SensorAssociationPosterior:
    """Posterior responsibilities for one sensor's measurements."""

    sensor_id: str
    labels: tuple[Any, ...]
    responsibilities: np.ndarray
    expected_counts: np.ndarray
    log_normalizers: np.ndarray
    hard_assignments: tuple[Any, ...]

    @property
    def target_responsibilities(self) -> np.ndarray:
        """Return responsibilities for existing targets only."""
        return self.responsibilities[:, : max(len(self.labels) - 2, 0)]

    @property
    def birth_responsibilities(self) -> np.ndarray:
        """Return responsibilities assigned to the shared birth atom."""
        return self.responsibilities[:, -2]

    @property
    def clutter_responsibilities(self) -> np.ndarray:
        """Return responsibilities assigned to clutter."""
        return self.responsibilities[:, -1]


@dataclass(frozen=True)
class MultisensorDDPAssociationResult:
    """Result of a multisensor DDP/HDP-style association update."""

    target_labels: tuple[Any, ...]
    assignment_labels: tuple[Any, ...]
    sensor_posteriors: Mapping[str, SensorAssociationPosterior]
    updated_target_weights: np.ndarray
    updated_birth_weight: float
    expected_target_counts: np.ndarray
    expected_birth_count: float
    expected_clutter_count: float

    def posterior_for_sensor(self, sensor_id: str) -> SensorAssociationPosterior:
        """Return the posterior block for ``sensor_id``."""
        return self.sensor_posteriors[sensor_id]


def predict_ddp_base_weights(
    target_weights: Sequence[float],
    survival_probabilities: float | Sequence[float] = 1.0,
    *,
    birth_weight: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Predict global DDP base weights through target survival and birth mass.

    The returned tuple contains normalized existing-target weights and a
    normalized birth weight.  It is a compact prior-prediction step for carrying
    target atoms from one scan to the next before calling
    :func:`multisensor_ddp_association_update`.
    """
    weights = _coerce_nonnegative_vector(target_weights, None, "target_weights")
    survival = _coerce_probability_vector(
        survival_probabilities, len(weights), "survival_probabilities"
    )
    birth = _as_nonnegative_float(birth_weight, "birth_weight")
    survived = weights * survival
    return _normalize_base_weights(survived, birth)


def multisensor_ddp_association_update(
    target_weights: Sequence[float],
    sensor_blocks: Sequence[SensorAssociationBlock],
    *,
    target_labels: Sequence[Any] | None = None,
    birth_weight: float = 1.0,
    prior_strength: float = 1.0,
    point_target: bool = False,
) -> MultisensorDDPAssociationResult:
    """Fuse sensor-specific likelihoods into shared-target responsibilities.

    ``target_weights`` and ``birth_weight`` define the global DDP/HDP base
    measure over existing target atoms and a shared birth atom.  Each
    :class:`SensorAssociationBlock` contributes local likelihoods and
    sensor-specific birth/clutter evidence.  The result can be used directly as a
    soft association proposal or converted to hard assignments via the returned
    ``hard_assignments`` fields.

    When ``point_target`` is true, each sensor posterior is projected to a greedy
    one-to-one assignment over existing targets.  Birth and clutter may still be
    reused by multiple measurements.  This gives a pragmatic approximation for
    point-target sensors while retaining the same input and output format.
    """
    if not sensor_blocks:
        raise ValueError("sensor_blocks must contain at least one block")

    weights = _coerce_nonnegative_vector(target_weights, None, "target_weights")
    num_targets = len(weights)
    labels = (
        tuple(range(num_targets)) if target_labels is None else tuple(target_labels)
    )
    if len(labels) != num_targets:
        raise ValueError("target_labels must have the same length as target_weights")

    prior_strength = _as_nonnegative_float(prior_strength, "prior_strength")
    target_base, normalized_birth_weight = _normalize_base_weights(
        weights, birth_weight
    )
    assignment_labels = (*labels, BIRTH_LABEL, CLUTTER_LABEL)

    sensor_posteriors: dict[str, SensorAssociationPosterior] = {}
    expected_target_counts = np.zeros(num_targets, dtype=float)
    expected_birth_count = 0.0
    expected_clutter_count = 0.0

    seen_sensor_ids: set[str] = set()
    for block in sensor_blocks:
        if block.sensor_id in seen_sensor_ids:
            raise ValueError(f"duplicate sensor_id {block.sensor_id!r}")
        seen_sensor_ids.add(block.sensor_id)

        log_likelihoods = _coerce_log_likelihoods(
            block.log_likelihoods, num_targets, block.sensor_id
        )
        num_measurements = int(log_likelihoods.shape[0])
        detection_probabilities = _coerce_probability_vector(
            block.detection_probabilities,
            num_targets,
            f"{block.sensor_id}.detection_probabilities",
        )
        birth_log_weights = _coerce_log_weight_vector(
            block.birth_log_weights,
            num_measurements,
            f"{block.sensor_id}.birth_log_weights",
        )
        clutter_log_weights = _coerce_log_weight_vector(
            block.clutter_log_weights,
            num_measurements,
            f"{block.sensor_id}.clutter_log_weights",
        )
        concentration = _as_positive_float(
            block.concentration, f"{block.sensor_id}.concentration"
        )

        log_scores = _association_log_scores(
            log_likelihoods,
            target_base,
            detection_probabilities,
            birth_log_weights,
            clutter_log_weights,
            normalized_birth_weight,
            concentration,
        )
        if point_target:
            responsibilities = _greedy_point_target_projection(log_scores, num_targets)
            log_normalizers = (
                np.max(log_scores, axis=1)
                if num_measurements
                else np.empty(0, dtype=float)
            )
        else:
            responsibilities, log_normalizers = _rowwise_softmax_from_log_scores(
                log_scores
            )

        expected_counts = responsibilities.sum(axis=0)
        expected_target_counts += expected_counts[:num_targets]
        expected_birth_count += float(expected_counts[-2])
        expected_clutter_count += float(expected_counts[-1])
        hard_assignments = tuple(
            assignment_labels[int(index)]
            for index in np.argmax(responsibilities, axis=1)
        )

        sensor_posteriors[block.sensor_id] = SensorAssociationPosterior(
            sensor_id=block.sensor_id,
            labels=assignment_labels,
            responsibilities=responsibilities,
            expected_counts=expected_counts,
            log_normalizers=log_normalizers,
            hard_assignments=hard_assignments,
        )

    posterior_target_pseudocounts = (
        prior_strength * target_base + expected_target_counts
    )
    posterior_birth_pseudocount = (
        prior_strength * normalized_birth_weight + expected_birth_count
    )
    posterior_total = float(
        np.sum(posterior_target_pseudocounts) + posterior_birth_pseudocount
    )
    if posterior_total <= 0.0:
        raise ValueError(
            "posterior target and birth mass is zero; increase prior_strength or birth_weight"
        )

    updated_target_weights = posterior_target_pseudocounts / posterior_total
    updated_birth_weight = float(posterior_birth_pseudocount / posterior_total)

    return MultisensorDDPAssociationResult(
        target_labels=labels,
        assignment_labels=assignment_labels,
        sensor_posteriors=sensor_posteriors,
        updated_target_weights=updated_target_weights,
        updated_birth_weight=updated_birth_weight,
        expected_target_counts=expected_target_counts,
        expected_birth_count=expected_birth_count,
        expected_clutter_count=expected_clutter_count,
    )


def _association_log_scores(
    log_likelihoods: np.ndarray,
    target_base: np.ndarray,
    detection_probabilities: np.ndarray,
    birth_log_weights: np.ndarray,
    clutter_log_weights: np.ndarray,
    birth_base: float,
    concentration: float,
) -> np.ndarray:
    num_measurements, num_targets = log_likelihoods.shape
    target_prior = np.full(num_targets, -np.inf, dtype=float)
    positive_target_mask = target_base > 0.0
    target_prior[positive_target_mask] = log(concentration) + np.log(
        target_base[positive_target_mask]
    )

    detection_log = np.full(num_targets, -np.inf, dtype=float)
    positive_detection_mask = detection_probabilities > 0.0
    detection_log[positive_detection_mask] = np.log(
        detection_probabilities[positive_detection_mask]
    )

    existing_scores = (
        log_likelihoods + target_prior.reshape(1, -1) + detection_log.reshape(1, -1)
    )
    birth_prior = -np.inf if birth_base <= 0.0 else log(concentration) + log(birth_base)
    birth_scores = (birth_log_weights + birth_prior).reshape(num_measurements, 1)
    clutter_scores = clutter_log_weights.reshape(num_measurements, 1)
    return np.concatenate((existing_scores, birth_scores, clutter_scores), axis=1)


def _rowwise_softmax_from_log_scores(
    log_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if log_scores.ndim != 2:
        raise ValueError("log_scores must be two-dimensional")
    if log_scores.shape[0] == 0:
        return np.empty_like(log_scores, dtype=float), np.empty(0, dtype=float)
    row_max = np.max(log_scores, axis=1)
    if np.any(~np.isfinite(row_max)):
        raise ValueError(
            "each measurement must have at least one finite association score"
        )
    shifted = log_scores - row_max.reshape(-1, 1)
    unnormalized = np.exp(shifted)
    normalizers = unnormalized.sum(axis=1)
    responsibilities = unnormalized / normalizers.reshape(-1, 1)
    log_normalizers = row_max + np.log(normalizers)
    return responsibilities, log_normalizers


def _greedy_point_target_projection(
    log_scores: np.ndarray, num_targets: int
) -> np.ndarray:
    if log_scores.shape[0] == 0:
        return np.empty_like(log_scores, dtype=float)
    if np.any(~np.isfinite(np.max(log_scores, axis=1))):
        raise ValueError(
            "each measurement must have at least one finite association score"
        )

    responsibilities = np.zeros_like(log_scores, dtype=float)
    used_targets: set[int] = set()
    priorities: list[tuple[float, int]] = []
    for measurement_index, row in enumerate(log_scores):
        finite_scores = np.sort(row[np.isfinite(row)])
        margin = (
            float("inf")
            if finite_scores.size == 1
            else float(finite_scores[-1] - finite_scores[-2])
        )
        priorities.append((margin, measurement_index))

    for _, measurement_index in sorted(priorities, reverse=True):
        row = log_scores[measurement_index]
        selected_column = None
        for column in np.argsort(row)[::-1]:
            if not np.isfinite(row[column]):
                continue
            if column < num_targets and int(column) in used_targets:
                continue
            selected_column = int(column)
            break
        if selected_column is None:
            raise ValueError(
                "point_target projection has no finite feasible association "
                f"for measurement {measurement_index}"
            )
        responsibilities[measurement_index, selected_column] = 1.0
        if selected_column < num_targets:
            used_targets.add(selected_column)
    return responsibilities


def _normalize_base_weights(
    target_weights: np.ndarray, birth_weight: float
) -> tuple[np.ndarray, float]:
    birth = _as_nonnegative_float(birth_weight, "birth_weight")
    total = float(np.sum(target_weights) + birth)
    if total <= 0.0:
        raise ValueError(
            "target_weights and birth_weight must contain positive total mass"
        )
    return target_weights.astype(float, copy=False) / total, float(birth / total)


def _reject_temporal_values(value: Any, name: str) -> None:
    if _contains_temporal_value(value):
        raise ValueError(f"{name} must not contain datetime64 or timedelta64 values")


def _contains_temporal_value(value: Any, *, _depth: int = 0) -> bool:
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return True
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if _is_temporal_dtype(array.dtype):
        return True
    if array.dtype != object or _depth >= 4:
        return False
    return any(
        _contains_temporal_value(item, _depth=_depth + 1) for item in array.ravel()
    )


def _is_temporal_dtype(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.datetime64) or np.issubdtype(dtype, np.timedelta64)


def _coerce_log_likelihoods(value: Any, num_targets: int, sensor_id: str) -> np.ndarray:
    _reject_temporal_values(value, f"{sensor_id}.log_likelihoods")
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        if num_targets != 1:
            raise ValueError(
                f"{sensor_id}.log_likelihoods must have shape (num_measurements, {num_targets})"
            )
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{sensor_id}.log_likelihoods must be two-dimensional")
    if array.shape[1] != num_targets:
        raise ValueError(f"{sensor_id}.log_likelihoods must have {num_targets} columns")
    if np.any(np.isnan(array)):
        raise ValueError(f"{sensor_id}.log_likelihoods must not contain NaN")
    return array.astype(float, copy=False)


def _coerce_log_weight_vector(
    value: float | Sequence[float], length: int, name: str
) -> np.ndarray:
    _reject_temporal_values(value, name)
    array = np.asarray(value, dtype=float)
    if array.shape == ():
        scalar = float(array.item())
        if np.isnan(scalar):
            raise ValueError(f"{name} must not contain NaN")
        return np.full(length, scalar, dtype=float)
    if array.ndim != 1 or array.shape[0] != length:
        raise ValueError(f"{name} must be a scalar or length-{length} vector")
    if np.any(np.isnan(array)):
        raise ValueError(f"{name} must not contain NaN")
    return array.astype(float, copy=False)


def _coerce_nonnegative_vector(
    value: Sequence[float], expected_length: int | None, name: str
) -> np.ndarray:
    _reject_temporal_values(value, name)
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if expected_length is not None and array.shape[0] != expected_length:
        raise ValueError(f"{name} must have length {expected_length}")
    if np.any(~np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    return array.astype(float, copy=False)


def _coerce_probability_vector(
    value: float | Sequence[float], length: int, name: str
) -> np.ndarray:
    _reject_temporal_values(value, name)
    array = np.asarray(value, dtype=float)
    if array.shape == ():
        scalar = float(array.item())
        if not np.isfinite(scalar) or scalar < 0.0 or scalar > 1.0:
            raise ValueError(f"{name} must contain probabilities in [0, 1]")
        return np.full(length, scalar, dtype=float)
    if array.ndim != 1 or array.shape[0] != length:
        raise ValueError(f"{name} must be a scalar or length-{length} vector")
    if np.any(~np.isfinite(array)) or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must contain probabilities in [0, 1]")
    return array.astype(float, copy=False)


def _as_nonnegative_float(value: float, name: str) -> float:
    _reject_temporal_values(value, name)
    try:
        scalar = float(np.asarray(value).item())
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{name} must be a scalar") from exc
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return scalar


def _as_positive_float(value: float, name: str) -> float:
    scalar = _as_nonnegative_float(value, name)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


__all__ = [
    "BIRTH_LABEL",
    "CLUTTER_LABEL",
    "MultisensorDDPAssociationResult",
    "SensorAssociationBlock",
    "SensorAssociationPosterior",
    "multisensor_ddp_association_update",
    "predict_ddp_base_weights",
]
