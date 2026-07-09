# pylint: disable=too-many-arguments,too-many-positional-arguments
"""Multisensor hierarchical-Dirichlet-process association utilities.

The helpers in this module implement a lightweight HDP/DDP-inspired
association proposal layer for heterogeneous multisensor multitarget tracking.
They do not replace an RFS, GLMB, PMBM, or nearest-neighbor tracker. Instead,
they convert sensor-specific measurement likelihoods into posterior probabilities
for shared global target atoms, a new-target birth atom, and a clutter atom.

The implementation is intentionally deterministic and array-oriented so that it
can be used as a proposal distribution or as an external pairwise cost matrix for
existing PyRecEst association routines.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Hashable, Literal

import numpy as np

AssociationKind = Literal["target", "birth", "clutter"]


@dataclass(frozen=True)
class HDPAssociationLabel:
    """Column label for one HDP association alternative.

    ``kind="target"`` denotes an existing global target atom and requires a
    nonnegative ``target_index``. ``kind="birth"`` denotes the shared new-target
    atom. ``kind="clutter"`` denotes a sensor-specific false-alarm process.
    """

    kind: AssociationKind
    target_index: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"target", "birth", "clutter"}:
            raise ValueError("kind must be one of 'target', 'birth', or 'clutter'")
        if self.kind == "target":
            object.__setattr__(
                self,
                "target_index",
                _nonnegative_integer(self.target_index, "target_index"),
            )
        elif self.target_index is not None:
            raise ValueError("target_index must be None for birth and clutter labels")


@dataclass(frozen=True)
class HDPAssociationDecision:
    """Maximum-posterior association decision for one measurement."""

    measurement_index: int
    label: HDPAssociationLabel
    probability: float
    log_weight: float


@dataclass(frozen=True)
class MultisensorHDPAssociationResult:
    """HDP association probabilities for one sensor.

    Rows correspond to measurements from ``sensor_id``. Columns correspond to
    ``labels`` and are ordered as existing targets, birth, and clutter.
    ``log_weights`` are unnormalized posterior log weights. ``probabilities`` are
    row-normalized posterior probabilities.
    """

    sensor_id: Hashable
    labels: tuple[HDPAssociationLabel, ...]
    log_weights: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        log_weights = np.asarray(self.log_weights, dtype=float)
        probabilities = np.asarray(self.probabilities, dtype=float)
        if log_weights.ndim != 2:
            raise ValueError("log_weights must be a two-dimensional array")
        if probabilities.shape != log_weights.shape:
            raise ValueError("probabilities must have the same shape as log_weights")
        if log_weights.shape[1] != len(self.labels):
            raise ValueError("labels must contain one entry per log-weight column")
        if np.any(np.isnan(log_weights)) or np.any(np.isposinf(log_weights)):
            raise ValueError("log_weights may contain finite values or -inf only")
        if np.any(~np.isfinite(probabilities)):
            raise ValueError("probabilities must be finite")
        if np.any(probabilities < 0.0):
            raise ValueError("probabilities must be nonnegative")
        if probabilities.size:
            row_sums = probabilities.sum(axis=1)
            if np.any(~np.isclose(row_sums, 1.0)):
                raise ValueError("probability rows must sum to one")
        object.__setattr__(self, "log_weights", log_weights)
        object.__setattr__(self, "probabilities", probabilities)

    @property
    def num_measurements(self) -> int:
        """Number of measurements represented by this result."""
        return int(self.probabilities.shape[0])

    @property
    def num_existing_targets(self) -> int:
        """Number of existing target columns represented by this result."""
        return len(self.target_column_indices())

    def target_column_indices(self) -> tuple[int, ...]:
        """Return result-column indices corresponding to existing targets."""
        return tuple(
            column_index
            for column_index, label in enumerate(self.labels)
            if label.kind == "target"
        )

    def target_probability_matrix(self) -> np.ndarray:
        """Return existing-target probabilities as ``(num_targets, num_measurements)``.

        The returned orientation matches the pairwise association cost interface
        used by PyRecEst nearest-neighbor trackers: target rows, measurement
        columns.
        """
        target_columns = self.target_column_indices()
        return self.probabilities[:, target_columns].T.copy()

    def target_log_weight_matrix(self) -> np.ndarray:
        """Return existing-target log weights as ``(num_targets, num_measurements)``."""
        target_columns = self.target_column_indices()
        return self.log_weights[:, target_columns].T.copy()

    def target_cost_matrix(self, *, probability_floor: float = 1e-300) -> np.ndarray:
        """Return ``-log(probability)`` costs for existing target associations.

        ``probability_floor`` prevents infinite costs for zero-probability target
        hypotheses when the matrix is passed to algorithms that expect finite
        pairwise costs.
        """
        probability_floor = _finite_scalar(probability_floor, "probability_floor")
        if not 0.0 < probability_floor <= 1.0:
            raise ValueError("probability_floor must be in (0, 1]")
        probabilities = np.clip(
            self.target_probability_matrix(),
            probability_floor,
            1.0,
        )
        return -np.log(probabilities)

    def best_label_indices(self) -> tuple[int, ...]:
        """Return the maximum-posterior label-column index for each measurement."""
        if self.num_measurements == 0:
            return ()
        return tuple(int(index) for index in np.argmax(self.probabilities, axis=1))

    def best_assignments(self) -> tuple[HDPAssociationDecision, ...]:
        """Return maximum-posterior association decisions for all measurements."""
        decisions = []
        for measurement_index, label_index in enumerate(self.best_label_indices()):
            decisions.append(
                HDPAssociationDecision(
                    measurement_index=measurement_index,
                    label=self.labels[label_index],
                    probability=float(
                        self.probabilities[measurement_index, label_index]
                    ),
                    log_weight=float(self.log_weights[measurement_index, label_index]),
                )
            )
        return tuple(decisions)


def predict_survival_weighted_hdp_masses(
    global_target_weights: Any,
    survival_probabilities: Any = 1.0,
) -> np.ndarray:
    """Apply a DDP-style survival prediction to global target-atom masses.

    This is a small utility for the temporal part of a dependent Dirichlet
    process: surviving atoms keep their identity but have their masses discounted
    by target survival probabilities before a new birth mass is supplied to
    :func:`multisensor_hdp_association`.
    """
    weights = _nonnegative_vector(global_target_weights, "global_target_weights")
    survival = _probability_vector(
        survival_probabilities,
        len(weights),
        "survival_probabilities",
    )
    return weights * survival


def multisensor_hdp_association(
    log_likelihoods_by_sensor: Mapping[Hashable, Any],
    global_target_weights: Any,
    *,
    global_birth_weight: float = 1.0,
    sensor_target_counts: Mapping[Hashable, Any] | Any | None = None,
    sensor_concentrations: Mapping[Hashable, Any] | Any = 1.0,
    detection_probabilities: Mapping[Hashable, Any] | Any = 1.0,
    birth_log_likelihoods: Mapping[Hashable, Any] | Any | None = None,
    clutter_log_likelihoods: Mapping[Hashable, Any] | Any | None = None,
    clutter_weights: Mapping[Hashable, Any] | Any = 1.0,
) -> dict[Hashable, MultisensorHDPAssociationResult]:
    """Compute HDP-style association probabilities for multiple sensors.

    Parameters
    ----------
    log_likelihoods_by_sensor : mapping
        Maps each sensor ID to an array of shape
        ``(num_measurements_for_sensor, num_existing_targets)`` containing
        ``log p(z | target)`` values. Use ``-np.inf`` for gated-out or impossible
        target-measurement pairs.
    global_target_weights : array-like, shape ``(num_existing_targets,)``
        Nonnegative global HDP atom masses for existing targets. These can be
        posterior target masses from the previous scan or the output of
        :func:`predict_survival_weighted_hdp_masses`.
    global_birth_weight : float, optional
        Nonnegative HDP mass assigned to the shared new-target atom.
    sensor_target_counts : mapping, scalar, array-like, optional
        Sensor-local reinforcement counts. A mapping value for a sensor may be a
        scalar or a vector with one entry per existing target. Missing values
        default to zero counts.
    sensor_concentrations : mapping or scalar, optional
        Positive sensor-specific HDP concentration ``alpha_m``. Larger values
        make sensor-local priors follow the global target/birth masses more
        strongly; smaller values emphasize sensor-local counts.
    detection_probabilities : mapping, scalar, or array-like, optional
        Sensor-specific detection probabilities for existing targets. A scalar is
        broadcast to all targets; a vector must match the number of targets.
    birth_log_likelihoods : mapping, scalar, array-like, optional
        Per-sensor marginal log likelihood for the birth alternative. Defaults to
        zero for all measurements.
    clutter_log_likelihoods : mapping, scalar, array-like, optional
        Per-sensor clutter log likelihood. Defaults to zero for all measurements.
    clutter_weights : mapping or scalar, optional
        Nonnegative sensor-specific false-alarm mass.

    Returns
    -------
    dict
        Mapping from sensor ID to :class:`MultisensorHDPAssociationResult`.

    Notes
    -----
    For sensor ``m`` and existing target ``ell``, the unnormalized log weight is

    ``log(n_mell + alpha_m * beta_ell) + log(p_D_mell) + log p_m(z | x_ell)``.

    The birth and clutter alternatives use
    ``log(alpha_m * beta_birth) + birth_log_likelihood`` and
    ``log(clutter_weight_m) + clutter_log_likelihood`` respectively.
    """
    if not isinstance(log_likelihoods_by_sensor, Mapping):
        raise ValueError("log_likelihoods_by_sensor must be a mapping")

    target_weights = _nonnegative_vector(global_target_weights, "global_target_weights")
    num_targets = len(target_weights)
    global_birth_weight = _nonnegative_finite_scalar(
        global_birth_weight,
        "global_birth_weight",
    )
    global_mass = float(target_weights.sum() + global_birth_weight)
    if global_mass <= 0.0:
        raise ValueError(
            "global target and birth weights must contain positive total mass"
        )

    base_target_weights = target_weights / global_mass
    base_birth_weight = global_birth_weight / global_mass
    labels = tuple(HDPAssociationLabel("target", index) for index in range(num_targets))
    labels += (HDPAssociationLabel("birth"), HDPAssociationLabel("clutter"))

    results: dict[Hashable, MultisensorHDPAssociationResult] = {}
    for sensor_id, raw_log_likelihoods in log_likelihoods_by_sensor.items():
        log_likelihoods = _log_likelihood_matrix(
            raw_log_likelihoods,
            num_targets,
            f"log_likelihoods_by_sensor[{sensor_id!r}]",
        )
        num_measurements = log_likelihoods.shape[0]

        counts = _nonnegative_vector_or_scalar(
            _sensor_parameter(
                sensor_target_counts,
                sensor_id,
                "sensor_target_counts",
                default=0.0,
            ),
            num_targets,
            f"sensor_target_counts[{sensor_id!r}]",
        )
        concentration = _positive_finite_scalar(
            _sensor_parameter(
                sensor_concentrations,
                sensor_id,
                "sensor_concentrations",
            ),
            f"sensor_concentrations[{sensor_id!r}]",
        )
        detection = _probability_vector(
            _sensor_parameter(
                detection_probabilities,
                sensor_id,
                "detection_probabilities",
            ),
            num_targets,
            f"detection_probabilities[{sensor_id!r}]",
        )
        birth_log_likelihood = _log_likelihood_vector(
            _sensor_parameter(
                birth_log_likelihoods,
                sensor_id,
                "birth_log_likelihoods",
                default=0.0,
            ),
            num_measurements,
            f"birth_log_likelihoods[{sensor_id!r}]",
        )
        clutter_log_likelihood = _log_likelihood_vector(
            _sensor_parameter(
                clutter_log_likelihoods,
                sensor_id,
                "clutter_log_likelihoods",
                default=0.0,
            ),
            num_measurements,
            f"clutter_log_likelihoods[{sensor_id!r}]",
        )
        clutter_weight = _nonnegative_finite_scalar(
            _sensor_parameter(clutter_weights, sensor_id, "clutter_weights"),
            f"clutter_weights[{sensor_id!r}]",
        )

        target_prior_masses = (counts + concentration * base_target_weights) * detection
        target_log_priors = _safe_log(target_prior_masses)
        existing_log_weights = log_likelihoods + target_log_priors[None, :]

        birth_log_weights = birth_log_likelihood + _safe_log(
            concentration * base_birth_weight
        )
        clutter_log_weights = clutter_log_likelihood + _safe_log(clutter_weight)
        log_weights = np.concatenate(
            [
                existing_log_weights,
                birth_log_weights[:, None],
                clutter_log_weights[:, None],
            ],
            axis=1,
        )
        probabilities = _softmax_rows(log_weights)
        results[sensor_id] = MultisensorHDPAssociationResult(
            sensor_id=sensor_id,
            labels=labels,
            log_weights=log_weights,
            probabilities=probabilities,
        )
    return results


def _sensor_parameter(
    parameter: Mapping[Hashable, Any] | Any | None,
    sensor_id: Hashable,
    name: str,
    *,
    default: Any | None = None,
) -> Any:
    if parameter is None:
        return default
    if isinstance(parameter, Mapping):
        if sensor_id not in parameter:
            raise KeyError(f"{name} is missing an entry for sensor {sensor_id!r}")
        return parameter[sensor_id]
    return parameter


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer")
    if not isinstance(value, Integral):
        raise ValueError(f"{name} must be a nonnegative integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _finite_scalar(value: Any, name: str) -> float:
    value_array = np.asarray(value)
    if value_array.shape != () or value_array.dtype == np.bool_:
        raise ValueError(f"{name} must be a finite scalar")
    try:
        scalar = float(value_array.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite scalar")
    return scalar


def _positive_finite_scalar(value: Any, name: str) -> float:
    scalar = _finite_scalar(value, name)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _nonnegative_finite_scalar(value: Any, name: str) -> float:
    scalar = _finite_scalar(value, name)
    if scalar < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return scalar


def _nonnegative_vector(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if np.any(~np.isfinite(vector)) or np.any(vector < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    return vector


def _nonnegative_vector_or_scalar(value: Any, length: int, name: str) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    if values.shape == ():
        scalar = _nonnegative_finite_scalar(values, name)
        return np.full(length, scalar, dtype=float)
    if values.shape != (length,):
        raise ValueError(f"{name} must be scalar or have shape ({length},)")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    return values


def _probability_vector(value: Any, length: int, name: str) -> np.ndarray:
    probabilities = np.asarray(value, dtype=float)
    if probabilities.shape == ():
        scalar = _finite_scalar(probabilities, name)
        probabilities = np.full(length, scalar, dtype=float)
    elif probabilities.shape != (length,):
        raise ValueError(f"{name} must be scalar or have shape ({length},)")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError(f"{name} must contain probabilities in [0, 1]")
    return probabilities


def _log_likelihood_matrix(value: Any, num_targets: int, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != num_targets:
        raise ValueError(f"{name} must have shape (num_measurements, {num_targets})")
    _validate_log_values(matrix, name)
    return matrix


def _log_likelihood_vector(value: Any, length: int, name: str) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    if values.shape == ():
        _validate_log_values(values, name)
        return np.full(length, float(values.item()), dtype=float)
    if values.shape != (length,):
        raise ValueError(f"{name} must be scalar or have shape ({length},)")
    _validate_log_values(values, name)
    return values


def _validate_log_values(values: np.ndarray, name: str) -> None:
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError(f"{name} may contain finite values or -inf only")


def _safe_log(values: Any) -> np.ndarray | float:
    values_array = np.asarray(values, dtype=float)
    with np.errstate(divide="ignore"):
        logged = np.where(values_array > 0.0, np.log(values_array), -np.inf)
    if logged.shape == ():
        return float(logged.item())
    return logged


def _softmax_rows(log_weights: np.ndarray) -> np.ndarray:
    if log_weights.shape[0] == 0:
        return np.empty_like(log_weights, dtype=float)
    row_max = np.max(log_weights, axis=1, keepdims=True)
    if np.any(np.isneginf(row_max)):
        raise ValueError("each measurement must have at least one feasible association")
    shifted = np.exp(log_weights - row_max)
    return shifted / shifted.sum(axis=1, keepdims=True)
