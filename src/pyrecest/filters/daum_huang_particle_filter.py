"""Daum-Huang Gaussian particle-flow filters for Euclidean particles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

# pylint: disable=no-name-in-module,no-member,too-many-arguments,too-many-positional-arguments
from pyrecest.backend import asarray, eye, ones, to_numpy
from pyrecest.distributions.nonperiodic.linear_dirac_distribution import (
    LinearDiracDistribution,
)
from pyrecest.models import (
    AdditiveNoiseMeasurementModel,
    LinearGaussianMeasurementModel,
)

from .euclidean_particle_filter import EuclideanParticleFilter

FlowType = Literal["edh", "ledh"]


@dataclass
class GaussianParticleFlowInfo:
    """Diagnostics recorded during a Gaussian particle-flow update."""

    flow_type: FlowType
    lambdas: list[float] = field(default_factory=lambda: [0.0])
    mean_trace: list[Any] = field(default_factory=list)
    cov_trace: list[Any] = field(default_factory=list)
    step_mean_displacements: list[float] = field(default_factory=list)
    step_max_displacements: list[float] = field(default_factory=list)
    linearization_points: list[Any] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        return max(0, len(self.lambdas) - 1)


def edh_particle_flow(
    particles,
    measurement_model,
    measurement=None,
    *,
    measurement_noise_covariance=None,
    weights=None,
    n_steps: int = 20,
    step_schedule=None,
    jitter: float = 1e-8,
    return_info: bool = False,
):
    """Run exact Daum-Huang flow with ensemble-mean linearization."""
    return gaussian_particle_flow_update(
        particles,
        measurement_model,
        measurement,
        flow_type="edh",
        measurement_noise_covariance=measurement_noise_covariance,
        weights=weights,
        n_steps=n_steps,
        step_schedule=step_schedule,
        jitter=jitter,
        return_info=return_info,
    )


def ledh_particle_flow(
    particles,
    measurement_model,
    measurement=None,
    *,
    measurement_noise_covariance=None,
    weights=None,
    n_steps: int = 20,
    step_schedule=None,
    jitter: float = 1e-8,
    return_info: bool = False,
):
    """Run localized Daum-Huang flow with particle-wise linearizations."""
    return gaussian_particle_flow_update(
        particles,
        measurement_model,
        measurement,
        flow_type="ledh",
        measurement_noise_covariance=measurement_noise_covariance,
        weights=weights,
        n_steps=n_steps,
        step_schedule=step_schedule,
        jitter=jitter,
        return_info=return_info,
    )


def gaussian_particle_flow_update(
    particles,
    measurement_model,
    measurement=None,
    *,
    flow_type: FlowType = "edh",
    measurement_noise_covariance=None,
    weights=None,
    n_steps: int = 20,
    step_schedule=None,
    jitter: float = 1e-8,
    return_info: bool = False,
):
    """Transport particles through a Gaussian measurement-likelihood homotopy."""
    flow_type = _validate_flow_type(flow_type)
    jitter = _validate_nonnegative_float(jitter, "jitter")
    X = _as_particle_matrix_np(particles)
    w = _as_weights_np(weights, X.shape[0]) if weights is not None else None
    y = _resolve_measurement_vector_np(measurement_model, measurement)
    R = _resolve_noise_covariance_np(
        measurement_model,
        measurement_noise_covariance,
        y.size,
    )
    deltas = _lambda_deltas_np(n_steps, step_schedule)
    info = GaussianParticleFlowInfo(flow_type=flow_type)
    _record_flow_state(info, X, w)

    lam = 0.0
    for delta in deltas:
        before = X.copy()
        if flow_type == "edh":
            X = _edh_step_np(X, w, measurement_model, y, R, float(delta), jitter)
            info.linearization_points.append(asarray(_weighted_mean_np(before, w)))
        else:
            X = _ledh_step_np(X, w, measurement_model, y, R, float(delta), jitter)
            info.linearization_points.append(asarray(before.copy()))

        lam += float(delta)
        displacement = np.linalg.norm(X - before, axis=1)
        info.step_mean_displacements.append(float(np.mean(displacement)))
        info.step_max_displacements.append(float(np.max(displacement)))
        info.lambdas.append(float(lam))
        _record_flow_state(info, X, w)

    result = asarray(X)
    return (result, info) if return_info else result


def gaussian_flow_affine_increment(
    particles,
    mean,
    covariance,
    measurement_matrix,
    measurement,
    measurement_noise_covariance,
    delta_lambda: float,
    *,
    jitter: float = 1e-8,
):
    """Apply one exact affine Gaussian bridge increment for a linear model."""
    jitter = _validate_nonnegative_float(jitter, "jitter")
    X = _as_particle_matrix_np(particles)
    m0, P0, H, y, R = _validate_linear_bridge_np(
        mean,
        covariance,
        measurement_matrix,
        measurement,
        measurement_noise_covariance,
        X.shape[1],
        jitter,
    )
    return asarray(
        _gaussian_flow_affine_increment_np(
            X,
            m0,
            P0,
            H,
            y,
            R,
            delta_lambda,
            jitter=jitter,
        )
    )


def gaussian_bridge_moments(
    mean,
    covariance,
    measurement_matrix,
    measurement,
    measurement_noise_covariance,
    delta_lambda: float,
    *,
    jitter: float = 1e-8,
):
    """Return Gaussian moments after a likelihood-power increment."""
    jitter = _validate_nonnegative_float(jitter, "jitter")
    m0, P0, H, y, R = _validate_linear_bridge_np(
        mean,
        covariance,
        measurement_matrix,
        measurement,
        measurement_noise_covariance,
        _as_vector_np(mean, "mean").size,
        jitter,
    )
    mean_next, cov_next = _gaussian_bridge_moments_np(
        m0,
        P0,
        H,
        y,
        R,
        delta_lambda,
        jitter=jitter,
    )
    return asarray(mean_next), asarray(cov_next)


def gaussian_particle_flow_drift(
    particles,
    mean,
    covariance,
    measurement_matrix,
    measurement,
    measurement_noise_covariance,
    *,
    jitter: float = 1e-8,
):
    """Closed-form affine Gaussian homotopy drift for a linear measurement."""
    jitter = _validate_nonnegative_float(jitter, "jitter")
    X = _as_particle_matrix_np(particles)
    m, P, H, y, R = _validate_linear_bridge_np(
        mean,
        covariance,
        measurement_matrix,
        measurement,
        measurement_noise_covariance,
        X.shape[1],
        jitter,
    )
    Rinv = np.linalg.inv(R)
    gram = H.T @ Rinv @ H
    mean_drift = P @ H.T @ Rinv @ (y - H @ m)
    affine_drift = -0.5 * P @ gram
    return asarray(mean_drift[None, :] + (X - m[None, :]) @ affine_drift.T)


class DaumHuangParticleFlowFilter(EuclideanParticleFilter):
    """Euclidean particle filter updated by exact Daum-Huang Gaussian flow."""

    flow_type: FlowType = "edh"

    def __init__(
        self,
        n_particles,
        dim,
        *,
        flow_type: FlowType | None = None,
        n_steps: int = 20,
        step_schedule=None,
        jitter: float = 1e-8,
    ):
        super().__init__(n_particles=n_particles, dim=dim)
        self.flow_type = _validate_flow_type(
            self.flow_type if flow_type is None else flow_type
        )
        self.n_steps = _validate_positive_int(n_steps, "n_steps")
        self.step_schedule = (
            None
            if step_schedule is None
            else tuple(float(value) for value in step_schedule)
        )
        self.jitter = _validate_nonnegative_float(jitter, "jitter")

    def update_identity(self, meas_noise, measurement, **kwargs):
        """Update with an identity measurement map."""
        return self.update_linear(
            measurement, eye(self.filter_state.dim), meas_noise, **kwargs
        )

    def update_linear(self, measurement, measurement_matrix, meas_noise, **kwargs):
        """Update with a linear Gaussian measurement model."""
        model = LinearGaussianMeasurementModel(measurement_matrix, meas_noise)
        return self.update_model(model, measurement, **kwargs)

    def update_nonlinear(
        self,
        measurement,
        measurement_function,
        meas_noise,
        jacobian,
        *,
        vectorized: bool = False,
        function_args: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Update with a nonlinear additive Gaussian measurement model."""
        model = AdditiveNoiseMeasurementModel(
            measurement_function,
            noise_covariance=meas_noise,
            jacobian=jacobian,
            vectorized=vectorized,
            function_args=function_args,
        )
        return self.update_model(model, measurement, **kwargs)

    def update_model(
        self,
        measurement_model,
        measurement,
        *,
        measurement_noise_covariance=None,
        n_steps: int | None = None,
        step_schedule=None,
        jitter: float | None = None,
        return_info: bool = False,
    ):
        """Update from a structural Gaussian measurement model with Jacobians."""
        result, info = gaussian_particle_flow_update(
            self.filter_state.d,
            measurement_model,
            measurement,
            flow_type=self.flow_type,
            measurement_noise_covariance=measurement_noise_covariance,
            weights=self.filter_state.w,
            n_steps=self.n_steps if n_steps is None else n_steps,
            step_schedule=(
                self.step_schedule if step_schedule is None else step_schedule
            ),
            jitter=self.jitter if jitter is None else jitter,
            return_info=True,
        )
        n_particles = self.filter_state.w.shape[0]
        self._filter_state = LinearDiracDistribution(
            result, ones(n_particles) / n_particles
        )
        return info if return_info else None


class LocalizedDaumHuangParticleFlowFilter(DaumHuangParticleFlowFilter):
    """Euclidean particle filter updated by localized Daum-Huang flow."""

    flow_type: FlowType = "ledh"


EDHParticleFlowFilter = DaumHuangParticleFlowFilter
LEDHParticleFlowFilter = LocalizedDaumHuangParticleFlowFilter


def _validate_flow_type(flow_type) -> FlowType:
    if flow_type not in {"edh", "ledh"}:
        raise ValueError("flow_type must be 'edh' or 'ledh'.")
    return flow_type


def _validate_positive_int(value, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _validate_nonnegative_float(value, name: str) -> float:
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


def _resolve_measurement_vector_np(model, measurement) -> np.ndarray:
    if measurement is None and hasattr(model, "y"):
        measurement = getattr(model, "y")
    if measurement is None:
        raise ValueError(
            "measurement must be supplied unless measurement_model exposes y."
        )
    return _as_vector_np(measurement, "measurement")


def _resolve_noise_covariance_np(model, override, measurement_dim: int) -> np.ndarray:
    value = override
    if value is None:
        for name in (
            "noise_covariance",
            "measurement_noise_cov",
            "measurement_noise_covariance",
            "meas_noise",
            "R",
        ):
            if hasattr(model, name):
                candidate = getattr(model, name)
                value = candidate() if callable(candidate) else candidate
                if value is not None:
                    break
    if value is None:
        raise ValueError(
            "measurement noise covariance must be supplied by argument or measurement_model."
        )
    covariance = _as_matrix_np(
        value, "measurement_noise_covariance", scalar_dim=measurement_dim
    )
    if covariance.shape != (measurement_dim, measurement_dim):
        raise ValueError("measurement noise covariance has incompatible shape.")
    return _regularize_cov_np(covariance, 0.0)


def _edh_step_np(X, weights, measurement_model, y, R, delta, jitter):
    mean, covariance = _weighted_mean_cov_np(X, weights, jitter)
    x_ref = mean[None, :]
    H = _measurement_jacobians_np(measurement_model, x_ref)[0]
    h_ref = _measurement_values_np(measurement_model, x_ref)[0]
    y_linear = y - h_ref + H @ mean
    return _gaussian_flow_affine_increment_np(
        X, mean, covariance, H, y_linear, R, delta, jitter=jitter
    )


def _ledh_step_np(X, weights, measurement_model, y, R, delta, jitter):
    mean, covariance = _weighted_mean_cov_np(X, weights, jitter)
    H_all = _measurement_jacobians_np(measurement_model, X)
    h_all = _measurement_values_np(measurement_model, X)
    if H_all.shape != (X.shape[0], y.size, X.shape[1]):
        raise ValueError("measurement jacobian returned incompatible shape.")
    if h_all.shape != (X.shape[0], y.size):
        raise ValueError("measurement function returned incompatible shape.")

    out = np.zeros_like(X)
    for index in range(X.shape[0]):
        H = H_all[index]
        y_linear = y - h_all[index] + H @ X[index]
        out[index] = _gaussian_flow_affine_increment_np(
            X[index : index + 1],
            mean,
            covariance,
            H,
            y_linear,
            R,
            delta,
            jitter=jitter,
        )[0]
    return out


def _gaussian_flow_affine_increment_np(
    X, mean, covariance, H, y, R, delta_lambda, *, jitter
):
    if delta_lambda < 0.0:
        raise ValueError("delta_lambda must be nonnegative.")
    if delta_lambda == 0.0:
        return X.copy()
    next_mean, next_covariance = _gaussian_bridge_moments_np(
        mean, covariance, H, y, R, delta_lambda, jitter=jitter
    )
    transport = _sym_sqrt_np(next_covariance) @ _sym_inv_sqrt_np(covariance)
    return next_mean[None, :] + (X - mean[None, :]) @ transport.T


def _gaussian_bridge_moments_np(mean, covariance, H, y, R, delta_lambda, *, jitter):
    if delta_lambda < 0.0:
        raise ValueError("delta_lambda must be nonnegative.")
    Rinv = np.linalg.inv(R)
    Pinv = np.linalg.inv(covariance)
    precision = Pinv + float(delta_lambda) * (H.T @ Rinv @ H)
    covariance_next = np.linalg.inv(_symmetrize_np(precision))
    information_vector = Pinv @ mean + float(delta_lambda) * (H.T @ Rinv @ y)
    mean_next = covariance_next @ information_vector
    return mean_next, _regularize_cov_np(covariance_next, jitter)


def _measurement_values_np(model, particles):
    X = _as_particle_matrix_np(particles)
    function = _measurement_function(model)
    batch_values = _try_batch_measurement(function, X)
    if batch_values is not None:
        return batch_values
    values = [
        _as_vector_np(function(asarray(particle)), "measurement value")
        for particle in X
    ]
    return np.vstack(values)


def _measurement_jacobians_np(model, particles):
    X = _as_particle_matrix_np(particles)
    matrix = _measurement_matrix(model)
    if matrix is not None:
        return np.repeat(matrix[None, :, :], X.shape[0], axis=0)
    jacobian = _jacobian_function(model)
    batch_jacobians = _try_batch_jacobian(jacobian, X)
    if batch_jacobians is not None:
        return batch_jacobians
    return np.stack(
        [
            _as_matrix_np(jacobian(asarray(particle)), "measurement jacobian")
            for particle in X
        ],
        axis=0,
    )


def _measurement_function(model):
    for name in ("h", "measurement_function", "evaluate", "predict_measurement"):
        if hasattr(model, name):
            function = getattr(model, name)
            if callable(function):
                return function
    matrix = _measurement_matrix(model)
    if matrix is not None:
        return lambda state: asarray(matrix @ _as_vector_np(state, "state"))
    raise TypeError(
        "measurement_model must expose h, measurement_function, evaluate, predict_measurement, or measurement_matrix."
    )


def _jacobian_function(model):
    if hasattr(model, "jacobian"):
        jacobian = getattr(model, "jacobian")
        if callable(jacobian):
            return jacobian
    raise TypeError(
        "measurement_model must expose a jacobian callable or measurement_matrix."
    )


def _measurement_matrix(model):
    for name in ("measurement_matrix", "matrix", "H"):
        if hasattr(model, name):
            matrix = getattr(model, name)
            matrix = matrix() if callable(matrix) else matrix
            return _as_matrix_np(matrix, "measurement_matrix")
    return None


def _try_batch_measurement(function, X):
    try:
        values = to_numpy(function(asarray(X)))
    except (TypeError, ValueError, NotImplementedError, IndexError):
        return None
    values = np.asarray(values, dtype=float)
    if values.ndim == 1 and X.shape[0] == 1:
        return values.reshape(1, -1)
    if values.ndim == 2 and values.shape[0] == X.shape[0]:
        return values
    return None


def _try_batch_jacobian(function, X):
    try:
        values = to_numpy(function(asarray(X)))
    except (TypeError, ValueError, NotImplementedError, IndexError):
        return None
    values = np.asarray(values, dtype=float)
    if values.ndim == 2 and X.shape[0] == 1:
        return values.reshape(1, values.shape[0], values.shape[1])
    if values.ndim == 3 and values.shape[0] == X.shape[0]:
        return values
    return None


def _validate_linear_bridge_np(mean, covariance, H, y, R, state_dim, jitter):
    m = _as_vector_np(mean, "mean")
    P = _as_matrix_np(covariance, "covariance")
    H = _as_matrix_np(H, "measurement_matrix")
    y = _as_vector_np(y, "measurement")
    R = _as_matrix_np(R, "measurement_noise_covariance", scalar_dim=y.size)
    if m.shape != (state_dim,):
        raise ValueError("mean has incompatible state dimension.")
    if P.shape != (state_dim, state_dim):
        raise ValueError("covariance has incompatible state dimension.")
    if H.shape[1] != state_dim:
        raise ValueError("measurement_matrix has incompatible state dimension.")
    if y.shape != (H.shape[0],):
        raise ValueError("measurement must have shape (measurement_dim,).")
    if R.shape != (H.shape[0], H.shape[0]):
        raise ValueError("measurement noise covariance has incompatible shape.")
    return m, _regularize_cov_np(P, jitter), H, y, _regularize_cov_np(R, 0.0)


def _record_flow_state(info, X, weights):
    mean, covariance = _weighted_mean_cov_np(X, weights, 0.0)
    info.mean_trace.append(asarray(mean))
    info.cov_trace.append(asarray(covariance))


def _weighted_mean_cov_np(particles, weights, jitter):
    X = _as_particle_matrix_np(particles)
    w = (
        np.full(X.shape[0], 1.0 / float(X.shape[0]))
        if weights is None
        else _as_weights_np(weights, X.shape[0])
    )
    mean = _weighted_mean_np(X, w)
    centered = X - mean[None, :]
    covariance = centered.T @ (centered * w[:, None])
    return mean, _regularize_cov_np(covariance, jitter)


def _weighted_mean_np(particles, weights):
    X = _as_particle_matrix_np(particles)
    w = (
        np.full(X.shape[0], 1.0 / float(X.shape[0]))
        if weights is None
        else _as_weights_np(weights, X.shape[0])
    )
    return w @ X


def _as_particle_matrix_np(value):
    X = np.asarray(to_numpy(value), dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2:
        raise ValueError("particles must have shape (n_particles, state_dim).")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("particles must contain at least one state vector.")
    if not np.all(np.isfinite(X)):
        raise ValueError("particles must be finite.")
    return X


def _as_weights_np(value, n_particles: int):
    weights = np.asarray(to_numpy(value), dtype=float).reshape(-1)
    if weights.shape != (n_particles,):
        raise ValueError("weights must have one entry per particle.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite.")
    if np.any(weights < 0.0):
        raise ValueError("weights must be nonnegative.")
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("weights must have positive finite total mass.")
    return weights / total


def _as_vector_np(value, name):
    vector = np.asarray(to_numpy(value), dtype=float)
    if vector.ndim == 0:
        vector = vector.reshape(1)
    vector = vector.reshape(-1) if vector.ndim == 1 else vector
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite.")
    return vector


def _as_matrix_np(value, name, *, scalar_dim: int | None = None):
    matrix = np.asarray(to_numpy(value), dtype=float)
    if matrix.ndim == 0 and scalar_dim == 1:
        matrix = matrix.reshape(1, 1)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite.")
    return matrix


def _lambda_deltas_np(n_steps, step_schedule):
    if step_schedule is None:
        n_steps = _validate_positive_int(n_steps, "n_steps")
        return np.full(n_steps, 1.0 / float(n_steps))
    deltas = np.asarray(step_schedule, dtype=float).reshape(-1)
    if deltas.size == 0:
        raise ValueError("step_schedule must not be empty.")
    if np.any(deltas <= 0.0):
        raise ValueError("step_schedule entries must be positive.")
    total = float(np.sum(deltas))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("step_schedule must have positive finite sum.")
    return deltas / total


def _regularize_cov_np(covariance, jitter):
    covariance = _symmetrize_np(np.asarray(covariance, dtype=float))
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square.")
    if jitter > 0.0:
        scale = max(float(np.trace(covariance) / max(covariance.shape[0], 1)), 1.0)
        covariance = covariance + float(jitter) * scale * np.eye(covariance.shape[0])
    sign = np.linalg.slogdet(covariance)[0]
    if sign <= 0.0:
        scale = max(float(np.trace(covariance) / max(covariance.shape[0], 1)), 1.0)
        covariance = covariance + max(float(jitter), 1e-10) * scale * np.eye(
            covariance.shape[0]
        )
    return _symmetrize_np(covariance)


def _symmetrize_np(matrix):
    return 0.5 * (matrix + matrix.T)


def _sym_sqrt_np(matrix):
    values, vectors = np.linalg.eigh(_symmetrize_np(matrix))
    values = np.maximum(values, 0.0)
    return (vectors * np.sqrt(values)) @ vectors.T


def _sym_inv_sqrt_np(matrix):
    values, vectors = np.linalg.eigh(_symmetrize_np(matrix))
    values = np.maximum(values, 1e-300)
    return (vectors * (1.0 / np.sqrt(values))) @ vectors.T


__all__ = [
    "DaumHuangParticleFlowFilter",
    "EDHParticleFlowFilter",
    "GaussianParticleFlowInfo",
    "LEDHParticleFlowFilter",
    "LocalizedDaumHuangParticleFlowFilter",
    "edh_particle_flow",
    "gaussian_bridge_moments",
    "gaussian_flow_affine_increment",
    "gaussian_particle_flow_drift",
    "gaussian_particle_flow_update",
    "ledh_particle_flow",
]
