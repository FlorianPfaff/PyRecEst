"""Robust sparse MAP smoothing for linear-Gaussian state sequences.

Prior and process factors are quadratic. Measurement factors may use a robust
loss on the norm of the whitened vector residual. The implementation supports
full-interval and timestamp-based fixed-lag solves.

MAP marginal covariances are not computed yet. Result objects therefore expose
``covariances=None`` rather than mislabelling filtered covariances as smoother
uncertainty.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr

RobustLinearGaussianMapLoss = Literal[
    "linear", "soft_l1", "huber", "cauchy", "arctan"
]
ROBUST_LINEAR_GAUSSIAN_MAP_LOSSES = (
    "linear",
    "soft_l1",
    "huber",
    "cauchy",
    "arctan",
)


@dataclass(frozen=True)
class LinearGaussianMeasurementFactor:
    """One factor ``measurement = H @ state[state_index] + offset + noise``."""

    state_index: int
    measurement: np.ndarray
    observation_matrix: np.ndarray
    covariance: np.ndarray
    offset: np.ndarray | None = None
    robust: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        index = _integer(self.state_index, "state_index", minimum=0)
        measurement = _vector(self.measurement, "measurement")
        observation = _matrix(self.observation_matrix, "observation_matrix")
        if observation.shape[0] != measurement.size:
            raise ValueError(
                "observation_matrix row count must match measurement dimension"
            )
        covariance = _covariance(self.covariance, measurement.size, "covariance")
        offset = (
            np.zeros(measurement.size)
            if self.offset is None
            else _vector(self.offset, "offset")
        )
        if offset.size != measurement.size:
            raise ValueError("offset must match measurement dimension")
        if not isinstance(self.robust, (bool, np.bool_)):
            raise ValueError("robust must be a Boolean scalar")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "state_index", index)
        object.__setattr__(self, "measurement", measurement.copy())
        object.__setattr__(self, "observation_matrix", observation.copy())
        object.__setattr__(self, "covariance", covariance.copy())
        object.__setattr__(self, "offset", offset.copy())
        object.__setattr__(self, "robust", bool(self.robust))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class RobustLinearGaussianMapConfig:
    """Controls for the sparse iteratively reweighted MAP solve."""

    loss: RobustLinearGaussianMapLoss = "huber"
    loss_scale: float = 3.0
    max_iterations: int = 50
    relative_tolerance: float = 1.0e-5
    covariance_jitter: float = 1.0e-9
    covariance_jitter_steps: int = 8
    line_search_steps: int = 12
    solver_max_iterations: int = 2000

    def __post_init__(self) -> None:
        if self.loss not in ROBUST_LINEAR_GAUSSIAN_MAP_LOSSES:
            raise ValueError(
                f"loss must be one of {ROBUST_LINEAR_GAUSSIAN_MAP_LOSSES}"
            )
        object.__setattr__(self, "loss_scale", _positive(self.loss_scale, "loss_scale"))
        object.__setattr__(
            self,
            "max_iterations",
            _integer(self.max_iterations, "max_iterations", minimum=1),
        )
        object.__setattr__(
            self,
            "relative_tolerance",
            _positive(self.relative_tolerance, "relative_tolerance"),
        )
        object.__setattr__(
            self,
            "covariance_jitter",
            _positive(self.covariance_jitter, "covariance_jitter"),
        )
        object.__setattr__(
            self,
            "covariance_jitter_steps",
            _integer(
                self.covariance_jitter_steps,
                "covariance_jitter_steps",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "line_search_steps",
            _integer(self.line_search_steps, "line_search_steps", minimum=0),
        )
        object.__setattr__(
            self,
            "solver_max_iterations",
            _integer(
                self.solver_max_iterations,
                "solver_max_iterations",
                minimum=1,
            ),
        )


@dataclass(frozen=True)
class RobustLinearGaussianMapResult:
    """Result of one full-interval MAP solve."""

    states: np.ndarray
    covariances: np.ndarray | None
    measurement_factor_count: int
    initial_cost: float
    final_cost: float
    iterations: int
    success: bool
    message: str
    measurement_sqrt_weights: np.ndarray


@dataclass(frozen=True)
class RobustLinearGaussianMapWindowSummary:
    """Diagnostics for one fixed-lag window."""

    start_index: int
    end_index: int
    measurement_factor_count: int
    initial_cost: float
    final_cost: float
    iterations: int
    success: bool
    message: str


@dataclass(frozen=True)
class FixedLagRobustLinearGaussianMapResult:
    """Fixed-lag states and one diagnostic summary per state."""

    states: np.ndarray
    covariances: np.ndarray | None
    lag: float
    windows: tuple[RobustLinearGaussianMapWindowSummary, ...]


@dataclass(frozen=True)
class _Factor:
    state_index: int
    observation: np.ndarray
    target: np.ndarray
    whitener: np.ndarray
    robust: bool


@dataclass(frozen=True)
class _Problem:
    initial_states: np.ndarray
    prior_mean: np.ndarray
    prior_whitener: np.ndarray
    transitions: np.ndarray
    offsets: np.ndarray
    process_whiteners: tuple[np.ndarray, ...]
    measurements: tuple[_Factor, ...]


__all__ = [
    "FixedLagRobustLinearGaussianMapResult",
    "LinearGaussianMeasurementFactor",
    "ROBUST_LINEAR_GAUSSIAN_MAP_LOSSES",
    "RobustLinearGaussianMapConfig",
    "RobustLinearGaussianMapLoss",
    "RobustLinearGaussianMapResult",
    "RobustLinearGaussianMapWindowSummary",
    "fixed_lag_robust_linear_gaussian_map_smooth",
    "robust_linear_gaussian_map_smooth",
]


def robust_linear_gaussian_map_smooth(
    initial_states: np.ndarray,
    *,
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    transition_matrices: Sequence[np.ndarray] | np.ndarray,
    process_covariances: Sequence[np.ndarray] | np.ndarray,
    measurements: Sequence[LinearGaussianMeasurementFactor] = (),
    transition_offsets: Sequence[np.ndarray] | np.ndarray | None = None,
    config: RobustLinearGaussianMapConfig | None = None,
) -> RobustLinearGaussianMapResult:
    """Solve a robust linear-Gaussian trajectory MAP problem.

    The transition model is ``x[k+1] = F[k] @ x[k] + offset[k] + noise``.
    Robust measurement losses act on the norm of each whitened vector residual,
    making a factor's robustification invariant to rotations of its coordinates.
    """

    cfg = _config(config)
    states = _states(initial_states)
    count, dimension = states.shape
    prior = _vector(prior_mean, "prior_mean")
    if prior.size != dimension:
        raise ValueError("prior_mean must match state dimension")
    transitions = _matrix_sequence(
        transition_matrices, count - 1, dimension, "transition_matrices"
    )
    process = _covariance_sequence(
        process_covariances, count - 1, dimension, "process_covariances"
    )
    offsets = _offsets(transition_offsets, count - 1, dimension)
    factors = _factors(measurements, count, dimension)
    problem = _Problem(
        initial_states=states,
        prior_mean=prior,
        prior_whitener=_whitener(
            _covariance(prior_covariance, dimension, "prior_covariance"), cfg
        ),
        transitions=transitions,
        offsets=offsets,
        process_whiteners=tuple(_whitener(item, cfg) for item in process),
        measurements=tuple(
            _Factor(
                state_index=item.state_index,
                observation=item.observation_matrix,
                target=item.measurement - item.offset,
                whitener=_whitener(item.covariance, cfg),
                robust=item.robust,
            )
            for item in factors
        ),
    )
    return _solve(problem, cfg)


def fixed_lag_robust_linear_gaussian_map_smooth(
    times: np.ndarray,
    initial_states: np.ndarray,
    *,
    anchor_covariances: Sequence[np.ndarray] | np.ndarray,
    transition_matrices: Sequence[np.ndarray] | np.ndarray,
    process_covariances: Sequence[np.ndarray] | np.ndarray,
    measurements: Sequence[LinearGaussianMeasurementFactor] = (),
    lag: float,
    transition_offsets: Sequence[np.ndarray] | np.ndarray | None = None,
    config: RobustLinearGaussianMapConfig | None = None,
) -> FixedLagRobustLinearGaussianMapResult:
    """Return the first-state solution of every timestamp-bounded MAP window."""

    cfg = _config(config)
    states = _states(initial_states)
    count, dimension = states.shape
    time_values = _vector(times, "times")
    if time_values.size != count:
        raise ValueError("times must contain one value per state")
    if np.any(np.diff(time_values) < 0.0):
        raise ValueError("times must be sorted by nondecreasing value")
    lag_value = _nonnegative(lag, "lag")
    anchors = _covariance_sequence(
        anchor_covariances, count, dimension, "anchor_covariances"
    )
    transitions = _matrix_sequence(
        transition_matrices, count - 1, dimension, "transition_matrices"
    )
    process = _covariance_sequence(
        process_covariances, count - 1, dimension, "process_covariances"
    )
    offsets = _offsets(transition_offsets, count - 1, dimension)
    factors = _factors(measurements, count, dimension)

    output = states.copy()
    windows: list[RobustLinearGaussianMapWindowSummary] = []
    for start, time_value in enumerate(time_values):
        end = int(np.searchsorted(time_values, time_value + lag_value, side="right") - 1)
        if end <= start:
            windows.append(
                RobustLinearGaussianMapWindowSummary(
                    start,
                    start,
                    0,
                    0.0,
                    0.0,
                    0,
                    True,
                    "window contains no future state",
                )
            )
            continue
        local = tuple(
            LinearGaussianMeasurementFactor(
                state_index=item.state_index - start,
                measurement=item.measurement,
                observation_matrix=item.observation_matrix,
                covariance=item.covariance,
                offset=item.offset,
                robust=item.robust,
                metadata=item.metadata,
            )
            for item in factors
            if start <= item.state_index <= end
        )
        result = robust_linear_gaussian_map_smooth(
            states[start : end + 1],
            prior_mean=states[start],
            prior_covariance=anchors[start],
            transition_matrices=transitions[start:end],
            process_covariances=process[start:end],
            measurements=local,
            transition_offsets=offsets[start:end],
            config=cfg,
        )
        output[start] = result.states[0]
        windows.append(
            RobustLinearGaussianMapWindowSummary(
                start,
                end,
                result.measurement_factor_count,
                result.initial_cost,
                result.final_cost,
                result.iterations,
                result.success,
                result.message,
            )
        )
    return FixedLagRobustLinearGaussianMapResult(
        states=output,
        covariances=None,
        lag=lag_value,
        windows=tuple(windows),
    )


def _solve(problem: _Problem, cfg: RobustLinearGaussianMapConfig) -> RobustLinearGaussianMapResult:
    states = problem.initial_states.copy()
    initial_cost = current_cost = _cost(states, problem, cfg)
    success = False
    message = "maximum iterations reached"
    iterations = 0
    quadratic = cfg.loss == "linear" or not any(
        item.robust for item in problem.measurements
    )
    for iterations in range(1, cfg.max_iterations + 1):
        matrix, rhs = _system(states, problem, cfg)
        proposal = lsmr(
            matrix,
            rhs,
            atol=1.0e-10,
            btol=1.0e-10,
            maxiter=cfg.solver_max_iterations,
        )[0].reshape(states.shape)
        candidate, candidate_cost, accepted = _descent(
            states, proposal, current_cost, problem, cfg
        )
        if not accepted:
            message = "line search stalled"
            break
        delta = _norm(candidate - states)
        reference = max(1.0, _norm(states))
        states = candidate
        current_cost = candidate_cost
        if quadratic:
            success = True
            message = "solved"
            break
        if delta <= cfg.relative_tolerance * reference:
            success = True
            message = "converged"
            break
    return RobustLinearGaussianMapResult(
        states=states,
        covariances=None,
        measurement_factor_count=len(problem.measurements),
        initial_cost=float(initial_cost),
        final_cost=float(current_cost),
        iterations=iterations,
        success=success,
        message=message,
        measurement_sqrt_weights=_weights(states, problem, cfg),
    )


def _descent(
    current: np.ndarray,
    proposal: np.ndarray,
    current_cost: float,
    problem: _Problem,
    cfg: RobustLinearGaussianMapConfig,
) -> tuple[np.ndarray, float, bool]:
    proposal_cost = _cost(proposal, problem, cfg)
    if np.isfinite(proposal_cost) and proposal_cost <= current_cost:
        return proposal, proposal_cost, True
    direction = proposal - current
    step = 0.5
    for _ in range(cfg.line_search_steps):
        candidate = current + step * direction
        candidate_cost = _cost(candidate, problem, cfg)
        if np.isfinite(candidate_cost) and candidate_cost <= current_cost:
            return candidate, candidate_cost, True
        step *= 0.5
    return current.copy(), float(current_cost), False


def _system(
    states: np.ndarray,
    problem: _Problem,
    cfg: RobustLinearGaussianMapConfig,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs: list[float] = []
    row = 0
    dimension = states.shape[1]
    row = _append(
        rows,
        cols,
        data,
        rhs,
        row,
        ((0, np.eye(dimension)),),
        problem.prior_mean,
        problem.prior_whitener,
        1.0,
        dimension,
    )
    for index, (transition, offset, whitener) in enumerate(
        zip(
            problem.transitions,
            problem.offsets,
            problem.process_whiteners,
            strict=True,
        )
    ):
        row = _append(
            rows,
            cols,
            data,
            rhs,
            row,
            ((index + 1, np.eye(dimension)), (index, -transition)),
            offset,
            whitener,
            1.0,
            dimension,
        )
    for factor, weight in zip(
        problem.measurements, _weights(states, problem, cfg), strict=True
    ):
        row = _append(
            rows,
            cols,
            data,
            rhs,
            row,
            ((factor.state_index, factor.observation),),
            factor.target,
            factor.whitener,
            float(weight),
            dimension,
        )
    matrix = sparse.coo_matrix(
        (data, (rows, cols)), shape=(row, states.size)
    ).tocsr()
    return matrix, np.asarray(rhs)


def _append(
    rows: list[int],
    cols: list[int],
    data: list[float],
    rhs: list[float],
    row: int,
    blocks: Sequence[tuple[int, np.ndarray]],
    target: np.ndarray,
    whitener: np.ndarray,
    weight: float,
    dimension: int,
) -> int:
    white = float(weight) * whitener
    transformed = tuple((index, white @ block) for index, block in blocks)
    target_white = white @ target
    for local_row in range(white.shape[0]):
        for index, block in transformed:
            for coordinate, value in enumerate(block[local_row]):
                if value != 0.0:
                    rows.append(row + local_row)
                    cols.append(dimension * index + coordinate)
                    data.append(float(value))
        rhs.append(float(target_white[local_row]))
    return row + white.shape[0]


def _cost(states: np.ndarray, problem: _Problem, cfg: RobustLinearGaussianMapConfig) -> float:
    residual = problem.prior_whitener @ (states[0] - problem.prior_mean)
    cost = 0.5 * _norm(residual) ** 2
    for index, (transition, offset, whitener) in enumerate(
        zip(
            problem.transitions,
            problem.offsets,
            problem.process_whiteners,
            strict=True,
        )
    ):
        residual = whitener @ (
            states[index + 1] - transition @ states[index] - offset
        )
        cost += 0.5 * _norm(residual) ** 2
    for factor in problem.measurements:
        residual = factor.whitener @ (
            factor.target - factor.observation @ states[factor.state_index]
        )
        magnitude = _norm(residual)
        cost += _rho(magnitude, cfg) if factor.robust else 0.5 * magnitude**2
    return float(cost)


def _weights(
    states: np.ndarray, problem: _Problem, cfg: RobustLinearGaussianMapConfig
) -> np.ndarray:
    values = np.ones(len(problem.measurements))
    for index, factor in enumerate(problem.measurements):
        if factor.robust:
            residual = factor.whitener @ (
                factor.target - factor.observation @ states[factor.state_index]
            )
            values[index] = _sqrt_weight(_norm(residual), cfg)
    return values


def _sqrt_weight(value: float, cfg: RobustLinearGaussianMapConfig) -> float:
    if cfg.loss == "linear" or value == 0.0:
        return 1.0
    scaled = value / cfg.loss_scale
    if cfg.loss == "huber":
        return float(np.sqrt(min(1.0, 1.0 / scaled)))
    if cfg.loss == "soft_l1":
        return float((1.0 + scaled**2) ** -0.25)
    if cfg.loss == "cauchy":
        return float((1.0 + scaled**2) ** -0.5)
    if cfg.loss == "arctan":
        return float((1.0 + scaled**4) ** -0.5)
    raise ValueError(f"unknown robust loss {cfg.loss!r}")


def _rho(value: float, cfg: RobustLinearGaussianMapConfig) -> float:
    if cfg.loss == "linear":
        return 0.5 * value**2
    scale = cfg.loss_scale
    scaled_sq = (value / scale) ** 2
    if cfg.loss == "huber":
        return (
            0.5 * value**2
            if value <= scale
            else scale * (value - 0.5 * scale)
        )
    if cfg.loss == "soft_l1":
        return scale**2 * (np.sqrt(1.0 + scaled_sq) - 1.0)
    if cfg.loss == "cauchy":
        return 0.5 * scale**2 * np.log1p(scaled_sq)
    if cfg.loss == "arctan":
        return 0.5 * scale**2 * np.arctan(scaled_sq)
    raise ValueError(f"unknown robust loss {cfg.loss!r}")


def _config(value: RobustLinearGaussianMapConfig | None) -> RobustLinearGaussianMapConfig:
    if value is None:
        return RobustLinearGaussianMapConfig()
    if not isinstance(value, RobustLinearGaussianMapConfig):
        raise TypeError("config must be a RobustLinearGaussianMapConfig or None")
    return value


def _factors(
    values: Sequence[LinearGaussianMeasurementFactor], count: int, dimension: int
) -> tuple[LinearGaussianMeasurementFactor, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("measurements must be a sequence of measurement factors")
    factors = tuple(values)
    for item in factors:
        if not isinstance(item, LinearGaussianMeasurementFactor):
            raise TypeError(
                "measurements must contain LinearGaussianMeasurementFactor instances"
            )
        if item.state_index >= count:
            raise ValueError("measurement state_index is outside the state sequence")
        if item.observation_matrix.shape[1] != dimension:
            raise ValueError(
                "observation_matrix column count must match state dimension"
            )
    return factors


def _states(value: Any) -> np.ndarray:
    result = _array(value, "initial_states")
    if result.ndim != 2 or min(result.shape) < 1:
        raise ValueError(
            "initial_states must have shape (state_count, state_dim) with positive dimensions"
        )
    return result.copy()


def _offsets(value: Any, count: int, dimension: int) -> np.ndarray:
    if value is None:
        return np.zeros((count, dimension))
    result = _array(value, "transition_offsets")
    expected = (count, dimension)
    if count == 0 and result.size == 0:
        return np.empty(expected)
    if result.shape != expected:
        raise ValueError(f"transition_offsets must have shape {expected}")
    return result.copy()


def _matrix_sequence(value: Any, count: int, dimension: int, name: str) -> np.ndarray:
    result = _array(value, name)
    expected = (count, dimension, dimension)
    if count == 0 and result.size == 0:
        return np.empty(expected)
    if result.shape != expected:
        raise ValueError(f"{name} must have shape {expected}")
    return result.copy()


def _covariance_sequence(
    value: Any, count: int, dimension: int, name: str
) -> np.ndarray:
    result = _matrix_sequence(value, count, dimension, name)
    return np.stack(
        [_covariance(item, dimension, f"{name}[{index}]") for index, item in enumerate(result)]
    ) if count else result


def _covariance(value: Any, dimension: int, name: str) -> np.ndarray:
    result = _matrix(value, name)
    if result.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape ({dimension}, {dimension})")
    result = 0.5 * (result + result.T)
    try:
        eigenvalues = np.linalg.eigvalsh(result)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive semidefinite") from exc
    scale = max(1.0, float(np.max(np.abs(result), initial=0.0)))
    if float(np.min(eigenvalues, initial=0.0)) < -1.0e-10 * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    return result


def _whitener(covariance: np.ndarray, cfg: RobustLinearGaussianMapConfig) -> np.ndarray:
    identity = np.eye(covariance.shape[0])
    jitter = cfg.covariance_jitter
    for _ in range(cfg.covariance_jitter_steps):
        try:
            return np.linalg.solve(
                np.linalg.cholesky(covariance + jitter * identity), identity
            )
        except np.linalg.LinAlgError:
            jitter *= 10.0
    values, vectors = np.linalg.eigh(covariance)
    return np.diag(1.0 / np.sqrt(np.maximum(values, jitter))) @ vectors.T


def _array(value: Any, name: str) -> np.ndarray:
    message = f"{name} must contain finite real numeric values"
    if np.ma.is_masked(value):
        raise ValueError(message)
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(message) from exc
    if raw.dtype.kind in "bUSMm" or np.iscomplexobj(raw):
        raise ValueError(message)
    if raw.dtype.kind == "O":
        invalid = (
            bool,
            np.bool_,
            complex,
            np.complexfloating,
            str,
            bytes,
            bytearray,
            np.datetime64,
            np.timedelta64,
        )
        if any(np.ma.is_masked(item) or isinstance(item, invalid) for item in raw.flat):
            raise ValueError(message)
    try:
        result = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(result).all():
        raise ValueError(message)
    return result


def _vector(value: Any, name: str) -> np.ndarray:
    result = _array(value, name)
    if result.ndim == 0:
        return result.reshape(1)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    return result


def _matrix(value: Any, name: str) -> np.ndarray:
    result = _array(value, name)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    return result


def _scalar(value: Any, name: str) -> float:
    message = f"{name} must be a finite real scalar"
    if np.ma.is_masked(value) or isinstance(value, (bool, np.bool_, complex, np.complexfloating)):
        raise ValueError(message)
    try:
        result = np.asarray(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(message) from exc
    if result.shape != () or result.dtype.kind in "bUSMm" or np.iscomplexobj(result):
        raise ValueError(message)
    try:
        number = float(result.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(number):
        raise ValueError(message)
    return number


def _positive(value: Any, name: str) -> float:
    result = _scalar(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative(value: Any, name: str) -> float:
    result = _scalar(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _integer(value: Any, name: str, *, minimum: int) -> int:
    message = f"{name} must be an exact integer scalar"
    if np.ma.is_masked(value) or isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        result = np.asarray(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(message) from exc
    if result.shape != () or result.dtype.kind in "bUSMm" or np.iscomplexobj(result):
        raise ValueError(message)
    try:
        number = int(operator.index(result.item()))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if number < minimum:
        qualifier = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return number


def _norm(value: np.ndarray) -> float:
    return float(np.hypot.reduce(np.abs(np.asarray(value)).reshape(-1), initial=0.0))
