"""Ready-made dynamic motion models and discretization helpers.

The functions in this module intentionally return the existing PyRecEst model
objects where possible. Linear Gaussian kinematic models return
:class:`LinearGaussianTransitionModel`; nonlinear coordinated-turn and pose
models return :class:`AdditiveNoiseTransitionModel`.
"""

from __future__ import annotations

from math import factorial
from typing import Any

import numpy as np

# pylint: disable=no-name-in-module,no-member,too-many-arguments,too-many-positional-arguments
from pyrecest.backend import abs as _abs
from pyrecest.backend import asarray, cos, sin, stack, where, zeros
from pyrecest.models.additive_noise import AdditiveNoiseTransitionModel
from pyrecest.models.linear_gaussian import LinearGaussianTransitionModel
from scipy.linalg import expm

__all__ = [
    "constant_acceleration_model",
    "constant_acceleration_transition_matrix",
    "constant_jerk_model",
    "constant_jerk_transition_matrix",
    "constant_velocity_model",
    "constant_velocity_transition_matrix",
    "continuous_to_discrete_lti",
    "coordinated_turn_model",
    "coordinated_turn_transition",
    "integrated_white_noise_covariance",
    "kinematic_transition_matrix",
    "nearly_constant_speed_model",
    "nearly_constant_speed_transition",
    "nearly_coordinated_turn_model",
    "se2_unicycle_model",
    "se2_unicycle_transition",
    "se3_pose_twist_model",
    "se3_pose_twist_transition",
    "singer_model",
    "singer_process_noise_covariance",
    "singer_transition_matrix",
    "white_noise_acceleration_covariance",
    "white_noise_jerk_covariance",
    "white_noise_snap_covariance",
]

_TEXT_OR_BOOL_SCALAR_TYPES = (
    bool,
    np.bool_,
    str,
    bytes,
    bytearray,
    np.str_,
    np.bytes_,
)
_REJECTED_NUMERIC_ARRAY_KINDS = frozenset({"b", "c", "S", "U", "M", "m"})


def _is_text_bool_or_complex(value: Any) -> bool:
    return isinstance(value, _TEXT_OR_BOOL_SCALAR_TYPES) or isinstance(
        value, (complex, np.complexfloating)
    )


def _as_scalar_float(value: Any, name: str) -> float:
    value_array = np.asarray(value)
    if (
        value_array.shape != ()
        or value_array.dtype.kind in _REJECTED_NUMERIC_ARRAY_KINDS
    ):
        raise ValueError(f"{name} must be a scalar number")
    scalar_value = value_array.item()
    if _is_text_bool_or_complex(scalar_value):
        raise ValueError(f"{name} must be a scalar number")
    try:
        scalar = float(scalar_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a scalar number") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _as_nonnegative_float(value: Any, name: str) -> float:
    scalar = _as_scalar_float(value, name)
    if scalar < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return scalar


def _as_positive_float(value: Any, name: str) -> float:
    scalar = _as_scalar_float(value, name)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _as_integer(value: Any, name: str) -> int:
    scalar = _as_scalar_float(value, name)
    if not scalar.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(scalar)


def _as_positive_integer(value: Any, name: str) -> int:
    integer = _as_integer(value, name)
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


def _as_nonnegative_integer(value: Any, name: str) -> int:
    integer = _as_integer(value, name)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative")
    return integer


def _as_nonnegative_vector(value: Any, length: int, name: str) -> np.ndarray:
    raw_value_array = np.asarray(value)
    if raw_value_array.dtype.kind in _REJECTED_NUMERIC_ARRAY_KINDS:
        raise ValueError(f"{name} must be numeric")
    if raw_value_array.dtype == object and any(
        _is_text_bool_or_complex(item) for item in raw_value_array.flat
    ):
        raise ValueError(f"{name} must be numeric")
    try:
        value_array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if value_array.shape == ():
        return np.repeat(_as_nonnegative_float(value_array, name), length)
    if value_array.shape != (length,):
        raise ValueError(f"{name} must be scalar or have shape (spatial_dim,)")
    if not np.all(np.isfinite(value_array)) or np.any(value_array < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return value_array


def kinematic_transition_matrix(
    dt: float, spatial_dim: int = 2, derivative_order: int = 1
):
    """Return a block kinematic transition matrix.

    ``derivative_order=1`` yields constant velocity states ``[p, v]``;
    ``derivative_order=2`` yields constant acceleration states ``[p, v, a]``;
    ``derivative_order=3`` yields constant jerk states ``[p, v, a, j]``. For
    ``spatial_dim > 1`` the state is grouped by derivative, e.g.
    ``[x, y, vx, vy]`` for 2D constant velocity and
    ``[x, y, vx, vy, ax, ay]`` for 2D constant acceleration.
    """
    dt = _as_scalar_float(dt, "dt")
    spatial_dim = _as_positive_integer(spatial_dim, "spatial_dim")
    derivative_order = _as_nonnegative_integer(derivative_order, "derivative_order")
    block_size = derivative_order + 1
    size = spatial_dim * block_size
    matrix = np.eye(size, dtype=float)
    for derivative_row in range(block_size):
        for derivative_col in range(derivative_row + 1, block_size):
            power = derivative_col - derivative_row
            coefficient = dt**power / float(factorial(power))
            for axis in range(spatial_dim):
                matrix[
                    _state_index(derivative_row, axis, spatial_dim),
                    _state_index(derivative_col, axis, spatial_dim),
                ] = coefficient
    return asarray(matrix)


def _state_index(derivative_index: int, axis: int, spatial_dim: int) -> int:
    return int(derivative_index) * int(spatial_dim) + int(axis)


def constant_velocity_transition_matrix(dt: float, spatial_dim: int = 2):
    """Return a constant-velocity transition matrix for ``[p, v]`` states."""
    return kinematic_transition_matrix(dt, spatial_dim=spatial_dim, derivative_order=1)


def constant_acceleration_transition_matrix(dt: float, spatial_dim: int = 2):
    """Return a constant-acceleration transition matrix for ``[p, v, a]`` states."""
    return kinematic_transition_matrix(dt, spatial_dim=spatial_dim, derivative_order=2)


def constant_jerk_transition_matrix(dt: float, spatial_dim: int = 2):
    """Return a constant-jerk transition matrix for ``[p, v, a, j]`` states."""
    return kinematic_transition_matrix(dt, spatial_dim=spatial_dim, derivative_order=3)


def integrated_white_noise_covariance(
    dt: float,
    spatial_dim: int = 2,
    derivative_order: int = 1,
    spectral_density: float | np.ndarray = 1.0,
):
    """Return process noise for an integrated white-noise model.

    The state contains derivatives through ``derivative_order`` and the white
    noise acts on the next derivative. For example, ``derivative_order=1`` is
    the white-noise-acceleration covariance for a constant-velocity state, and
    ``derivative_order=2`` is the white-noise-jerk covariance for a
    constant-acceleration state.
    """
    dt = _as_nonnegative_float(dt, "dt")
    spatial_dim = _as_positive_integer(spatial_dim, "spatial_dim")
    derivative_order = _as_nonnegative_integer(derivative_order, "derivative_order")
    block_size = derivative_order + 1
    densities = _as_nonnegative_vector(
        spectral_density,
        spatial_dim,
        "spectral_density",
    )

    size = spatial_dim * block_size
    covariance = np.zeros((size, size), dtype=float)
    for derivative_row in range(block_size):
        for derivative_col in range(block_size):
            exponent = 2 * block_size - 1 - derivative_row - derivative_col
            denominator = (
                factorial(block_size - 1 - derivative_row)
                * factorial(block_size - 1 - derivative_col)
                * exponent
            )
            coefficient = dt**exponent / float(denominator)
            for axis, density in enumerate(densities):
                covariance[
                    _state_index(derivative_row, axis, spatial_dim),
                    _state_index(derivative_col, axis, spatial_dim),
                ] = (
                    float(density) * coefficient
                )
    return asarray(covariance)


def white_noise_acceleration_covariance(
    dt: float, spatial_dim: int = 2, spectral_density: float | np.ndarray = 1.0
):
    """Return white-noise-acceleration covariance for constant-velocity states."""
    return integrated_white_noise_covariance(
        dt,
        spatial_dim=spatial_dim,
        derivative_order=1,
        spectral_density=spectral_density,
    )


def white_noise_jerk_covariance(
    dt: float, spatial_dim: int = 2, spectral_density: float | np.ndarray = 1.0
):
    """Return white-noise-jerk covariance for constant-acceleration states."""
    return integrated_white_noise_covariance(
        dt,
        spatial_dim=spatial_dim,
        derivative_order=2,
        spectral_density=spectral_density,
    )


def white_noise_snap_covariance(
    dt: float, spatial_dim: int = 2, spectral_density: float | np.ndarray = 1.0
):
    """Return white-noise-snap covariance for constant-jerk states."""
    return integrated_white_noise_covariance(
        dt,
        spatial_dim=spatial_dim,
        derivative_order=3,
        spectral_density=spectral_density,
    )


def constant_velocity_model(
    dt: float, spatial_dim: int = 2, spectral_density: float | np.ndarray = 1.0
) -> LinearGaussianTransitionModel:
    """Return a linear Gaussian constant-velocity transition model."""
    return LinearGaussianTransitionModel(
        constant_velocity_transition_matrix(dt, spatial_dim=spatial_dim),
        white_noise_acceleration_covariance(
            dt, spatial_dim=spatial_dim, spectral_density=spectral_density
        ),
    )


def constant_acceleration_model(
    dt: float, spatial_dim: int = 2, spectral_density: float | np.ndarray = 1.0
) -> LinearGaussianTransitionModel:
    """Return a linear Gaussian constant-acceleration transition model."""
    return LinearGaussianTransitionModel(
        constant_acceleration_transition_matrix(dt, spatial_dim=spatial_dim),
        white_noise_jerk_covariance(
            dt, spatial_dim=spatial_dim, spectral_density=spectral_density
        ),
    )


def constant_jerk_model(
    dt: float, spatial_dim: int = 2, spectral_density: float | np.ndarray = 1.0
) -> LinearGaussianTransitionModel:
    """Return a linear Gaussian constant-jerk transition model."""
    return LinearGaussianTransitionModel(
        constant_jerk_transition_matrix(dt, spatial_dim=spatial_dim),
        white_noise_snap_covariance(
            dt, spatial_dim=spatial_dim, spectral_density=spectral_density
        ),
    )


def continuous_to_discrete_lti(
    continuous_matrix: Any,
    noise_input_matrix: Any | None = None,
    continuous_noise_covariance: Any | None = None,
    dt: float = 1.0,
):
    """Discretize a continuous-time LTI model and process noise.

    For ``dx/dt = A x + L w`` with continuous white-noise covariance ``Qc``, the
    function returns ``(F, Q)``. If no noise matrices are supplied, ``Q`` is a
    zero matrix.
    """
    continuous_matrix_np = np.asarray(continuous_matrix, dtype=float)
    if (
        continuous_matrix_np.ndim != 2
        or continuous_matrix_np.shape[0] != continuous_matrix_np.shape[1]
    ):
        raise ValueError("continuous_matrix must be square")
    if not np.all(np.isfinite(continuous_matrix_np)):
        raise ValueError("continuous_matrix must contain only finite values")
    dim = continuous_matrix_np.shape[0]
    dt = _as_scalar_float(dt, "dt")
    transition = expm(continuous_matrix_np * dt)

    if noise_input_matrix is None and continuous_noise_covariance is None:
        return asarray(transition), zeros((dim, dim))
    if noise_input_matrix is None or continuous_noise_covariance is None:
        raise ValueError(
            "noise_input_matrix and continuous_noise_covariance must be supplied together"
        )

    noise_input_np = np.asarray(noise_input_matrix, dtype=float)
    continuous_noise_np = np.asarray(continuous_noise_covariance, dtype=float)
    if noise_input_np.ndim != 2 or noise_input_np.shape[0] != dim:
        raise ValueError("noise_input_matrix has incompatible shape")
    if not np.all(np.isfinite(noise_input_np)):
        raise ValueError("noise_input_matrix must contain only finite values")
    if (
        continuous_noise_np.ndim != 2
        or continuous_noise_np.shape[0] != continuous_noise_np.shape[1]
        or continuous_noise_np.shape[0] != noise_input_np.shape[1]
    ):
        raise ValueError("continuous_noise_covariance has incompatible shape")
    if not np.all(np.isfinite(continuous_noise_np)):
        raise ValueError("continuous_noise_covariance must contain only finite values")

    spectral = noise_input_np @ continuous_noise_np @ noise_input_np.T
    van_loan = np.zeros((2 * dim, 2 * dim), dtype=float)
    van_loan[:dim, :dim] = -continuous_matrix_np
    van_loan[:dim, dim:] = spectral
    van_loan[dim:, dim:] = continuous_matrix_np.T
    van_loan_exp = expm(van_loan * dt)
    transition_from_van_loan = van_loan_exp[dim:, dim:].T
    process_noise = transition_from_van_loan @ van_loan_exp[:dim, dim:]
    return asarray(transition), asarray(0.5 * (process_noise + process_noise.T))


def singer_transition_matrix(dt: float, spatial_dim: int = 2, tau: float = 20.0):
    """Return a Singer acceleration transition matrix for ``[p, v, a]`` states."""
    dt = _as_scalar_float(dt, "dt")
    spatial_dim = _as_positive_integer(spatial_dim, "spatial_dim")
    tau = _as_positive_float(tau, "tau")
    alpha = 1.0 / tau
    decay = np.exp(-alpha * dt)
    block = np.array(
        [
            [1.0, dt, dt / alpha - (1.0 - decay) / alpha**2],
            [0.0, 1.0, (1.0 - decay) / alpha],
            [0.0, 0.0, decay],
        ],
        dtype=float,
    )
    matrix = np.zeros((3 * spatial_dim, 3 * spatial_dim), dtype=float)
    for row_derivative in range(3):
        for col_derivative in range(3):
            for axis in range(spatial_dim):
                matrix[
                    _state_index(row_derivative, axis, spatial_dim),
                    _state_index(col_derivative, axis, spatial_dim),
                ] = block[row_derivative, col_derivative]
    return asarray(matrix)


def singer_process_noise_covariance(
    dt: float,
    spatial_dim: int = 2,
    tau: float = 20.0,
    acceleration_variance: float | np.ndarray = 1.0,
):
    """Return Singer process noise covariance via Van Loan discretization."""
    dt = _as_nonnegative_float(dt, "dt")
    spatial_dim = _as_positive_integer(spatial_dim, "spatial_dim")
    tau = _as_positive_float(tau, "tau")
    alpha = 1.0 / tau
    variances = _as_nonnegative_vector(
        acceleration_variance,
        spatial_dim,
        "acceleration_variance",
    )

    continuous_block = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -alpha]],
        dtype=float,
    )
    noise_input = np.array([[0.0], [0.0], [1.0]], dtype=float)
    covariance = np.zeros((3 * spatial_dim, 3 * spatial_dim), dtype=float)
    for axis, variance in enumerate(variances):
        _, q_block = continuous_to_discrete_lti(
            continuous_block,
            noise_input,
            np.array([[2.0 * alpha * float(variance)]], dtype=float),
            dt=dt,
        )
        q_block_np = np.asarray(q_block, dtype=float)
        for row_derivative in range(3):
            for col_derivative in range(3):
                covariance[
                    _state_index(row_derivative, axis, spatial_dim),
                    _state_index(col_derivative, axis, spatial_dim),
                ] = q_block_np[row_derivative, col_derivative]
    return asarray(covariance)


def singer_model(
    dt: float,
    spatial_dim: int = 2,
    tau: float = 20.0,
    acceleration_variance: float | np.ndarray = 1.0,
) -> LinearGaussianTransitionModel:
    """Return a linear Gaussian Singer acceleration transition model."""
    return LinearGaussianTransitionModel(
        singer_transition_matrix(dt, spatial_dim=spatial_dim, tau=tau),
        singer_process_noise_covariance(
            dt,
            spatial_dim=spatial_dim,
            tau=tau,
            acceleration_variance=acceleration_variance,
        ),
    )


def coordinated_turn_transition(state, dt: float = 1.0, turn_threshold: float = 1e-8):
    """Propagate a 2D coordinated-turn state ``[x, y, vx, vy, omega]``."""
    state = asarray(state)
    x_pos, y_pos, x_vel, y_vel, omega = state[0], state[1], state[2], state[3], state[4]
    omega_dt = omega * float(dt)
    sin_omega_dt = sin(omega_dt)
    cos_omega_dt = cos(omega_dt)
    safe_omega = where(_abs(omega) < float(turn_threshold), 1.0, omega)

    turn_x = (
        x_pos
        + sin_omega_dt / safe_omega * x_vel
        - (1.0 - cos_omega_dt) / safe_omega * y_vel
    )
    turn_y = (
        y_pos
        + (1.0 - cos_omega_dt) / safe_omega * x_vel
        + sin_omega_dt / safe_omega * y_vel
    )
    turn_vx = cos_omega_dt * x_vel - sin_omega_dt * y_vel
    turn_vy = sin_omega_dt * x_vel + cos_omega_dt * y_vel

    linear_x = x_pos + float(dt) * x_vel
    linear_y = y_pos + float(dt) * y_vel
    is_nearly_straight = _abs(omega) < float(turn_threshold)
    return stack(
        [
            where(is_nearly_straight, linear_x, turn_x),
            where(is_nearly_straight, linear_y, turn_y),
            where(is_nearly_straight, x_vel, turn_vx),
            where(is_nearly_straight, y_vel, turn_vy),
            omega,
        ]
    )


def coordinated_turn_model(
    dt: float = 1.0, noise_covariance: Any | None = None
) -> AdditiveNoiseTransitionModel:
    """Return an additive-noise coordinated-turn model for ``[x, y, vx, vy, omega]``."""
    if noise_covariance is None:
        noise_covariance = zeros((5, 5))
    return AdditiveNoiseTransitionModel(
        transition_function=coordinated_turn_transition,
        noise_covariance=noise_covariance,
        dt=dt,
    )


def nearly_coordinated_turn_model(
    dt: float = 1.0,
    position_spectral_density: float = 1.0,
    turn_rate_variance: float = 1e-4,
) -> AdditiveNoiseTransitionModel:
    """Return a coordinated-turn model with a simple nearly-constant-turn covariance."""
    covariance = np.zeros((5, 5), dtype=float)
    covariance[:4, :4] = np.asarray(
        white_noise_acceleration_covariance(
            dt, spatial_dim=2, spectral_density=position_spectral_density
        ),
        dtype=float,
    )
    covariance[4, 4] = float(turn_rate_variance) * float(dt)
    return coordinated_turn_model(dt=dt, noise_covariance=asarray(covariance))


def nearly_constant_speed_transition(state, dt: float = 1.0):
    """Propagate a 2D ``[x, y, speed, heading]`` nearly-constant-speed state."""
    state = asarray(state)
    x_pos, y_pos, speed, heading = state[0], state[1], state[2], state[3]
    return stack(
        [
            x_pos + speed * cos(heading) * float(dt),
            y_pos + speed * sin(heading) * float(dt),
            speed,
            heading,
        ]
    )


def nearly_constant_speed_model(
    dt: float = 1.0, noise_covariance: Any | None = None
) -> AdditiveNoiseTransitionModel:
    """Return an additive-noise nearly-constant-speed model."""
    if noise_covariance is None:
        noise_covariance = zeros((4, 4))
    return AdditiveNoiseTransitionModel(
        transition_function=nearly_constant_speed_transition,
        noise_covariance=noise_covariance,
        dt=dt,
    )


def se2_unicycle_transition(state, dt: float = 1.0, turn_threshold: float = 1e-8):
    """Propagate an SE(2)-style unicycle state ``[x, y, theta, v, omega]``."""
    state = asarray(state)
    x_pos, y_pos, theta, speed, omega = state[0], state[1], state[2], state[3], state[4]
    theta_next = theta + omega * float(dt)
    safe_omega = where(_abs(omega) < float(turn_threshold), 1.0, omega)
    arc_x = x_pos + speed / safe_omega * (sin(theta_next) - sin(theta))
    arc_y = y_pos - speed / safe_omega * (cos(theta_next) - cos(theta))
    linear_x = x_pos + speed * cos(theta) * float(dt)
    linear_y = y_pos + speed * sin(theta) * float(dt)
    is_nearly_straight = _abs(omega) < float(turn_threshold)
    return stack(
        [
            where(is_nearly_straight, linear_x, arc_x),
            where(is_nearly_straight, linear_y, arc_y),
            theta_next,
            speed,
            omega,
        ]
    )


def se2_unicycle_model(
    dt: float = 1.0, noise_covariance: Any | None = None
) -> AdditiveNoiseTransitionModel:
    """Return an additive-noise SE(2)-style unicycle transition model."""
    if noise_covariance is None:
        noise_covariance = zeros((5, 5))
    return AdditiveNoiseTransitionModel(
        transition_function=se2_unicycle_transition,
        noise_covariance=noise_covariance,
        dt=dt,
    )


def se3_pose_twist_transition(state, dt: float = 1.0):
    """Propagate a local SE(3)-style pose/twist state.

    The state is ``[x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]``. The
    pose is updated in local coordinates by integrating linear and angular
    velocities over ``dt``.
    """
    state = asarray(state)
    return stack(
        [
            state[0] + state[6] * float(dt),
            state[1] + state[7] * float(dt),
            state[2] + state[8] * float(dt),
            state[3] + state[9] * float(dt),
            state[4] + state[10] * float(dt),
            state[5] + state[11] * float(dt),
            state[6],
            state[7],
            state[8],
            state[9],
            state[10],
            state[11],
        ]
    )


def se3_pose_twist_model(
    dt: float = 1.0, noise_covariance: Any | None = None
) -> AdditiveNoiseTransitionModel:
    """Return an additive-noise local SE(3) pose/twist transition model."""
    if noise_covariance is None:
        noise_covariance = zeros((12, 12))
    return AdditiveNoiseTransitionModel(
        transition_function=se3_pose_twist_transition,
        noise_covariance=noise_covariance,
        dt=dt,
    )
