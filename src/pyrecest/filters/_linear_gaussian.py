# pylint: disable=no-name-in-module,no-member
"""Backend-native linear-Gaussian predict/update primitives."""

from pyrecest.backend import (
    asarray,
    atleast_1d,
    atleast_2d,
    eye,
    float64,
    linalg,
    maximum,
    sqrt,
    transpose,
)


def _as_vector(x, name):
    x = atleast_1d(asarray(x, dtype=float64))
    if len(x.shape) != 1:
        raise ValueError(f"{name} must be one-dimensional after coercion")
    return x


def _as_matrix(x, name):
    x = atleast_2d(asarray(x, dtype=float64))
    if len(x.shape) != 2:
        raise ValueError(f"{name} must be two-dimensional after coercion")
    return x


def huber_covariance_scale(normalized_innovation_squared, huber_threshold=2.0):
    """Return measurement-covariance scaling for a Huber robust update.

    The Huber weight is one for Mahalanobis innovation norm below ``k`` and
    ``k / norm`` for outliers. Inflating the measurement covariance by the
    reciprocal weight gives

    ``max(1, sqrt(normalized_innovation_squared) / huber_threshold)``.

    Parameters
    ----------
    normalized_innovation_squared : scalar or array-like
        Squared Mahalanobis innovation, i.e. NIS.
    huber_threshold : float, optional
        Huber threshold ``k`` in Mahalanobis-norm units. Must be positive.
    """
    huber_threshold = float(huber_threshold)
    if huber_threshold <= 0.0:
        raise ValueError("huber_threshold must be positive")

    nis = asarray(normalized_innovation_squared, dtype=float64)
    return maximum(1.0, sqrt(nis) / huber_threshold)


def student_t_covariance_scale(
    normalized_innovation_squared,
    measurement_dim,
    dof=4.0,
    min_scale=1.0,
):
    """Return Student-t measurement-covariance scaling from innovation NIS.

    This helper implements the scale-mixture weight used for an approximate
    Student-t Kalman measurement update. For normalized innovation squared
    ``nis``, measurement dimension ``d``, and degrees of freedom ``nu``, the
    Student-t IRLS/EM weight is

    ``w = (nu + d) / (nu + nis)``.

    A Gaussian update can therefore be made heavy-tailed by replacing the
    measurement covariance ``R`` with ``R * scale``, where ``scale = 1 / w``.
    The default ``min_scale=1`` prevents inliers from becoming more confident
    than the supplied Gaussian measurement model.

    Parameters
    ----------
    normalized_innovation_squared : scalar or array-like
        Squared Mahalanobis innovation, i.e. NIS.
    measurement_dim : int
        Dimension ``d`` of the measurement vector. Must be positive.
    dof : float, optional
        Student-t degrees of freedom ``nu``. Must be greater than two.
    min_scale : float, optional
        Lower bound on the returned covariance scale. Must be nonnegative.
    """
    measurement_dim = int(measurement_dim)
    if measurement_dim <= 0:
        raise ValueError("measurement_dim must be positive")

    dof = float(dof)
    if dof <= 2.0:
        raise ValueError("dof must be greater than 2")

    min_scale = float(min_scale)
    if min_scale < 0.0:
        raise ValueError("min_scale must be nonnegative")

    nis = asarray(normalized_innovation_squared, dtype=float64)
    scale = (dof + nis) / (dof + measurement_dim)
    return maximum(min_scale, scale)


def linear_gaussian_predict(
    mean, covariance, system_matrix, sys_noise_cov, sys_input=None
):
    """Predict step for x_k = F x_{k-1} + u + w with w ~ N(0, Q)."""
    mean = _as_vector(mean, "mean")
    covariance = _as_matrix(covariance, "covariance")
    system_matrix = _as_matrix(system_matrix, "system_matrix")
    sys_noise_cov = _as_matrix(sys_noise_cov, "sys_noise_cov")

    state_dim = mean.shape[0]
    pred_dim = system_matrix.shape[0]

    if covariance.shape != (state_dim, state_dim):
        raise ValueError("covariance must have shape (state_dim, state_dim)")
    if system_matrix.shape[1] != state_dim:
        raise ValueError("system_matrix has incompatible shape")
    if sys_noise_cov.shape != (pred_dim, pred_dim):
        raise ValueError("sys_noise_cov must have shape (pred_dim, pred_dim)")

    predicted_mean = system_matrix @ mean
    if sys_input is not None:
        sys_input = _as_vector(sys_input, "sys_input")
        if sys_input.shape[0] != pred_dim:
            raise ValueError(
                "The number of elements in sys_input must match the number of rows "
                "in system_matrix"
            )
        predicted_mean = predicted_mean + sys_input

    predicted_covariance = (
        system_matrix @ covariance @ transpose(system_matrix) + sys_noise_cov
    )
    predicted_covariance = 0.5 * (
        predicted_covariance + transpose(predicted_covariance)
    )
    return predicted_mean, predicted_covariance


def linear_gaussian_update(
    mean, covariance, measurement, measurement_matrix, meas_noise
):
    """Update step for z_k = H x_k + v with v ~ N(0, R)."""
    mean = _as_vector(mean, "mean")
    covariance = _as_matrix(covariance, "covariance")
    measurement = _as_vector(measurement, "measurement")
    measurement_matrix = _as_matrix(measurement_matrix, "measurement_matrix")
    meas_noise = _as_matrix(meas_noise, "meas_noise")

    state_dim = mean.shape[0]
    meas_dim = measurement_matrix.shape[0]

    if covariance.shape != (state_dim, state_dim):
        raise ValueError("covariance must have shape (state_dim, state_dim)")
    if measurement_matrix.shape[1] != state_dim:
        raise ValueError("measurement_matrix has incompatible shape")
    if measurement.shape[0] != meas_dim:
        raise ValueError("measurement has incompatible shape")
    if meas_noise.shape != (meas_dim, meas_dim):
        raise ValueError("meas_noise must have shape (meas_dim, meas_dim)")

    innovation = measurement - measurement_matrix @ mean
    innovation_cov = (
        measurement_matrix @ covariance @ transpose(measurement_matrix) + meas_noise
    )
    cross_cov = covariance @ transpose(measurement_matrix)

    kalman_gain = transpose(
        linalg.solve(transpose(innovation_cov), transpose(cross_cov))
    )

    updated_mean = mean + kalman_gain @ innovation

    identity = eye(state_dim)
    correction = identity - kalman_gain @ measurement_matrix
    updated_covariance = correction @ covariance @ transpose(correction)
    updated_covariance = updated_covariance + kalman_gain @ meas_noise @ transpose(
        kalman_gain
    )
    updated_covariance = 0.5 * (updated_covariance + transpose(updated_covariance))

    return updated_mean, updated_covariance
