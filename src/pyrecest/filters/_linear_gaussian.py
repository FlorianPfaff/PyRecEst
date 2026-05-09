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


def _as_positive_float(x, name):
    x = float(x)
    if x <= 0.0:
        raise ValueError(f"{name} must be positive")
    return x


def normalized_innovation_squared(innovation, innovation_covariance):
    """Return innovation.T @ inv(innovation_covariance) @ innovation."""
    innovation = _as_vector(innovation, "innovation")
    innovation_covariance = _as_matrix(
        innovation_covariance,
        "innovation_covariance",
    )
    innovation_dim = innovation.shape[0]
    if innovation_covariance.shape != (innovation_dim, innovation_dim):
        raise ValueError(
            "innovation_covariance must have shape "
            "(innovation_dim, innovation_dim)"
        )
    return transpose(innovation) @ linalg.solve(innovation_covariance, innovation)


def huber_covariance_scale(  # pylint: disable=redefined-outer-name
    normalized_innovation_squared,
    huber_threshold=2.0,
):
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


def student_t_covariance_scale(  # pylint: disable=redefined-outer-name
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


def _robust_update_decision(
    normalized_innovation_squared_value,
    measurement_dim,
    *,
    robust_update,
    gate_threshold,
    student_t_dof,
    huber_threshold,
    inflation_alpha,
):
    robust_update = None if robust_update in (None, "none") else robust_update
    nis = float(normalized_innovation_squared_value)

    if robust_update is None:
        if gate_threshold is not None:
            gate_threshold = _as_positive_float(gate_threshold, "gate_threshold")
            if nis > gate_threshold:
                return False, "rejected", 1.0
        return True, "updated", 1.0

    if robust_update == "nis-inflate":
        inflation_alpha = _as_positive_float(inflation_alpha, "inflation_alpha")
        if gate_threshold is None:
            return True, "updated", 1.0
        gate_threshold = _as_positive_float(gate_threshold, "gate_threshold")
        scale = max(1.0, (nis / gate_threshold) ** inflation_alpha)
        return True, "inflated" if scale > 1.0 else "updated", scale

    if robust_update == "student-t":
        scale = float(
            student_t_covariance_scale(nis, measurement_dim, dof=student_t_dof)
        )
        return True, "student_t" if scale > 1.0 else "updated", scale

    if robust_update == "huber":
        scale = float(huber_covariance_scale(nis, huber_threshold=huber_threshold))
        return True, "huberized" if scale > 1.0 else "updated", scale

    raise ValueError(f"unknown robust update mode {robust_update!r}")


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
    mean,
    covariance,
    measurement,
    measurement_matrix,
    meas_noise,
    *,
    return_diagnostics=False,
    scale=1.0,
    action="updated",
):
    """Update step for z_k = H x_k + v with v ~ N(0, R).

    If ``return_diagnostics`` is true, return a third value containing the
    normalized innovation squared (NIS), residual, covariance scale, and action.
    ``scale`` multiplies ``meas_noise`` for the update but diagnostics report
    the pre-scaled NIS, matching the usual gating/robust-update convention.
    """
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

    scale = float(scale)
    if scale <= 0.0:
        raise ValueError("scale must be positive")

    innovation = measurement - measurement_matrix @ mean
    nominal_innovation_cov = (
        measurement_matrix @ covariance @ transpose(measurement_matrix) + meas_noise
    )
    scaled_meas_noise = meas_noise * scale
    innovation_cov = (
        measurement_matrix @ covariance @ transpose(measurement_matrix)
        + scaled_meas_noise
    )
    cross_cov = covariance @ transpose(measurement_matrix)

    kalman_gain = transpose(
        linalg.solve(transpose(innovation_cov), transpose(cross_cov))
    )

    updated_mean = mean + kalman_gain @ innovation

    identity = eye(state_dim)
    correction = identity - kalman_gain @ measurement_matrix
    updated_covariance = correction @ covariance @ transpose(correction)
    updated_covariance = updated_covariance + kalman_gain @ scaled_meas_noise @ transpose(
        kalman_gain
    )
    updated_covariance = 0.5 * (updated_covariance + transpose(updated_covariance))

    if return_diagnostics:
        diagnostics = {
            "nis": normalized_innovation_squared(innovation, nominal_innovation_cov),
            "residual": innovation,
            "scale": scale,
            "action": action,
        }
        return updated_mean, updated_covariance, diagnostics

    return updated_mean, updated_covariance


def linear_gaussian_update_robust(
    mean,
    covariance,
    measurement,
    measurement_matrix,
    meas_noise,
    *,
    robust_update="student-t",
    gate_threshold=None,
    student_t_dof=4.0,
    huber_threshold=2.0,
    inflation_alpha=1.0,
    return_diagnostics=False,
):
    """Robust linear-Gaussian update with adaptive measurement covariance.

    Supported modes are ``None``/``"none"`` for ordinary Gaussian updates with
    optional NIS rejection, ``"student-t"`` for Student-t down-weighting,
    ``"huber"`` for Huber down-weighting, and ``"nis-inflate"`` for
    threshold-based covariance inflation.
    """
    mean = _as_vector(mean, "mean")
    covariance = _as_matrix(covariance, "covariance")
    measurement = _as_vector(measurement, "measurement")
    measurement_matrix = _as_matrix(measurement_matrix, "measurement_matrix")
    meas_noise = _as_matrix(meas_noise, "meas_noise")

    meas_dim = measurement_matrix.shape[0]
    innovation = measurement - measurement_matrix @ mean
    nominal_innovation_cov = (
        measurement_matrix @ covariance @ transpose(measurement_matrix) + meas_noise
    )
    nis = normalized_innovation_squared(innovation, nominal_innovation_cov)
    accepted, action, scale = _robust_update_decision(
        nis,
        meas_dim,
        robust_update=robust_update,
        gate_threshold=gate_threshold,
        student_t_dof=student_t_dof,
        huber_threshold=huber_threshold,
        inflation_alpha=inflation_alpha,
    )

    if not accepted:
        diagnostics = {
            "nis": nis,
            "residual": innovation,
            "scale": scale,
            "action": action,
            "accepted": False,
            "robust_update": robust_update,
        }
        if return_diagnostics:
            return mean, covariance, diagnostics
        return mean, covariance

    result = linear_gaussian_update(
        mean,
        covariance,
        measurement,
        measurement_matrix,
        meas_noise,
        return_diagnostics=return_diagnostics,
        scale=scale,
        action=action,
    )

    if return_diagnostics:
        updated_mean, updated_covariance, diagnostics = result
        diagnostics["accepted"] = True
        diagnostics["robust_update"] = robust_update
        return updated_mean, updated_covariance, diagnostics

    return result
