"""Reusable linear Gaussian transition and measurement models."""

from numbers import Complex, Integral, Real

from pyrecest.backend import (
    asarray,
)
from pyrecest.backend import copy as backend_copy
from pyrecest.backend import (
    eye,
    matmul,
    matvec,
    ndim,
    reshape,
    transpose,
    zeros,
)
from pyrecest.distributions import GaussianDistribution

_INVALID_DIMENSION_SCALAR_TYPES = (bool, str, bytes, bytearray)
_NOISE_COV_UNSET = object()


def _has_boolean_dtype(value):
    dtype = getattr(value, "dtype", None)
    return dtype is not None and str(dtype).lower() in {"bool", "bool_", "torch.bool"}


def _has_complex_dtype(value):
    dtype = getattr(value, "dtype", None)
    return dtype is not None and "complex" in str(dtype).lower()


def _contains_complex_values(value):
    if _has_complex_dtype(value):
        return True

    dtype = getattr(value, "dtype", None)
    if dtype is not None and "object" not in str(dtype).lower():
        return False

    try:
        values = value.reshape(-1)
    except (AttributeError, TypeError, ValueError):
        values = (value,)
    for item in values:
        try:
            scalar = item.item()
        except AttributeError:
            scalar = item
        if isinstance(scalar, Complex) and not isinstance(scalar, Real):
            return True
    return False


def _as_matrix(value, name):
    arr = asarray(value)
    if _contains_complex_values(arr):
        raise ValueError(f"{name} must be real-valued")
    if ndim(arr) != 2:
        raise ValueError(f"{name} must be two-dimensional")
    return arr


def _as_vector(value, name):
    arr = asarray(value)
    if _contains_complex_values(arr):
        raise ValueError(f"{name} must be real-valued")
    if ndim(arr) == 0:
        arr = reshape(arr, (1,))
    if ndim(arr) != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return arr


def _shape(value):
    return tuple(value.shape)


def _as_positive_integer(value, name):
    message = f"{name} must be a positive integer"
    if isinstance(value, _INVALID_DIMENSION_SCALAR_TYPES):
        raise ValueError(message)
    try:
        arr = asarray(value)
    except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
        raise ValueError(message) from exc
    if ndim(arr) != 0 or _has_boolean_dtype(arr):
        raise ValueError(message)
    try:
        scalar = arr.item()
    except AttributeError:
        scalar = arr
    if isinstance(scalar, _INVALID_DIMENSION_SCALAR_TYPES):
        raise ValueError(message)
    if isinstance(scalar, Integral):
        parsed = int(scalar)
    else:
        try:
            parsed = int(scalar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(message) from exc
        if scalar != parsed:
            raise ValueError(message)
    if parsed <= 0:
        raise ValueError(message)
    return parsed


def _as_square(value, name):
    arr = _as_matrix(value, name)
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be square")
    return arr


def _as_identity_noise_covariance(value, dim):
    arr = asarray(value)
    if ndim(arr) == 0:
        if _has_boolean_dtype(arr):
            raise ValueError("noise_cov must be a scalar variance or square covariance")
        return arr * eye(dim)
    return arr


def _mean(distribution):
    mean = getattr(distribution, "mean", None)
    if callable(mean):
        return mean()
    if mean is not None:
        return mean
    return distribution.mu


def _covariance(distribution):
    covariance = getattr(distribution, "covariance", None)
    if callable(covariance):
        return covariance()
    if covariance is not None:
        return covariance
    return distribution.C


def _resolve_noise_covariance(noise_cov, noise_covariance, model_name):
    if noise_cov is _NOISE_COV_UNSET:
        if noise_covariance is _NOISE_COV_UNSET:
            raise TypeError(f"{model_name} missing required argument: 'noise_cov'")
        return noise_covariance
    if noise_covariance is not _NOISE_COV_UNSET:
        raise TypeError(f"{model_name} got both noise_cov and noise_covariance")
    return noise_cov


class LinearGaussianTransitionModel:
    """Transition model ``x_next = F x + u + w`` with Gaussian noise.

    The noise covariance can be supplied as ``noise_cov`` or, for consistency
    with additive-noise models, as the keyword alias ``noise_covariance``.
    """

    def __init__(
        self,
        matrix,
        noise_cov=_NOISE_COV_UNSET,
        offset=None,
        *,
        noise_covariance=_NOISE_COV_UNSET,
    ):
        noise_cov = _resolve_noise_covariance(
            noise_cov,
            noise_covariance,
            type(self).__name__,
        )
        self.matrix = backend_copy(_as_matrix(matrix, "matrix"))
        self.noise_cov = backend_copy(_as_square(noise_cov, "noise_cov"))
        self.predicted_dim, self.state_dim = _shape(self.matrix)
        if _shape(self.noise_cov) != (self.predicted_dim, self.predicted_dim):
            raise ValueError("noise_cov has incompatible shape")
        self.offset = (
            None if offset is None else backend_copy(_as_vector(offset, "offset"))
        )
        if self.offset is not None and _shape(self.offset) != (self.predicted_dim,):
            raise ValueError("offset has incompatible shape")

    @property
    def sys_noise_cov(self):
        return self.noise_cov

    @property
    def system_noise_cov(self):
        return self.noise_cov

    @property
    def sys_input(self):
        return self.offset

    @property
    def system_matrix(self):
        return self.matrix

    def predict_mean(self, state_mean):
        state_mean = _as_vector(state_mean, "state_mean")
        if _shape(state_mean) != (self.state_dim,):
            raise ValueError("state_mean has incompatible shape")
        result = matvec(self.matrix, state_mean)
        if self.offset is not None:
            result = result + self.offset
        return result

    def predict_covariance(self, state_covariance):
        state_covariance = _as_square(state_covariance, "state_covariance")
        if _shape(state_covariance) != (self.state_dim, self.state_dim):
            raise ValueError("state_covariance has incompatible shape")
        return (
            matmul(matmul(self.matrix, state_covariance), transpose(self.matrix))
            + self.noise_cov
        )

    def predict_distribution(self, state_distribution, check_validity=False):
        return GaussianDistribution(
            self.predict_mean(_mean(state_distribution)),
            self.predict_covariance(_covariance(state_distribution)),
            check_validity=check_validity,
        )

    def noise_distribution(self, check_validity=False):
        return GaussianDistribution(
            zeros((self.predicted_dim,)), self.noise_cov, check_validity=check_validity
        )


class IdentityGaussianTransitionModel(LinearGaussianTransitionModel):
    def __init__(
        self,
        dim,
        noise_cov=_NOISE_COV_UNSET,
        offset=None,
        *,
        noise_covariance=_NOISE_COV_UNSET,
    ):
        dim = _as_positive_integer(dim, "dim")
        noise_cov = _resolve_noise_covariance(
            noise_cov,
            noise_covariance,
            type(self).__name__,
        )
        noise_cov = _as_identity_noise_covariance(noise_cov, dim)
        super().__init__(eye(dim), noise_cov, offset=offset)


class LinearGaussianMeasurementModel:
    """Measurement model ``z = H x + v`` with Gaussian noise.

    The noise covariance can be supplied as ``noise_cov`` or, for consistency
    with additive-noise models, as the keyword alias ``noise_covariance``.
    """

    def __init__(
        self,
        matrix,
        noise_cov=_NOISE_COV_UNSET,
        *,
        noise_covariance=_NOISE_COV_UNSET,
    ):
        noise_cov = _resolve_noise_covariance(
            noise_cov,
            noise_covariance,
            type(self).__name__,
        )
        self.matrix = backend_copy(_as_matrix(matrix, "matrix"))
        self.noise_cov = backend_copy(_as_square(noise_cov, "noise_cov"))
        self.measurement_dim, self.state_dim = _shape(self.matrix)
        if _shape(self.noise_cov) != (self.measurement_dim, self.measurement_dim):
            raise ValueError("noise_cov has incompatible shape")

    @property
    def meas_noise(self):
        return self.noise_cov

    @property
    def measurement_noise_cov(self):
        return self.noise_cov

    @property
    def measurement_matrix(self):
        return self.matrix

    def predict_mean(self, state_mean):
        state_mean = _as_vector(state_mean, "state_mean")
        if _shape(state_mean) != (self.state_dim,):
            raise ValueError("state_mean has incompatible shape")
        return matvec(self.matrix, state_mean)

    def innovation_covariance(self, state_covariance):
        state_covariance = _as_square(state_covariance, "state_covariance")
        if _shape(state_covariance) != (self.state_dim, self.state_dim):
            raise ValueError("state_covariance has incompatible shape")
        return (
            matmul(matmul(self.matrix, state_covariance), transpose(self.matrix))
            + self.noise_cov
        )

    def predict_distribution(self, state_distribution, check_validity=False):
        return GaussianDistribution(
            self.predict_mean(_mean(state_distribution)),
            self.innovation_covariance(_covariance(state_distribution)),
            check_validity=check_validity,
        )

    def noise_distribution(self, check_validity=False):
        return GaussianDistribution(
            zeros((self.measurement_dim,)),
            self.noise_cov,
            check_validity=check_validity,
        )


class IdentityGaussianMeasurementModel(LinearGaussianMeasurementModel):
    def __init__(
        self,
        dim,
        noise_cov=_NOISE_COV_UNSET,
        *,
        noise_covariance=_NOISE_COV_UNSET,
    ):
        dim = _as_positive_integer(dim, "dim")
        noise_cov = _resolve_noise_covariance(
            noise_cov,
            noise_covariance,
            type(self).__name__,
        )
        noise_cov = _as_identity_noise_covariance(noise_cov, dim)
        super().__init__(eye(dim), noise_cov)
