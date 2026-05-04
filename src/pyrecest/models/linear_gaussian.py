"""Reusable linear Gaussian transition and measurement models."""

from pyrecest.backend import (
    asarray,
    eye,
    matmul,
    matvec,
    ndim,
    reshape,
    transpose,
    zeros,
)
from pyrecest.distributions import GaussianDistribution


def _as_matrix(value, name):
    arr = asarray(value)
    if ndim(arr) != 2:
        raise ValueError(f"{name} must be two-dimensional")
    return arr


def _as_vector(value, name):
    arr = asarray(value)
    if ndim(arr) == 0:
        arr = reshape(arr, (1,))
    if ndim(arr) != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return arr


def _shape(value):
    return tuple(value.shape)


def _as_square(value, name):
    arr = _as_matrix(value, name)
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be square")
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


class LinearGaussianTransitionModel:
    """Transition model ``x_next = F x + u + w`` with Gaussian noise."""

    def __init__(self, matrix, noise_cov, offset=None):
        self.matrix = _as_matrix(matrix, "matrix")
        self.noise_cov = _as_square(noise_cov, "noise_cov")
        self.predicted_dim, self.state_dim = _shape(self.matrix)
        if _shape(self.noise_cov) != (self.predicted_dim, self.predicted_dim):
            raise ValueError("noise_cov has incompatible shape")
        self.offset = None if offset is None else _as_vector(offset, "offset")
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
    def __init__(self, dim, noise_cov, offset=None):
        super().__init__(eye(dim), noise_cov, offset=offset)


class LinearGaussianMeasurementModel:
    """Measurement model ``z = H x + v`` with Gaussian noise."""

    def __init__(self, matrix, noise_cov):
        self.matrix = _as_matrix(matrix, "matrix")
        self.noise_cov = _as_square(noise_cov, "noise_cov")
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
    def __init__(self, dim, noise_cov):
        super().__init__(eye(dim), noise_cov)
