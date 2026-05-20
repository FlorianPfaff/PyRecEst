import copy
from numbers import Integral

# pylint: disable=no-name-in-module
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import all as backend_all
from pyrecest.backend import (
    allclose,
    asarray,
    exp,
    eye,
    linalg,
    matvec,
    ndim,
    random,
    reshape,
    transpose,
)

from .abstract_linear_distribution import AbstractLinearDistribution


def _as_backend_array(value, name):
    """Return ``value`` as a backend array with a user-facing error."""
    try:
        return pyrecest.backend.asarray(value)
    except Exception as exc:  # pragma: no cover - backend-specific exception type
        raise ValueError(f"{name} must be convertible to a backend array.") from exc


def _to_python_bool(value):
    """Convert scalar backend boolean values to Python ``bool``."""
    if isinstance(value, bool):
        return value
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


def _validate_same_dimension(first, second, operation):
    if first.dim != second.dim:
        raise ValueError(
            f"Cannot {operation} Gaussian distributions with dimensions "
            f"{first.dim} and {second.dim}."
        )


class GaussianDistribution(AbstractLinearDistribution):
    """Multivariate Gaussian distribution on a Euclidean space.

    Parameters
    ----------
    mu : backend array, shape (n,) or 0-D scalar array
        Mean vector. A 0-D backend array is treated as a one-dimensional
        Gaussian.
    C : backend array, shape (n, n) or 0-D scalar array
        Positive-definite covariance matrix. A 0-D backend array is treated as
        the covariance of a one-dimensional Gaussian.
    check_validity : bool, optional
        If true, validate that ``C`` is symmetric positive definite.

    Attributes
    ----------
    mu : array-like, shape (n,)
        Mean vector.
    C : array-like, shape (n, n)
        Covariance matrix.
    """

    def __init__(self, mu, C, check_validity=True):
        mu = _as_backend_array(mu, "mu")
        C = _as_backend_array(C, "C")
        if ndim(mu) > 1:
            raise ValueError("mu must be one-dimensional or scalar.")
        if ndim(mu) == 0:
            mu = reshape(mu, (1,))

        if ndim(C) == 0 and mu.shape[0] == 1:
            C = reshape(C, (1, 1))
        if ndim(C) != 2:
            raise ValueError("C must be two-dimensional or scalar for dim=1.")
        AbstractLinearDistribution.__init__(self, dim=mu.shape[0])
        expected_shape = (mu.shape[0], mu.shape[0])
        if C.shape != expected_shape:
            raise ValueError(f"C must have shape {expected_shape}, got {C.shape}.")
        self.mu = mu

        if check_validity:
            if not _to_python_bool(allclose(C, transpose(C))):
                raise ValueError("C must be symmetric.")
            if not _to_python_bool(backend_all(linalg.eigvalsh(C) > 0.0)):
                raise ValueError("C must be positive definite.")

        self.C = C

    def set_mean(self, new_mean):
        """Return a copy with a replaced mean vector.

        Parameters
        ----------
        new_mean : array-like, shape (n,)
            New mean vector.
        """
        new_mean = _as_backend_array(new_mean, "new_mean")
        if ndim(new_mean) != 1 or new_mean.shape[0] != self.dim:
            raise ValueError(
                f"new_mean must have shape ({self.dim},), got {new_mean.shape}."
            )
        new_dist = copy.deepcopy(self)
        new_dist.mu = new_mean
        return new_dist

    def _validate_evaluation_points(self, xs):
        xs = _as_backend_array(xs, "xs")
        if not (
            (self.dim == 1 and ndim(xs) <= 1)
            or (ndim(xs) >= 1 and xs.shape[-1] == self.dim)
        ):
            raise ValueError(
                f"xs must have trailing dimension {self.dim}; got shape {xs.shape}."
            )
        return xs

    def pdf(self, xs):
        """Evaluate the probability density.

        Parameters
        ----------
        xs : array-like, shape (n,) or (..., n)
            Evaluation point or batch of evaluation points. For a
            one-dimensional Gaussian, one-dimensional arrays are interpreted as
            a batch of scalar evaluation points.

        Returns
        -------
        array-like
            Density values with one value per evaluation point.
        """
        return exp(self.ln_pdf(xs))

    def ln_pdf(self, xs):
        """Evaluate the log probability density.

        Log-density evaluation is preferable to :meth:`pdf` when products of
        many likelihoods are accumulated or when densities may underflow.
        """
        xs = self._validate_evaluation_points(xs)
        if pyrecest.backend.__backend_name__ == "numpy":
            from scipy.stats import multivariate_normal as mvn

            log_pdf_vals = mvn.logpdf(xs, self.mu, self.C)
        elif pyrecest.backend.__backend_name__ == "pytorch":
            # Disable import errors for megalinter
            import torch as _torch  # pylint: disable=import-error

            distribution = _torch.distributions.MultivariateNormal(self.mu, self.C)
            if ndim(xs) <= 1 and self.dim == 1:
                # For 1-D distributions, reshape the input to a 2-D tensor so
                # that torch.distributions.MultivariateNormal sees event dim 1.
                xs = _torch.reshape(xs, (-1, 1))
            log_pdf_vals = distribution.log_prob(xs)
        elif pyrecest.backend.__backend_name__ == "jax":
            from jax.scipy.stats import (  # pylint: disable=import-error
                multivariate_normal,
            )

            if ndim(xs) <= 1 and self.dim == 1:
                xs = reshape(xs, (-1, 1))

            log_pdf_vals = multivariate_normal.logpdf(xs, self.mu, self.C)
        else:
            raise NotImplementedError("Backend not supported")

        return log_pdf_vals

    log_pdf = ln_pdf

    def shift(self, shift_by):
        """Return a copy with its mean shifted by ``shift_by``.

        Parameters
        ----------
        shift_by : array-like, shape (n,) or scalar
            Additive shift for the mean vector.
        """
        shift_by = _as_backend_array(shift_by, "shift_by")
        if not (
            (ndim(shift_by) == 0 and self.dim == 1)
            or (ndim(shift_by) == 1 and shift_by.shape[0] == self.dim)
        ):
            raise ValueError(
                f"shift_by must be scalar for dim=1 or have shape ({self.dim},)."
            )

        new_gaussian = copy.deepcopy(self)
        new_gaussian.mu = self.mu + shift_by
        return new_gaussian

    def mean(self):
        """Return the mean vector with shape ``(n,)``."""
        return self.mu

    def mode(self, starting_point=None):
        """Return the mode of the Gaussian, equal to the mean vector."""
        _ = starting_point
        return self.mu

    def set_mode(self, new_mode):
        """Return a copy with a replaced mode.

        For a Gaussian distribution, the mode and mean are identical.
        """
        return self.set_mean(new_mode)

    def covariance(self):
        """Return the covariance matrix with shape ``(n, n)``."""
        return self.C

    def multiply(self, other):
        """Multiply two Gaussian densities and return the normalized product.

        Parameters
        ----------
        other : GaussianDistribution
            Gaussian distribution with the same dimension as ``self``.

        Returns
        -------
        GaussianDistribution
            Gaussian proportional to the pointwise product of both densities.
        """
        _validate_same_dimension(self, other, "multiply")
        identity = eye(self.dim)
        self_precision = linalg.solve(self.C, identity)
        other_precision = linalg.solve(other.C, identity)
        new_precision = self_precision + other_precision
        information_vector = matvec(self_precision, self.mu) + matvec(
            other_precision, other.mu
        )
        new_mu = linalg.solve(new_precision, information_vector)
        new_C = linalg.solve(new_precision, identity)
        new_C = 0.5 * (new_C + transpose(new_C))
        return GaussianDistribution(new_mu, new_C, check_validity=False)

    def convolve(self, other):
        """Convolve two independent Gaussian distributions.

        Parameters
        ----------
        other : GaussianDistribution
            Gaussian distribution with the same dimension as ``self``.

        Returns
        -------
        GaussianDistribution
            Gaussian whose mean and covariance are the sums of both operands.
        """
        _validate_same_dimension(self, other, "convolve")
        new_mu = self.mu + other.mu
        new_C = self.C + other.C
        return GaussianDistribution(new_mu, new_C, check_validity=False)

    def marginalize_out(self, dimensions):
        """Return the marginal distribution after dropping dimensions.

        Parameters
        ----------
        dimensions : int or iterable of int
            Zero-based state dimensions to remove from the distribution.
        """
        if isinstance(dimensions, Integral):  # Make it iterable if single integer
            dimensions = [dimensions]
        else:
            dimensions = list(dimensions)

        invalid_dimensions = [
            dim
            for dim in dimensions
            if not isinstance(dim, Integral) or not 0 <= int(dim) < self.dim
        ]
        if invalid_dimensions:
            raise ValueError(
                "dimensions must contain valid zero-based integer indices; "
                f"got {invalid_dimensions}."
            )

        dimensions = [int(dim) for dim in dimensions]
        remaining_dims = [i for i in range(self.dim) if i not in dimensions]
        new_mu = self.mu[remaining_dims]
        new_C = self.C[remaining_dims][
            :, remaining_dims
        ]  # Instead of np.ix_ for interface compatibiliy
        return GaussianDistribution(new_mu, new_C, check_validity=False)

    def sample(self, n):
        """Draw ``n`` random samples with shape ``(n, dim)``."""
        if not isinstance(n, Integral) or int(n) <= 0:
            raise ValueError("n must be a positive integer.")
        return random.multivariate_normal(mean=self.mu, cov=self.C, size=int(n))

    @staticmethod
    def from_distribution(distribution, check_validity=False):
        """Approximate or convert another distribution as a Gaussian.

        Gaussian mixtures are converted with ``to_gaussian``. Other
        distributions must expose mean and covariance information compatible
        with :class:`GaussianDistribution`.
        """
        from .gaussian_mixture import GaussianMixture

        if isinstance(distribution, GaussianMixture):
            gaussian = distribution.to_gaussian(check_validity=check_validity)
        else:
            mean = distribution.mean
            if callable(mean):
                mean = mean()

            covariance = distribution.covariance
            if callable(covariance):
                covariance = covariance()

            gaussian = GaussianDistribution(
                mean, covariance, check_validity=check_validity
            )
        return gaussian
