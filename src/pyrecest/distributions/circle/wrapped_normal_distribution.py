from math import isfinite
from numbers import Integral
from operator import index as _operator_index
from typing import Union

import pyrecest.backend

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    all,
    angle,
    any,
    array,
    erf,
    exp,
    int32,
    int64,
    isfinite as backend_isfinite,
    log,
    mod,
    ndim,
    ones,
    pi,
    random,
    sqrt,
    squeeze,
    where,
    zeros,
)

from ..hypertorus._input_validation import as_shift_vector
from ..hypertorus.hypertoroidal_wrapped_normal_distribution import (
    HypertoroidalWrappedNormalDistribution,
    _validate_series_order,
)
from .abstract_circular_distribution import AbstractCircularDistribution
from .von_mises_distribution import VonMisesDistribution


def _validate_finite_scalar(value, name):
    value = array(value)
    if value.shape not in ((), (1,)):
        raise ValueError(f"{name} must be a scalar.")
    if not bool(all(backend_isfinite(value))):
        raise ValueError(f"{name} must be finite.")
    return value


class WrappedNormalDistribution(
    AbstractCircularDistribution, HypertoroidalWrappedNormalDistribution
):
    """Wrapped normal distribution on the circle.

    References
    ----------
    Jammalamadaka, S. R., & SenGupta, A. (2001). Topics in Circular
    Statistics. World Scientific.
    """

    MAX_SIGMA_BEFORE_UNIFORM = 10
    _MOMENT_NORM_TOL = 1e-12

    def __init__(
        self,
        mu,
        sigma,
    ):
        """
        Initialize a wrapped normal distribution with mean mu and standard deviation sigma.
        """
        mu = array(mu)
        sigma = array(sigma)
        if ndim(mu) > 1 or (ndim(mu) == 1 and mu.shape[0] != 1):
            raise ValueError(f"mu must be a scalar, but got shape {mu.shape}.")
        if ndim(sigma) > 1 or (ndim(sigma) == 1 and sigma.shape[0] != 1):
            raise ValueError(f"sigma must be a scalar, but got shape {sigma.shape}.")
        sigma_scalar = squeeze(sigma)
        if not bool(isfinite(sigma_scalar)) or not bool(sigma_scalar > 0):
            raise ValueError(f"sigma must be a positive finite scalar, got {sigma}.")
        AbstractCircularDistribution.__init__(self)
        HypertoroidalWrappedNormalDistribution.__init__(
            self, squeeze(mu), sigma_scalar**2
        )

    @property
    def sigma(self):
        return squeeze(sqrt(self.C))

    @property
    def scalar_mu(self):
        return squeeze(self.mu)

    # pylint: disable=too-many-locals
    def pdf(self, xs, m: Union[int, int32, int64] = 3):
        m = _validate_series_order(m)
        sigma = self.sigma
        mu = self.scalar_mu
        if sigma <= 0:
            raise ValueError(f"sigma must be >0, but received {sigma}.")
        xs = array(xs)
        if ndim(xs) == 0:
            xs = array([xs])
        # check if sigma is large and return uniform distribution in this case
        if sigma > self.MAX_SIGMA_BEFORE_UNIFORM:
            return (1.0 / (2.0 * pi) * ones(xs.shape[0])).squeeze()
        x = mod(xs, 2.0 * pi)
        x = where(x < 0, x + 2.0 * pi, x)
        x -= mu
        if pyrecest.backend.__backend_name__ != "jax":
            n_inputs = xs.shape[0]
            result = zeros(n_inputs)

            tmp = -1.0 / (2.0 * sigma**2)
            nc = 1.0 / sqrt(2.0 * pi) / sigma

            for i in range(n_inputs):
                result[i] = squeeze(exp(x[i] * x[i] * tmp))
                for k in range(1, m + 1):
                    xp = x[i] + 2 * pi * k
                    xm = x[i] - 2 * pi * k
                    tp = xp * xp * tmp
                    tm = xm * xm * tmp
                    increment = (exp(tp) + exp(tm)).squeeze()
                    new_result = result[i] + increment

                    if new_result == result[i]:
                        break

                    result[i] = new_result

                result[i] *= nc
        else:
            from jax import lax  # pylint: disable=import-error
            from jax.numpy import logical_and  # pylint: disable=import-error

            tmp = -1.0 / (2.0 * sigma**2)
            nc = 1.0 / (sqrt(2.0 * pi) * sigma)

            def body_fun(val):
                i, result, _last_increment = val
                xp = x + 2 * pi * i
                xm = x - 2 * pi * i
                tp = xp * xp * tmp
                tm = xm * xm * tmp
                increment = exp(tp) + exp(tm)
                new_result = result + increment
                return (i + 1, new_result, increment)

            def cond_fun(val):
                i, _result, last_increment = val
                # The accumulated density is positive and may keep changing by
                # tiny floating-point amounts for a long time. Stop based on the
                # latest wrapped contribution instead.
                return logical_and(any(last_increment > 1e-10), i <= m)

            initial_val = (
                1,
                exp(x * x * tmp),
                ones(x.shape) * float("inf"),
            )
            _, result, _last_increment = lax.while_loop(cond_fun, body_fun, initial_val)

            result *= nc

        return result.squeeze()

    def cdf(
        self,
        xs,
        starting_point: float = 0.0,
        n_wraps: Union[int, int32, int64] = 10,
    ):
        n_wraps = _validate_series_order(n_wraps)
        mu = self.scalar_mu
        sigma = self.sigma
        starting_point = _validate_finite_scalar(starting_point, "starting_point")
        starting_point = mod(starting_point, 2 * pi)
        xs = mod(xs, 2 * pi)

        def ncdf(from_, to):
            return (
                1
                / 2
                * (
                    erf((mu - from_) / (sqrt(2) * sigma))
                    - erf((mu - to) / (sqrt(2) * sigma))
                )
            )

        val = ncdf(starting_point, xs)
        for i in range(1, n_wraps + 1):
            val = (
                val
                + ncdf(starting_point + 2 * pi * i, xs + 2 * pi * i)
                + ncdf(starting_point - 2 * pi * i, xs - 2 * pi * i)
            )
        # Val should be negative when x < starting_point
        val = where(xs < starting_point, 1 + val, val)
        return squeeze(val)

    def trigonometric_moment(self, n: Union[int, int32, int64]):
        dtype = getattr(n, "dtype", None)
        if isinstance(n, bool) or (
            dtype is not None and str(dtype).lower().endswith("bool")
        ):
            raise ValueError("n must be an integer")
        try:
            n = int(_operator_index(n))
        except (TypeError, ValueError) as exc:
            raise ValueError("n must be an integer") from exc
        return exp(1j * n * self.scalar_mu - n**2 * self.sigma**2 / 2)

    def multiply(
        self, other: "WrappedNormalDistribution"
    ) -> "WrappedNormalDistribution":
        """Return a wrapped-normal approximation of the density product.

        Wrapped normal distributions are not closed under pointwise
        multiplication on the circle. To preserve the existing single-component
        wrapped-normal API, this method intentionally returns the same
        von-Mises-based approximation as :meth:`multiply_vm_approximation`.
        """
        return self.multiply_vm_approximation(other)

    def multiply_vm_approximation(
        self, other: "WrappedNormalDistribution"
    ) -> "WrappedNormalDistribution":
        """Approximate a product through the von Mises family.

        The two wrapped normals are converted to von Mises distributions with
        ``kappa = 1 / sigma**2``. Their von Mises product is computed in closed
        form and then converted back to a wrapped normal by first-moment
        matching. The returned density is therefore not the exact normalized
        product of the two wrapped-normal densities.
        """
        if not isinstance(other, WrappedNormalDistribution):
            raise TypeError("other must be a WrappedNormalDistribution")
        vm1 = self.to_vm()
        vm2 = other.to_vm()
        vm = vm1.multiply(vm2)
        wn = vm.to_wn()
        return wn

    def multiply_vm(self, other: "WrappedNormalDistribution"):
        """Backward-compatible alias for :meth:`multiply_vm_approximation`."""
        return self.multiply_vm_approximation(other)

    def convolve(
        self, other: HypertoroidalWrappedNormalDistribution
    ) -> "WrappedNormalDistribution":
        """Convolve two 1-D wrapped normal distributions.

        The circular specialization exposes ``sigma`` in its constructor, while the
        hypertoroidal base class stores the covariance ``C``. Therefore, the
        covariance sum has to be converted back to a standard deviation before
        constructing the result.
        """
        if not isinstance(other, WrappedNormalDistribution):
            raise TypeError("other must be a WrappedNormalDistribution")
        return WrappedNormalDistribution(
            mod(self.scalar_mu + other.scalar_mu, 2.0 * pi),
            sqrt(squeeze(self.C) + squeeze(other.C)),
        )

    def sample(self, n: Union[int, int32, int64]):
        if isinstance(n, bool) or not isinstance(n, Integral) or int(n) <= 0:
            raise ValueError("n must be a positive integer.")
        n = int(n)
        return mod(self.scalar_mu + self.sigma * random.normal(size=(n,)), 2.0 * pi)

    def to_dirac5(self):
        from .circular_dirac_distribution import CircularDiracDistribution

        offsets = array([-2.0, -1.0, 0.0, 1.0, 2.0])
        weights = array([1.0, 2.0, 6.0, 2.0, 1.0]) / 12.0
        samples = mod(self.mu + self.sigma * offsets, 2.0 * pi)
        return CircularDiracDistribution(samples, weights)

    def shift(self, shift_by):
        shift_by = as_shift_vector(shift_by, 1)
        return WrappedNormalDistribution(self.scalar_mu + shift_by[0], self.sigma)

    def to_vm(self) -> VonMisesDistribution:
        # Convert to Von Mises distribution
        kappa = self.sigma_to_kappa(self.sigma)
        return VonMisesDistribution(self.scalar_mu, kappa)

    @staticmethod
    def from_moment(m) -> "WrappedNormalDistribution":
        moment = squeeze(array(m))
        if ndim(moment) != 0:
            raise ValueError("First trigonometric moment must be a scalar.")
        moment_abs = float(abs(moment))
        if not isfinite(moment_abs):
            raise ValueError("First trigonometric moment must be finite.")
        if moment_abs > 1.0 + WrappedNormalDistribution._MOMENT_NORM_TOL:
            raise ValueError(
                "First trigonometric moment must have magnitude at most 1."
            )

        moment_abs = min(moment_abs, 1.0)
        if moment_abs == 0.0:
            raise ValueError(
                "A zero first trigonometric moment cannot be represented by a "
                "wrapped normal with finite variance."
            )
        if moment_abs == 1.0:
            raise ValueError(
                "First trigonometric moment with |m| = 1 cannot be moment-matched "
                "to a wrapped normal with positive variance."
            )

        mu = mod(angle(moment), 2.0 * pi)
        sigma = sqrt(-2 * log(moment_abs))
        return WrappedNormalDistribution(mu, sigma)

    @staticmethod
    def sigma_to_kappa(sigma):
        # Approximate conversion from sigma to a Von Mises distribution
        return 1.0 / sigma**2
