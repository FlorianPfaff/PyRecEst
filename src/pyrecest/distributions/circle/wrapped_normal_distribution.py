from typing import Union

import pyrecest.backend

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    angle,
    any,
    array,
    exp,
    int32,
    int64,
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
from scipy.special import erf  # pylint: disable=no-name-in-module

from ..hypertorus.hypertoroidal_wrapped_normal_distribution import (
    HypertoroidalWrappedNormalDistribution,
)
from .abstract_circular_distribution import AbstractCircularDistribution
from .von_mises_distribution import VonMisesDistribution


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
        if ndim(mu) != 0:
            raise ValueError(f"mu must be a scalar, but got shape {mu.shape}.")
        if ndim(sigma) != 0:
            raise ValueError(f"sigma must be a scalar, but got shape {sigma.shape}.")
        AbstractCircularDistribution.__init__(self)
        HypertoroidalWrappedNormalDistribution.__init__(self, mu, sigma**2)

    @property
    def sigma(self):
        return sqrt(self.C)

    # pylint: disable=too-many-locals
    def pdf(self, xs, m: Union[int, int32, int64] = 3):
        _ = m
        if self.sigma <= 0:
            raise ValueError(f"sigma must be >0, but received {self.sigma}.")
        xs = array(xs)
        if ndim(xs) == 0:
            xs = array([xs])
        # check if sigma is large and return uniform distribution in this case
        if self.sigma > self.MAX_SIGMA_BEFORE_UNIFORM:
            return 1.0 / (2.0 * pi) * ones(xs.shape[0])
        x = mod(xs, 2.0 * pi)
        x = where(x < 0, x + 2.0 * pi, x)
        x -= self.mu
        max_iterations = 1000
        if pyrecest.backend.__backend_name__ != "jax":
            n_inputs = xs.shape[0]
            result = zeros(n_inputs)

            tmp = -1.0 / (2.0 * self.sigma**2)
            nc = 1.0 / sqrt(2.0 * pi) / self.sigma

            for i in range(n_inputs):
                result[i] = squeeze(exp(x[i] * x[i] * tmp))

                for k in range(1, max_iterations + 1):
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

            tmp = -1.0 / (2.0 * self.sigma**2)
            nc = 1.0 / (sqrt(2.0 * pi) * self.sigma)

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
                return logical_and(any(last_increment > 1e-10), i <= max_iterations)

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
        starting_point = mod(starting_point, 2 * pi)
        xs = mod(xs, 2 * pi)

        def ncdf(from_, to):
            return (
                1
                / 2
                * (
                    erf((self.mu - from_) / (sqrt(2) * self.sigma))
                    - erf((self.mu - to) / (sqrt(2) * self.sigma))
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
        return exp(1j * n * self.mu - n**2 * self.sigma**2 / 2)

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
            mod(self.mu + other.mu, 2.0 * pi), sqrt(self.C + other.C)
        )

    def sample(self, n: Union[int, int32, int64]):
        return mod(self.mu + self.sigma * random.normal(size=(n,)), 2.0 * pi)

    def to_dirac5(self):
        from .circular_dirac_distribution import CircularDiracDistribution

        offsets = array([-2.0, -1.0, 0.0, 1.0, 2.0])
        weights = array([1.0, 2.0, 6.0, 2.0, 1.0]) / 12.0
        samples = mod(self.mu + self.sigma * offsets, 2.0 * pi)
        return CircularDiracDistribution(samples, weights)

    def shift(self, shift_by):
        assert shift_by.shape in ((1,), ())
        return WrappedNormalDistribution(self.mu + shift_by, self.sigma)

    def to_vm(self) -> VonMisesDistribution:
        # Convert to Von Mises distribution
        kappa = self.sigma_to_kappa(self.sigma)
        return VonMisesDistribution(self.mu, kappa)

    @staticmethod
    def from_moment(m) -> "WrappedNormalDistribution":
        mu = mod(angle(m.squeeze()), 2.0 * pi)
        sigma = sqrt(-2 * log(abs(m.squeeze())))
        return WrappedNormalDistribution(mu, sigma)

    @staticmethod
    def sigma_to_kappa(sigma):
        # Approximate conversion from sigma to kappa for a Von Mises distribution
        return 1.0 / sigma**2
