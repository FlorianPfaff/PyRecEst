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
    """
    This class implements the wrapped normal distribution.
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
        AbstractCircularDistribution.__init__(self)
        HypertoroidalWrappedNormalDistribution.__init__(self, mu, sigma**2)

    @property
    def sigma(self):
        return sqrt(self.C)

    # pylint: disable=too-many-locals
    def pdf(self, xs):
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
        max_iterations: int = 1000
        if pyrecest.backend.__backend_name__ != "jax":
            n_inputs = xs.shape[0]
            result = zeros(n_inputs)

            tmp = -1.0 / (2.0 * self.sigma**2)
            nc = 1.0 / sqrt(2.0 * pi) / self.sigma

            for i in range(n_inputs):
                old_result = 0.0
                xi = x[i]
                if hasattr(xi, "item"):
                    xi = xi.item()
                result[i] = exp(xi * xi * tmp)

                for k in range(1, max_iterations + 1):
                    xp = xi + 2 * pi * k
                    xm = xi - 2 * pi * k
                    tp = xp * xp * tmp
                    tm = xm * xm * tmp
                    old_result = result[i]
                    addendum = exp(tp) + exp(tm)
                    if hasattr(addendum, "item"):
                        addendum = addendum.item()
                    result[i] += addendum

                    if result[i] == old_result:
                        break

                result[i] *= nc
        else:
            from jax import lax  # pylint: disable=import-error
            from jax.numpy import logical_and  # pylint: disable=import-error

            tmp = -1.0 / (2.0 * self.sigma**2)
            nc = 1.0 / (sqrt(2.0 * pi) * self.sigma)

            def body_fun(val):
                i, result = val
                xp = x + 2 * pi * i
                xm = x - 2 * pi * i
                tp = xp * xp * tmp
                tm = xm * xm * tmp
                addendum = exp(tp) + exp(tm)
                new_result = result + addendum
                return (i + 1, new_result)

            def cond_fun(val):
                i, result = val
                # Check both convergence and max_iterations
                return logical_and(
                    any(result - result.at[...].set(0) > 1e-10), i < max_iterations
                )

            initial_val = (
                1,
                exp(x * x * tmp),
            )  # Initial iteration index set to 1, and initial result based on x
            _, result = lax.while_loop(cond_fun, body_fun, initial_val)

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
        return self.multiply_vm(other)

    def multiply_vm(self, other):
        vm1 = self.to_vm()
        vm2 = other.to_vm()
        vm = vm1.multiply(vm2)
        wn = vm.to_wn()
        return wn

    def sample(self, n: Union[int, int32, int64]):
        return mod(self.mu + self.sigma * random.normal(size=(n,)), 2.0 * pi)

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
