import matplotlib.pyplot as plt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, atleast_1d, cos, linspace, log, mod, pi, sin

from ..hypertorus.abstract_hypertoroidal_distribution import (
    AbstractHypertoroidalDistribution,
)


class AbstractCircularDistribution(AbstractHypertoroidalDistribution):
    def __init__(self):
        AbstractHypertoroidalDistribution.__init__(self, dim=1)

    def cdf_numerical(self, xs, starting_point: float = 0.0):
        """
        Calculates the cumulative distribution function.

        Args:
            xs (): The 1D array to calculate the CDF on.
            starting_point (float, optional): Defaults to 0.

        Returns:
            : The computed CDF as a numpy array.
        """
        xs = atleast_1d(array(xs))
        if xs.ndim != 1:
            raise ValueError("xs must be a 1D array.")

        return array([self._cdf_numerical_single(x, starting_point) for x in xs])

    def _cdf_numerical_single(
        self,
        x,
        starting_point,
    ):
        """Helper method for cdf_numerical"""
        starting_point_mod = mod(starting_point, 2.0 * pi)
        x_mod = mod(x, 2.0 * pi)

        if x_mod < starting_point_mod:
            return 1.0 - self.integrate_numerically(array([x_mod, starting_point_mod]))

        return self.integrate_numerically(array([starting_point_mod, x_mod]))

    def kld_numerical(self, other):
        """
        Calculates the Kullback-Leibler divergence numerically.

        Args:
            other (AbstractCircularDistribution): Distribution to compare against.

        Returns:
            : The Kullback-Leibler divergence D_KL(self || other).
        """
        if not isinstance(other, AbstractCircularDistribution):
            raise TypeError("other must be an AbstractCircularDistribution.")
        if self.dim != other.dim:
            raise ValueError(
                "Cannot compare distributions with different number of dimensions."
            )

        def kld_fun(*args):
            x = array(args)
            pdf_self = self.pdf(x)
            if pdf_self <= 0.0:
                return 0.0 * pdf_self
            return pdf_self * log(pdf_self / other.pdf(x))

        return self.integrate_fun_over_domain(kld_fun, self.dim)

    def to_vm(self):
        """
        Convert to von Mises by trigonometric moment matching.

        Returns:
            vm (VMDistribution): Distribution with the same first trigonometric moment.
        """
        from .von_mises_distribution import VonMisesDistribution

        vm = VonMisesDistribution.from_moment(self.trigonometric_moment(1))
        return vm

    def to_wn(self):
        """
        Convert to wrapped normal by trigonometric moment matching.

        Returns:
            wn (WrappedNormalDistribution): Distribution with the same first trigonometric moment.
        """
        from .wrapped_normal_distribution import WrappedNormalDistribution

        wn = WrappedNormalDistribution.from_moment(self.trigonometric_moment(1))
        return wn

    @staticmethod
    def plot_circle(*args, **kwargs):
        theta = linspace(0.0, 2.0 * pi, 320)
        p = plt.plot(cos(theta), sin(theta), *args, **kwargs)
        return p
