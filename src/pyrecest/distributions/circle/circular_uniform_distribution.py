from pyrecest.backend import array, mod, pi, prod

from ..hypertorus._input_validation import as_shift_vector
from ..hypertorus.hypertoroidal_uniform_distribution import (
    HypertoroidalUniformDistribution,
    _validate_boundary,
)
from .abstract_circular_distribution import AbstractCircularDistribution


class CircularUniformDistribution(
    HypertoroidalUniformDistribution, AbstractCircularDistribution
):
    """
    Circular uniform distribution
    """

    def __init__(self):
        HypertoroidalUniformDistribution.__init__(self, 1)
        AbstractCircularDistribution.__init__(self)

    def get_manifold_size(self):
        return AbstractCircularDistribution.get_manifold_size(self)

    def shift(self, shift_by):
        as_shift_vector(shift_by, self.dim)
        return CircularUniformDistribution()

    def integrate(self, integration_boundaries=None) -> float:
        """Integrate over an oriented angular interval.

        Circular integration historically accepts decreasing boundaries and
        returns the corresponding signed integral, matching
        :meth:`integrate_numerically`.  The generic hypertoroidal uniform
        distribution rejects reversed boundaries because they would otherwise
        describe an invalid rectangular volume; keep the circular contract in
        this subclass rather than weakening that generic validation.
        """
        if integration_boundaries is None:
            return 1.0

        left, right = integration_boundaries
        left = _validate_boundary("left", left, self.dim)
        right = _validate_boundary("right", right, self.dim)
        return prod(right - left) / (2.0 * pi)

    def cdf(self, xa, starting_point=0):
        """
        Evaluate cumulative distribution function

        Parameters
        ----------
        xa : (1, n)
            points where the cdf should be evaluated
        starting_point : scalar
            point where the cdf is zero (starting point can be
            [0, 2pi) on the circle, default 0

        Returns
        -------
        val : (1, n)
            cdf evaluated at columns of xa
        """

        xa = array(xa)
        return mod(xa - starting_point, 2.0 * pi) / (2.0 * pi)
