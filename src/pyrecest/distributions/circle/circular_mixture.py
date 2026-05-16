import collections
import warnings

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import asarray, atleast_1d, ndim, reshape, shape, sum, zeros

from ..hypertorus.hypertoroidal_mixture import HypertoroidalMixture
from .abstract_circular_distribution import AbstractCircularDistribution
from .circular_dirac_distribution import CircularDiracDistribution
from .circular_fourier_distribution import CircularFourierDistribution


class CircularMixture(AbstractCircularDistribution, HypertoroidalMixture):
    def __init__(
        self,
        dists: collections.abc.Sequence[AbstractCircularDistribution],
        w,
    ):
        """
        Creates a new circular mixture.

        Args:
            dists: The list of distributions.
            w: The weights of the distributions. They must have the same shape as 'dists'
                and the sum of all weights must be 1.
        """
        HypertoroidalMixture.__init__(self, dists, w)
        AbstractCircularDistribution.__init__(self)
        if not all(isinstance(cd, AbstractCircularDistribution) for cd in dists):
            raise TypeError(
                "All elements of 'dists' must be of type AbstractCircularDistribution."
            )

        if shape(dists) != shape(w):
            raise ValueError("'dists' and 'w' must have the same shape.")

        if all(isinstance(cd, CircularFourierDistribution) for cd in dists):
            warnings.warn(
                "Warning: Mixtures of Fourier distributions can be built by combining the Fourier coefficients so using a mixture may not be necessary"
            )
        elif all(isinstance(cd, CircularDiracDistribution) for cd in dists):
            warnings.warn(
                "Warning: Mixtures of WDDistributions can usually be combined into one WDDistribution."
            )

        self.dists = dists
        self.w = w / sum(w)

    def pdf(self, xs):
        """Evaluate the circular-mixture density.

        Circular distributions in this package commonly accept a one-dimensional
        array of angles with shape ``(n,)``. The generic mixture implementation
        expects the last axis to encode the manifold dimension, which rejects
        such arrays for one-dimensional circular distributions and can broadcast
        incorrectly for ``(n, 1)`` column vectors.
        """
        xs = asarray(xs)
        xs_ndim = ndim(xs)

        if xs_ndim == 0:
            xs_eval = xs
            scalar_input = True
        elif xs_ndim == 1:
            xs_eval = xs
            scalar_input = False
        elif xs_ndim == 2 and shape(xs)[-1] == 1:
            xs_eval = reshape(xs, (-1,))
            scalar_input = False
        else:
            raise AssertionError("Dimension mismatch")

        p = zeros(shape(atleast_1d(xs_eval)))

        for i, dist in enumerate(self.dists):
            component_pdf = reshape(atleast_1d(dist.pdf(xs_eval)), shape(p))
            p += self.w[i] * component_pdf

        if scalar_input:
            return p[0]
        return p
