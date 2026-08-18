# pylint: disable=no-name-in-module,no-member
from numbers import Integral

from pyrecest.backend import ones

from .abstract_se2_distribution import AbstractSE2Distribution
from .cart_prod.hypercylindrical_dirac_distribution import (
    HypercylindricalDiracDistribution,
)


class SE2DiracDistribution(HypercylindricalDiracDistribution, AbstractSE2Distribution):
    """Partially wrapped Dirac distribution on SE(2).

    Represents a distribution on SE(2) = S^1 x R^2 using weighted Dirac
    components. Each component d[i] = (angle, x, y) encodes a pose.

    References:
        Gerhard Kurz, Igor Gilitschenski, Uwe D. Hanebeck,
        The Partially Wrapped Normal Distribution for SE(2) Estimation,
        Proceedings of the 2014 IEEE International Conference on Multisensor
        Fusion and Information Integration (MFI 2014), Beijing, China,
        September 2014.
    """

    def __init__(self, d, w=None):
        """Initialize SE2DiracDistribution.

        Parameters
        ----------
        d : array of shape (n, 3)
            Dirac locations with columns [angle, x, y], where angle is in
            [0, 2*pi).
        w : array of shape (n,), optional
            Weights for each Dirac component. Defaults to uniform weights.
        """
        AbstractSE2Distribution.__init__(self)
        HypercylindricalDiracDistribution.__init__(self, bound_dim=1, d=d, w=w)

    def mean_4d(self):
        """Compute the 4D mean [E[cos(angle)], E[sin(angle)], E[x], E[y]].

        Returns
        -------
        array of shape (4,)
        """
        return self.hybrid_moment()

    def covariance_4d(self):
        """Compute the covariance of [cos(angle), sin(angle), x, y].

        Returns
        -------
        array of shape (4, 4)
        """
        from pyrecest.backend import (  # pylint: disable=import-outside-toplevel
            column_stack,
            cos,
            sin,
        )

        samples = column_stack(
            (cos(self.d[:, 0:1]), sin(self.d[:, 0:1]), self.d[:, 1:])
        )
        centered = samples - self.mean_4d()
        return centered.T @ (self.w[:, None] * centered)

    def mean(self):
        """Return the hybrid mean for a consistent interface.

        Returns
        -------
        array of shape (4,)
        """
        return self.hybrid_mean()

    @classmethod
    def from_distribution(cls, distribution, n_particles):
        """Create an SE2DiracDistribution by sampling from a given distribution.

        Parameters
        ----------
        distribution : AbstractHypercylindricalDistribution
            Source distribution on SE(2) (bound_dim=1, lin_dim=2) to sample
            from.
        n_particles : int
            Number of particles (Dirac components).

        Returns
        -------
        SE2DiracDistribution
        """
        from .cart_prod.abstract_hypercylindrical_distribution import (  # pylint: disable=import-outside-toplevel
            AbstractHypercylindricalDistribution,
        )

        if not isinstance(distribution, AbstractHypercylindricalDistribution):
            raise TypeError(
                "distribution must be an instance of AbstractHypercylindricalDistribution"
            )
        if distribution.bound_dim != 1 or distribution.lin_dim != 2:
            raise ValueError("distribution must have bound_dim=1 and lin_dim=2")
        n_particles = cls._validate_particle_count(n_particles)

        return cls(
            distribution.sample(n_particles),
            ones(n_particles) / n_particles,
        )

    @staticmethod
    def _validate_particle_count(n_particles):
        dtype_kind = getattr(getattr(n_particles, "dtype", None), "kind", None)
        if (
            isinstance(n_particles, bool)
            or dtype_kind in {"M", "m"}
            or not isinstance(n_particles, Integral)
            or int(n_particles) <= 0
        ):
            raise ValueError("n_particles must be a positive integer")
        return int(n_particles)
