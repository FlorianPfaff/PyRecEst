import numpy as np
import pytest
from pyrecest.distributions import SE2DiracDistribution
from pyrecest.distributions.cart_prod.abstract_hypercylindrical_distribution import (
    AbstractHypercylindricalDistribution,
)


class _NoSamplingSE2Distribution(AbstractHypercylindricalDistribution):
    def __init__(self):
        super().__init__(bound_dim=1, lin_dim=2)

    def pdf(self, xs):
        raise AssertionError("pdf must not be evaluated for count validation")

    def sample(self, n):
        raise AssertionError("invalid temporal counts must not reach sampling")

    def marginalize_linear(self):
        raise AssertionError(
            "marginalization must not be evaluated for count validation"
        )

    def marginalize_periodic(self):
        raise AssertionError(
            "marginalization must not be evaluated for count validation"
        )


@pytest.mark.parametrize(
    "n_particles",
    (
        np.timedelta64(3, "ns"),
        np.timedelta64(3, "us"),
    ),
)
def test_se2_dirac_from_distribution_rejects_temporal_particle_counts(n_particles):
    with pytest.raises(ValueError, match="positive integer"):
        SE2DiracDistribution.from_distribution(
            _NoSamplingSE2Distribution(), n_particles
        )
