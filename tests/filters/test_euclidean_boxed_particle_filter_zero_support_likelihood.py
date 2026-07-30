import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.backend import array, to_numpy
from pyrecest.distributions.nonperiodic.linear_dirac_distribution import (
    LinearDiracDistribution,
)
from pyrecest.filters.euclidean_boxed_particle_filter import (
    EuclideanBoxedParticleFilter,
)


class EuclideanBoxedParticleFilterZeroSupportLikelihoodTest(unittest.TestCase):
    def test_reweight_ignores_nonfinite_likelihood_on_zero_support_particles(self):
        pf = EuclideanBoxedParticleFilter(4, 1)
        pf.set_resampling_criterion(lambda _state: False)
        pf.filter_state = LinearDiracDistribution(
            array([[0.0], [1.0], [2.0], [3.0]]),
            array([0.2, 0.3, 0.0, 0.5]),
        )

        pf.reweight_by_box(
            array([0.0]),
            array([2.0]),
            likelihood=lambda _particles: array([1.0, 2.0, np.nan, np.nan]),
        )

        npt.assert_allclose(
            to_numpy(pf.filter_state.w),
            [0.25, 0.75, 0.0, 0.0],
        )


if __name__ == "__main__":
    unittest.main()
