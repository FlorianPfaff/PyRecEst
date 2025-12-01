import unittest

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, pi, random
from pyrecest.distributions.hypertorus.hypertoroidal_wrapped_normal_distribution import (
    HypertoroidalWrappedNormalDistribution,
)
from pyrecest.distributions.hypertorus.toroidal_wrapped_normal_distribution import (
    ToroidalWrappedNormalDistribution,
)
from pyrecest.filters.toroidal_particle_filter import ToroidalParticleFilter


class ToroidalParticleFilterTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_toroidal_particle_filter(self):
        random.seed(0)
        C = array([[0.7, 0.4], [0.4, 0.6]])
        mu = array([1.0, 1.0]) + pi / 2.0
        hwnd = ToroidalWrappedNormalDistribution(mu, C)
        tpf = ToroidalParticleFilter(200)
        tpf.filter_state = hwnd
        forced_mean = array([1.0, 1.0])

        for _ in range(50):
            tpf.predict_identity(
                HypertoroidalWrappedNormalDistribution(array([0.0, 0.0]), C)
            )
            for _ in range(3):
                tpf.update_identity(
                    HypertoroidalWrappedNormalDistribution(array([0.0, 0.0]), 0.5 * C),
                    forced_mean,
                )

        npt.assert_allclose(tpf.get_point_estimate(), forced_mean, atol=0.2)


if __name__ == "__main__":
    unittest.main()
