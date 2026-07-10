"""Regression coverage for vMF hyperhemispherical grid updates."""

import math
import unittest

import pyrecest.backend
from pyrecest.backend import array, linalg
from pyrecest.distributions import VonMisesFisherDistribution
from pyrecest.filters import HyperhemisphericalGridFilter


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
    reason="Not supported on JAX backend",
)
class TestHyperhemisphericalGridFilterVmfUpdate(unittest.TestCase):
    def test_accepts_numerically_equatorial_vmf_measurement(self):
        filter_ = HyperhemisphericalGridFilter(50, 2)
        equator_residual = 1e-12
        measurement = array(
            [math.sqrt(1.0 - equator_residual**2), 0.0, equator_residual]
        )
        standard_pole = array([0.0, 0.0, 1.0])
        measurement_noise = VonMisesFisherDistribution(standard_pole, 3.0)

        filter_.update_identity(measurement_noise, measurement)

        estimate = filter_.get_point_estimate()
        self.assertAlmostEqual(float(linalg.norm(estimate)), 1.0, places=5)
        self.assertGreater(abs(float(estimate[0])), 0.9)

    def test_rejects_vmf_measurement_outside_equator_tolerance(self):
        filter_ = HyperhemisphericalGridFilter(50, 2)
        off_equator = 1e-6
        measurement = array([math.sqrt(1.0 - off_equator**2), 0.0, off_equator])
        measurement_noise = VonMisesFisherDistribution(array([0.0, 0.0, 1.0]), 3.0)

        with self.assertRaisesRegex(ValueError, "unsupported measurement noise"):
            filter_.update_identity(measurement_noise, measurement)

    def test_rejects_non_zonal_vmf_noise(self):
        filter_ = HyperhemisphericalGridFilter(50, 2)
        equatorial_measurement = array([1.0, 0.0, 0.0])
        non_zonal_noise = VonMisesFisherDistribution(array([1.0, 0.0, 0.0]), 3.0)

        with self.assertRaisesRegex(ValueError, "mu needs to be"):
            filter_.update_identity(non_zonal_noise, equatorial_measurement)


if __name__ == "__main__":
    unittest.main()
