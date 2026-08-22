import unittest

import pyrecest.backend
from pyrecest.backend import allclose, array
from pyrecest.distributions.hypersphere_subset.hyperspherical_mixture import (
    HypersphericalMixture,
)
from pyrecest.distributions.hypersphere_subset.von_mises_fisher_distribution import (
    VonMisesFisherDistribution,
)
from pyrecest.filters.hyperhemispherical_grid_filter import (
    HyperhemisphericalGridFilter,
)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
    reason="HyperhemisphericalGridFilter is not supported on the JAX backend",
)
class TestHyperhemisphericalGridFilterMixtureUpdate(unittest.TestCase):
    def setUp(self):
        self.pole = array([0.0, 0.0, 1.0])
        self.measurement = array([1.0, 0.0, 0.0])

    def _symmetric_vmf_mixture(self, kappa=3.0):
        return HypersphericalMixture(
            [
                VonMisesFisherDistribution(self.pole, kappa),
                VonMisesFisherDistribution(-self.pole, kappa),
            ],
            array([0.5, 0.5]),
        )

    def test_update_identity_does_not_mutate_measurement_noise(self):
        filt = HyperhemisphericalGridFilter(20, 2)
        meas_noise = self._symmetric_vmf_mixture()

        filt.update_identity(meas_noise, self.measurement)

        self.assertTrue(bool(allclose(meas_noise.dists[0].mu, self.pole)))
        self.assertTrue(bool(allclose(meas_noise.dists[1].mu, -self.pole)))

    def test_update_identity_rejects_unequal_vmf_concentrations(self):
        filt = HyperhemisphericalGridFilter(20, 2)
        meas_noise = HypersphericalMixture(
            [
                VonMisesFisherDistribution(self.pole, 2.0),
                VonMisesFisherDistribution(-self.pole, 3.0),
            ],
            array([0.5, 0.5]),
        )

        with self.assertRaisesRegex(ValueError, "UnsupportedNoise"):
            filt.update_identity(meas_noise, self.measurement)

    def test_update_identity_rejects_nonantipodal_vmf_components(self):
        filt = HyperhemisphericalGridFilter(20, 2)
        meas_noise = HypersphericalMixture(
            [
                VonMisesFisherDistribution(self.pole, 3.0),
                VonMisesFisherDistribution(array([1.0, 0.0, 0.0]), 3.0),
            ],
            array([0.5, 0.5]),
        )

        with self.assertRaisesRegex(ValueError, "UnsupportedNoise"):
            filt.update_identity(meas_noise, self.measurement)

    def test_update_identity_rejects_nonunit_measurement_for_mixture(self):
        filt = HyperhemisphericalGridFilter(20, 2)
        meas_noise = self._symmetric_vmf_mixture()

        with self.assertRaisesRegex(ValueError, "normalized"):
            filt.update_identity(meas_noise, array([2.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
