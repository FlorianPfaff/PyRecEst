import unittest

import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, random
from pyrecest.distributions.circle.circular_dirac_distribution import CircularDiracDistribution
from pyrecest.distributions.circle.von_mises_distribution import VonMisesDistribution
from pyrecest.filters.circular_particle_filter import CircularParticleFilter


class CircularParticleFilterVonMisesUpdateTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Von Mises update regression uses the NumPy SciPy-backed pdf.",
    )
    def test_update_identity_accepts_von_mises_noise(self):
        random.seed(0)
        particle_filter = CircularParticleFilter(16)
        measurement_noise = VonMisesDistribution(array(0.0), array(2.0))

        particle_filter.update_identity(measurement_noise, array(1.0))

        self.assertIsInstance(particle_filter.filter_state, CircularDiracDistribution)
        self.assertEqual(particle_filter.filter_state.dim, 1)
        self.assertAlmostEqual(float(measurement_noise.mu), 0.0)


if __name__ == "__main__":
    unittest.main()
