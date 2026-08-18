import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, copy
from pyrecest.distributions import HypersphericalDiracDistribution
from pyrecest.filters.hyperspherical_particle_filter import HypersphericalParticleFilter


class TestHypersphericalParticleFilterStateAssignment(unittest.TestCase):
    def test_setting_dirac_state_does_not_alias_input_distribution(self):
        filt = HypersphericalParticleFilter(3, 3)
        state = HypersphericalDiracDistribution(
            array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            array([0.2, 0.3, 0.5]),
        )

        filt.filter_state = state
        assigned_particles = copy(filt.filter_state.d)
        assigned_weights = copy(filt.filter_state.w)

        self.assertIsNot(filt.filter_state, state)
        state.d = array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )
        state.w = array([0.5, 0.25, 0.25])

        self.assertTrue(bool(allclose(filt.filter_state.d, assigned_particles)))
        self.assertTrue(bool(allclose(filt.filter_state.w, assigned_weights)))

    def test_setting_dirac_state_rejects_particle_shape_change(self):
        filt = HypersphericalParticleFilter(3, 3)
        state = HypersphericalDiracDistribution(
            array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            )
        )

        with self.assertRaisesRegex(ValueError, "shape"):
            filt.filter_state = state


if __name__ == "__main__":
    unittest.main()
