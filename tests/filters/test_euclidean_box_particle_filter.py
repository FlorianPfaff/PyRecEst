import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, ones, random
from pyrecest.distributions.nonperiodic.linear_box_particle_distribution import (
    LinearBoxParticleDistribution,
)
from pyrecest.filters.euclidean_box_particle_filter import EuclideanBoxParticleFilter


class EuclideanBoxParticleFilterTest(unittest.TestCase):
    def test_identity_box_update_uses_contracted_volume_ratio(self):
        pf = EuclideanBoxParticleFilter(2, 1)
        pf.filter_state = LinearBoxParticleDistribution(
            array([[0.0], [2.0]]),
            array([[2.0], [4.0]]),
            array([0.5, 0.5]),
        )
        pf.set_resampling_criterion(lambda _state: False)

        pf.update_identity_box(array([0.0]), array([1.0]))

        npt.assert_allclose(pf.filter_state.lower, array([[0.0], [2.0]]))
        npt.assert_allclose(pf.filter_state.upper, array([[1.0], [4.0]]))
        npt.assert_allclose(pf.filter_state.w, array([1.0, 0.0]))

    def test_resample_splits_duplicate_boxes_along_widest_dimension(self):
        random.seed(1)
        pf = EuclideanBoxParticleFilter(4, 1)
        pf.filter_state = LinearBoxParticleDistribution(
            array([[0.0], [10.0], [20.0], [30.0]]),
            array([[4.0], [11.0], [21.0], [31.0]]),
            array([1.0, 0.0, 0.0, 0.0]),
        )

        pf.resample()

        npt.assert_allclose(pf.filter_state.lower, array([[0.0], [1.0], [2.0], [3.0]]))
        npt.assert_allclose(pf.filter_state.upper, array([[1.0], [2.0], [3.0], [4.0]]))
        npt.assert_allclose(pf.filter_state.w, ones(4) / 4)

    def test_predict_interval_applies_bounded_additive_noise(self):
        pf = EuclideanBoxParticleFilter(1, 1)
        pf.filter_state = LinearBoxParticleDistribution(array([[1.0]]), array([[2.0]]))

        pf.predict_interval(
            lambda lower, upper: (2.0 * lower, 2.0 * upper),
            process_noise_bounds=(array([-0.5]), array([0.25])),
        )

        npt.assert_allclose(pf.filter_state.lower, array([[1.5]]))
        npt.assert_allclose(pf.filter_state.upper, array([[4.25]]))


if __name__ == "__main__":
    unittest.main()
