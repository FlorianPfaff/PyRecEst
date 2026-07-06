import unittest

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, diag, eye, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.dirichlet_process_clutter import DirichletProcessGaussianClutterIntensity
from pyrecest.filters.multi_bernoulli_tracker import BernoulliComponent, MultiBernoulliTracker


@unittest.skipIf(
    pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
    reason="The online DP clutter model is intended for numpy-backed RFS trackers.",
)
class DirichletProcessGaussianClutterIntensityTest(unittest.TestCase):
    def make_scalar_model(self, **kwargs):
        params = {
            "concentration": 0.2,
            "base_mean": array([0.0]),
            "base_covariance": array([[100.0]]),
            "kernel_covariance": array([[0.25]]),
            "clutter_rate": 1.0,
        }
        params.update(kwargs)
        return DirichletProcessGaussianClutterIntensity(**params)

    def test_predictive_intensity_learns_local_clutter_region(self):
        model = self.make_scalar_model()

        near_before = model(array([5.0]))
        model.observe(array([[5.0, 5.2, 4.9, 5.1]]))
        near_after = model(array([5.0]))
        far_after = model(array([20.0]))

        self.assertGreater(model.get_number_of_components(), 0)
        self.assertGreater(near_after, near_before)
        self.assertGreater(near_after, far_after)

    def test_zero_weight_observations_do_not_create_components(self):
        model = self.make_scalar_model()

        model.observe(array([[5.0]]), weights=array([0.0]))

        self.assertEqual(model.get_number_of_components(), 0)
        npt.assert_allclose(model.get_component_counts(), array([]))

    def test_intensity_for_measurements_returns_columnwise_values(self):
        model = self.make_scalar_model()
        model.observe(array([[0.0, 0.1, -0.1]]))

        intensities = model.intensity_for_measurements(array([[0.0, 8.0]]))

        self.assertEqual(intensities.shape, (2,))
        self.assertGreater(float(intensities[0]), float(intensities[1]))

    def test_initial_components_are_exposed_as_predictive_weights(self):
        model = self.make_scalar_model(
            initial_means=array([[2.0, 5.0]]),
            initial_counts=array([2.0, 3.0]),
        )

        npt.assert_allclose(model.get_component_means(), array([[2.0, 5.0]]))
        npt.assert_allclose(model.get_component_counts(), array([2.0, 3.0]))
        self.assertAlmostEqual(
            model.get_base_weight() + float(sum(model.get_component_weights())),
            1.0,
        )

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            self.make_scalar_model(concentration=0.0)
        with self.assertRaises(ValueError):
            self.make_scalar_model(kernel_covariance=array([[0.0]]))
        with self.assertRaises(ValueError):
            self.make_scalar_model(new_component_probability_threshold=1.5)
        with self.assertRaises(ValueError):
            self.make_scalar_model(initial_means=array([[1.0]]), initial_counts=array([0.0]))

    def test_learned_clutter_intensities_plug_into_bernoulli_update(self):
        learned_clutter = self.make_scalar_model(clutter_rate=25.0)
        learned_clutter.observe(array([[0.0, 0.05, -0.05]]))
        measurement = array([[0.0]])
        adaptive_clutter_intensity = learned_clutter.intensity_for_measurements(measurement)

        initial_state = GaussianDistribution(zeros(1), diag(array([1.0])))
        low_clutter_tracker = MultiBernoulliTracker(
            initial_prior=[BernoulliComponent(0.8, initial_state)],
            tracker_param={
                "detection_probability": 0.9,
                "clutter_intensity": 1e-12,
                "birth_covariance": None,
            },
        )
        adaptive_clutter_tracker = MultiBernoulliTracker(
            initial_prior=[BernoulliComponent(0.8, initial_state)],
            tracker_param={
                "detection_probability": 0.9,
                "clutter_intensity": adaptive_clutter_intensity,
                "birth_covariance": None,
            },
        )

        low_clutter_tracker.update_linear(measurement, eye(1), eye(1))
        adaptive_clutter_tracker.update_linear(measurement, eye(1), eye(1))

        self.assertGreater(
            low_clutter_tracker.bernoulli_components[0].existence_probability,
            adaptive_clutter_tracker.bernoulli_components[0].existence_probability,
        )


if __name__ == "__main__":
    unittest.main()
