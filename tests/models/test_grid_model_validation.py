"""Regression tests for reusable grid-model callback validation."""

import unittest

from pyrecest.models.grid import (
    GridLikelihoodMeasurementModel,
    GridTransitionDensityFactoryModel,
    GridTransitionDensityModel,
)


class TestGridModelValidation(unittest.TestCase):
    def test_likelihood_model_rejects_noncallable_likelihood(self):
        with self.assertRaisesRegex(TypeError, "likelihood must be callable"):
            GridLikelihoodMeasurementModel(likelihood=object())

    def test_transition_density_factory_model_rejects_noncallable_factory(self):
        with self.assertRaisesRegex(
            TypeError,
            "transition_density_factory must be callable",
        ):
            GridTransitionDensityFactoryModel(transition_density_factory=object())

    def test_valid_grid_model_callbacks_still_evaluate(self):
        likelihood_model = GridLikelihoodMeasurementModel(
            lambda measurement, grid: (measurement, grid)
        )
        self.assertEqual(likelihood_model.likelihood_values("z", "grid"), ("z", "grid"))

        transition_density = object()
        density_model = GridTransitionDensityModel(transition_density)
        self.assertIs(
            density_model.transition_density_for_filter(object()), transition_density
        )

        filter_instance = object()
        factory_model = GridTransitionDensityFactoryModel(
            lambda filter_arg: ("density", filter_arg)
        )
        self.assertEqual(
            factory_model.transition_density_for_filter(filter_instance),
            ("density", filter_instance),
        )


if __name__ == "__main__":
    unittest.main()
