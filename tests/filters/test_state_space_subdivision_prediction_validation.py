import copy
import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, eye
from pyrecest.distributions.cart_prod.state_space_subdivision_gaussian_distribution import (
    StateSpaceSubdivisionGaussianDistribution,
)
from pyrecest.distributions.circle.circular_uniform_distribution import (
    CircularUniformDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution import (
    HypertoroidalGridDistribution,
)
from pyrecest.distributions.nonperiodic.gaussian_distribution import (
    GaussianDistribution,
)
from pyrecest.filters.state_space_subdivision_filter import StateSpaceSubdivisionFilter


def _make_state(n_areas=3):
    grid = HypertoroidalGridDistribution.from_distribution(
        CircularUniformDistribution(), (n_areas,)
    )
    linear_distributions = [
        GaussianDistribution(array([1.0]), eye(1)) for _ in range(n_areas)
    ]
    return StateSpaceSubdivisionGaussianDistribution(grid, linear_distributions)


class TestStateSpaceSubdivisionPredictionValidation(unittest.TestCase):
    def test_rejects_underspecified_per_area_inputs_before_mutation(self):
        cases = (
            (
                "system_matrices",
                {"system_matrices": array([[[2.0, 3.0]]])},
            ),
            (
                "covariance_matrices",
                {"covariance_matrices": array([[[0.1, 0.2]]])},
            ),
            (
                "linear_input_vectors",
                {"linear_input_vectors": array([[4.0, 5.0]])},
            ),
        )

        for name, kwargs in cases:
            with self.subTest(name=name):
                filt = StateSpaceSubdivisionFilter(_make_state(n_areas=3))
                state_before = copy.deepcopy(filt.filter_state)

                with self.assertRaisesRegex(ValueError, name):
                    filt.predict_linear(**kwargs)

                for actual, expected in zip(
                    filt.filter_state.linear_distributions,
                    state_before.linear_distributions,
                    strict=True,
                ):
                    npt.assert_allclose(actual.mu, expected.mu)
                    npt.assert_allclose(actual.C, expected.C)

    def test_rejects_extra_per_area_slices(self):
        filt = StateSpaceSubdivisionFilter(_make_state(n_areas=3))
        too_many_system_matrices = array([[[1.0, 1.0, 1.0, 1.0]]])

        with self.assertRaisesRegex(ValueError, "system_matrices"):
            filt.predict_linear(system_matrices=too_many_system_matrices)


if __name__ == "__main__":
    unittest.main()
