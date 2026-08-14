import unittest

import numpy as np
import numpy.testing as npt
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


def _make_state(n_grid_points: int = 5) -> StateSpaceSubdivisionGaussianDistribution:
    grid_distribution = HypertoroidalGridDistribution.from_distribution(
        CircularUniformDistribution(),
        (n_grid_points,),
    )
    linear_distributions = [
        GaussianDistribution(array([0.0]), eye(1)) for _ in range(n_grid_points)
    ]
    return StateSpaceSubdivisionGaussianDistribution(
        grid_distribution,
        linear_distributions,
    )


class TestStateSpaceSubdivisionUpdateValidation(unittest.TestCase):
    def test_rejects_wrong_periodic_likelihood_size_without_mutation(self):
        filter_instance = StateSpaceSubdivisionFilter(_make_state())
        grid_before = np.asarray(
            filter_instance.filter_state.gd.grid_values,
            dtype=float,
        ).copy()

        with self.assertRaisesRegex(ValueError, "one value per grid point"):
            filter_instance.update(likelihood_periodic_grid=array([2.0]))

        npt.assert_allclose(
            np.asarray(filter_instance.filter_state.gd.grid_values, dtype=float),
            grid_before,
        )

    def test_invalid_linear_count_does_not_apply_periodic_likelihood(self):
        n_grid_points = 5
        filter_instance = StateSpaceSubdivisionFilter(_make_state(n_grid_points))
        state = filter_instance.filter_state
        grid_before = np.asarray(state.gd.grid_values, dtype=float).copy()
        means_before = [
            np.asarray(distribution.mu, dtype=float).copy()
            for distribution in state.linear_distributions
        ]
        covariances_before = [
            np.asarray(distribution.C, dtype=float).copy()
            for distribution in state.linear_distributions
        ]
        invalid_linear_likelihoods = [
            GaussianDistribution(array([1.0]), eye(1)),
            GaussianDistribution(array([2.0]), eye(1)),
        ]

        with self.assertRaisesRegex(ValueError, "1 or n_areas"):
            filter_instance.update(
                likelihood_periodic_grid=array(np.linspace(1.0, 2.0, n_grid_points)),
                likelihoods_linear=invalid_linear_likelihoods,
            )

        npt.assert_allclose(
            np.asarray(state.gd.grid_values, dtype=float),
            grid_before,
        )
        for distribution, mean_before, covariance_before in zip(
            state.linear_distributions,
            means_before,
            covariances_before,
            strict=True,
        ):
            npt.assert_allclose(
                np.asarray(distribution.mu, dtype=float),
                mean_before,
            )
            npt.assert_allclose(
                np.asarray(distribution.C, dtype=float),
                covariance_before,
            )

    def test_column_periodic_likelihood_matches_flat_input(self):
        n_grid_points = 5
        flat_filter = StateSpaceSubdivisionFilter(_make_state(n_grid_points))
        column_filter = StateSpaceSubdivisionFilter(_make_state(n_grid_points))
        likelihood_values = np.linspace(0.5, 1.5, n_grid_points)

        flat_filter.update(likelihood_periodic_grid=array(likelihood_values))
        column_filter.update(
            likelihood_periodic_grid=array(likelihood_values.reshape(-1, 1))
        )

        npt.assert_allclose(
            np.asarray(flat_filter.filter_state.gd.grid_values, dtype=float),
            np.asarray(column_filter.filter_state.gd.grid_values, dtype=float),
        )


if __name__ == "__main__":
    unittest.main()
