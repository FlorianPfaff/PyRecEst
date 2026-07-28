import unittest

from pyrecest.backend import array, eye, linspace, ones, pi, zeros
from pyrecest.distributions.cart_prod.state_space_subdivision_gaussian_distribution import (
    StateSpaceSubdivisionGaussianDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution import (
    HypertoroidalGridDistribution,
)
from pyrecest.distributions.nonperiodic.gaussian_distribution import (
    GaussianDistribution,
)
from pyrecest.filters.relaxed_s3f_circular import predict_circular_relaxed
from pyrecest.filters.state_space_subdivision_filter import StateSpaceSubdivisionFilter


class RelaxedS3FProcessNoiseValidationTest(unittest.TestCase):
    def test_rejects_nonsymmetric_process_noise_without_mutating_state(self):
        filter_ = _make_filter(8)
        covariance_before = array(filter_.filter_state.linear_distributions[0].C)

        with self.assertRaisesRegex(ValueError, "positive-semidefinite"):
            predict_circular_relaxed(
                filter_,
                array([0.4, 0.1]),
                process_noise_cov=array([[1.0, 0.5], [0.0, 1.0]]),
            )

        self.assertTrue(
            bool(
                (
                    filter_.filter_state.linear_distributions[0].C == covariance_before
                ).all()
            )
        )

    def test_rejects_indefinite_process_noise_without_mutating_state(self):
        filter_ = _make_filter(8)
        covariance_before = array(filter_.filter_state.linear_distributions[0].C)

        with self.assertRaisesRegex(ValueError, "positive-semidefinite"):
            predict_circular_relaxed(
                filter_,
                array([0.4, 0.1]),
                process_noise_cov=array([[1.0, 0.0], [0.0, -0.1]]),
            )

        self.assertTrue(
            bool(
                (
                    filter_.filter_state.linear_distributions[0].C == covariance_before
                ).all()
            )
        )


def _make_filter(n_cells: int) -> StateSpaceSubdivisionFilter:
    grid = linspace(0.0, 2.0 * pi, n_cells, endpoint=False).reshape(-1, 1)
    gd = HypertoroidalGridDistribution(
        ones(n_cells) / (2.0 * pi),
        grid_type="custom",
        grid=grid,
    )
    gd.normalize_in_place(warn_unnorm=False)
    gaussians = [
        GaussianDistribution(
            zeros(2),
            eye(2) * 0.1,
            check_validity=False,
        )
        for _ in range(n_cells)
    ]
    state = StateSpaceSubdivisionGaussianDistribution(gd, gaussians)
    return StateSpaceSubdivisionFilter(state)


if __name__ == "__main__":
    unittest.main()
