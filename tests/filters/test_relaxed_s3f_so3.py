# pylint: disable=no-value-for-parameter
import unittest

from pyrecest.backend import allclose, array, eye, linalg, ones, zeros
from pyrecest.distributions.cart_prod.state_space_subdivision_gaussian_distribution import (
    StateSpaceSubdivisionGaussianDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_grid_distribution import (
    HyperhemisphericalGridDistribution,
)
from pyrecest.distributions.nonperiodic.gaussian_distribution import GaussianDistribution
from pyrecest.filters.relaxed_s3f_so3 import (
    _cached_s3r3_cell_statistics,
    predict_s3r3_relaxed,
    rotate_quaternion_body_increment,
    s3r3_cell_statistics,
    s3r3_orientation_distance,
)
from pyrecest.filters.state_space_subdivision_filter import StateSpaceSubdivisionFilter


class RelaxedS3FSO3Test(unittest.TestCase):
    def test_covariance_inflation_is_positive_semidefinite(self):
        stats = s3r3_cell_statistics(
            _small_quaternion_grid(),
            array([0.4, 0.1, 0.2]),
            cell_sample_count=27,
        )

        self.assertEqual(stats.representative_displacements.shape, (4, 3))
        self.assertEqual(stats.mean_displacements.shape, (4, 3))
        self.assertEqual(stats.covariance_inflations.shape, (4, 3, 3))
        self.assertEqual(stats.method, "local_tangent_samples")
        for cov_matrix in stats.covariance_inflations:
            self.assertTrue(bool(allclose(cov_matrix, cov_matrix.T, atol=1e-12)))
            eigvals = linalg.eigvalsh(cov_matrix)
            self.assertGreaterEqual(float(eigvals[0]), -1e-12)

    def test_cell_statistics_reuses_identical_grid_cache(self):
        _cached_s3r3_cell_statistics.cache_clear()
        grid = _small_quaternion_grid()

        stats_a = s3r3_cell_statistics(grid, array([0.4, 0.1, 0.2]), cell_sample_count=27)
        stats_b = s3r3_cell_statistics(array(grid, dtype=float), array([0.4, 0.1, 0.2]), cell_sample_count=27)

        self.assertIs(stats_a, stats_b)
        cache_info = _cached_s3r3_cell_statistics.cache_info()
        self.assertEqual(cache_info.misses, 1)
        self.assertEqual(cache_info.hits, 1)

    def test_prediction_conserves_grid_mass_and_inflates_covariance(self):
        filter_ = _make_filter()
        values_before = array(filter_.filter_state.gd.grid_values, dtype=float)

        stats = predict_s3r3_relaxed(
            filter_,
            array([0.4, 0.1, 0.2]),
            variant="r1_r2",
            process_noise_cov=eye(3) * 0.01,
            cell_sample_count=27,
        )

        self.assertTrue(bool(allclose(filter_.filter_state.gd.grid_values, values_before, atol=1e-12)))
        self.assertTrue(bool(allclose(float(filter_.filter_state.gd.integrate()), 1.0, atol=1e-12)))
        self.assertTrue(any(float(linalg.norm(covariance)) > 0.0 for covariance in stats.covariance_inflations))

    def test_baseline_and_r1_variants_use_expected_inputs(self):
        body_increment = array([0.4, 0.0, 0.0])
        baseline_filter = _make_filter()
        r1_filter = _make_filter()

        baseline_stats = predict_s3r3_relaxed(
            baseline_filter,
            body_increment,
            variant="baseline",
            cell_sample_count=27,
        )
        r1_stats = predict_s3r3_relaxed(
            r1_filter,
            body_increment,
            variant="r1",
            cell_sample_count=27,
        )

        baseline_mean0 = baseline_filter.filter_state.linear_distributions[0].mu
        r1_mean0 = r1_filter.filter_state.linear_distributions[0].mu
        self.assertTrue(bool(allclose(baseline_mean0, baseline_stats.representative_displacements[0], atol=1e-12)))
        self.assertTrue(bool(allclose(r1_mean0, r1_stats.mean_displacements[0], atol=1e-12)))
        self.assertFalse(bool(allclose(baseline_mean0, r1_mean0, atol=1e-14)))

    def test_quaternion_rotation_and_distance_helpers(self):
        identity = array([0.0, 0.0, 0.0, 1.0])
        half_turn_z = array([0.0, 0.0, 1.0, 0.0])

        rotated = rotate_quaternion_body_increment(identity, array([0.4, 0.1, 0.2]))

        self.assertTrue(bool(allclose(rotated[0], array([0.4, 0.1, 0.2]), atol=1e-12)))
        self.assertAlmostEqual(s3r3_orientation_distance(identity, -identity), 0.0, places=12)
        self.assertAlmostEqual(s3r3_orientation_distance(identity, half_turn_z), 3.141592653589793, places=12)

    def test_validation_errors_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "method"):
            s3r3_cell_statistics(_small_quaternion_grid(), array([0.4, 0.1, 0.2]), method="exact_voronoi")
        with self.assertRaisesRegex(ValueError, "cell_sample_count"):
            s3r3_cell_statistics(_small_quaternion_grid(), array([0.4, 0.1, 0.2]), cell_sample_count=0)
        with self.assertRaisesRegex(ValueError, "body_increment"):
            s3r3_cell_statistics(_small_quaternion_grid(), array([0.4, 0.1]), cell_sample_count=27)
        with self.assertRaisesRegex(ValueError, "variant"):
            predict_s3r3_relaxed(_make_filter(), array([0.4, 0.1, 0.2]), variant="r2_only")


def _make_filter() -> StateSpaceSubdivisionFilter:
    grid = _small_quaternion_grid()
    gd = HyperhemisphericalGridDistribution(
        grid,
        ones(grid.shape[0]),
        enforce_pdf_nonnegative=True,
    )
    gd.normalize_in_place(warn_unnorm=False)
    gaussians = [
        GaussianDistribution(
            zeros(3),
            eye(3) * 0.1,
            check_validity=False,
        )
        for _ in range(grid.shape[0])
    ]
    state = StateSpaceSubdivisionGaussianDistribution(gd, gaussians)
    return StateSpaceSubdivisionFilter(state)


def _small_quaternion_grid():
    return array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.5, 0.8660254037844386],
            [0.0, 0.5, 0.0, 0.8660254037844386],
            [0.5, 0.0, 0.0, 0.8660254037844386],
        ],
        dtype=float,
    )


if __name__ == "__main__":
    unittest.main()
