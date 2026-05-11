import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, random, repeat, vstack
from pyrecest.evaluation.eot_shape_database import Cross, Star
from pyrecest.utils.metrics import (
    anees,
    anis,
    consistency_fraction,
    eot_shape_iou,
    extent_error,
    extent_matrix_error,
    extent_wasserstein_distance,
    gaussian_wasserstein_distance,
    gospa_distance,
    iou_polygon,
    is_chi_square_consistent,
    mae,
    mospa_distance,
    mse,
    nees,
    nees_confidence_bounds,
    nis,
    nis_confidence_bounds,
    ospa_distance,
    rmse,
)


class TestPointEstimateMetrics(unittest.TestCase):
    def test_mse_rmse_mae(self):
        estimates = array([[1.0, 2.0], [3.0, 4.0]])
        groundtruths = array([[1.0, 0.0], [1.0, 4.0]])

        self.assertEqual(mse(estimates, groundtruths), 2.0)
        npt.assert_allclose(rmse(estimates, groundtruths), np.sqrt(2.0))
        self.assertEqual(mae(estimates, groundtruths), 1.0)
        npt.assert_allclose(mse(estimates, groundtruths, axis=0), np.array([2.0, 2.0]))
        npt.assert_allclose(mae(estimates, groundtruths, axis=0), np.array([1.0, 1.0]))


class TestANEES(unittest.TestCase):
    def setUp(self):
        self.groundtruths = array([[1.5, 2.5], [2.5, 3.5], [4.5, 5.5]])
        self.uncertainties = array(
            [
                [[1.0, 0.5], [0.5, 2.0]],
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.5, 0.0], [0.0, 1.5]],
            ]
        )
        self.n_timesteps_constant = 10000

    def test_ANEES_is_close_to_one(self):
        """Test that the ANEES is close to 1 when we sample from the groundtruths with the given uncertainties.
        Simulate that the state stays constant for 10000 time steps, then changes, stays constant for another 10000 time steps
        and then changes once more before staying constant for the remaining 10000 time steps.
        """
        samples = []

        for i in range(len(self.groundtruths)):
            samples_for_i = random.multivariate_normal(
                mean=self.groundtruths[i],
                cov=self.uncertainties[i],
                size=self.n_timesteps_constant,
            )
            samples.append(samples_for_i)

        samples_mat = vstack(samples)

        repeated_groundtruths = repeat(
            self.groundtruths, repeats=self.n_timesteps_constant, axis=0
        )
        repeated_uncertainties = repeat(
            self.uncertainties, repeats=self.n_timesteps_constant, axis=0
        )

        computed_ANEES = anees(
            samples_mat, repeated_uncertainties, repeated_groundtruths
        )

        # Assert that computed ANEES is close to 1 with a tolerance of 0.05.
        npt.assert_allclose(computed_ANEES, self.groundtruths.shape[-1], atol=0.05)

    def test_nees_nis_and_bounds(self):
        estimates = array([[1.0, 1.0], [2.0, 0.0]])
        groundtruths = array([[0.0, 0.0], [0.0, 0.0]])
        covariances = array([np.eye(2), np.eye(2)])

        npt.assert_allclose(
            nees(estimates, covariances, groundtruths), np.array([2.0, 4.0])
        )
        self.assertEqual(anees(estimates, covariances, groundtruths), 3.0)

        innovations = array([[1.0, 0.0], [0.0, 2.0]])
        innovation_covariances = array([np.eye(2), 2.0 * np.eye(2)])
        npt.assert_allclose(
            nis(innovations, innovation_covariances), np.array([1.0, 2.0])
        )
        npt.assert_allclose(
            nis(innovations, groundtruths, innovation_covariances), np.array([1.0, 2.0])
        )
        self.assertEqual(anis(innovations, innovation_covariances), 1.5)

        lower, upper = nees_confidence_bounds(2, n_samples=2)
        self.assertLess(lower, 2.0)
        self.assertGreater(upper, 2.0)
        self.assertTrue(is_chi_square_consistent(3.0, 2, n_samples=2))
        nis_lower, nis_upper = nis_confidence_bounds(2, n_samples=2)
        self.assertEqual((lower, upper), (nis_lower, nis_upper))
        self.assertEqual(
            consistency_fraction([lower - 1.0, 2.0, upper + 1.0], lower, upper),
            1.0 / 3.0,
        )


class TestSetDistances(unittest.TestCase):
    def test_ospa_distance(self):
        estimates = array([[0.0], [2.0]])
        groundtruths = array([[0.0], [3.0]])
        npt.assert_allclose(
            ospa_distance(estimates, groundtruths, cutoff=10.0, order=2.0), np.sqrt(0.5)
        )

        cardinality_case = ospa_distance(
            array([[0.0]]),
            array([[0.0], [3.0]]),
            cutoff=2.0,
            order=1.0,
            return_components=True,
        )
        self.assertEqual(cardinality_case["ospa"], 1.0)
        self.assertEqual(cardinality_case["localization"], 0.0)
        self.assertEqual(cardinality_case["cardinality"], 1.0)
        self.assertEqual(cardinality_case["assignments"], 1)

    def test_mospa_distance(self):
        estimated_sets = [array([[0.0]]), array([[1.0]])]
        groundtruth_sets = [array([[0.0]]), array([[3.0]])]
        self.assertEqual(
            mospa_distance(estimated_sets, groundtruth_sets, cutoff=10.0, order=1.0),
            1.0,
        )
        mean_distance, per_step = mospa_distance(
            estimated_sets,
            groundtruth_sets,
            cutoff=10.0,
            order=1.0,
            return_per_step=True,
        )
        self.assertEqual(mean_distance, 1.0)
        npt.assert_allclose(per_step, np.array([0.0, 2.0]))

    def test_gospa_distance(self):
        result = gospa_distance(
            array([[0.0]]),
            array([[0.0], [4.0]]),
            cutoff=2.0,
            order=1.0,
            alpha=2.0,
            return_components=True,
        )
        self.assertEqual(result["gospa"], 1.0)
        self.assertEqual(result["localization"], 0.0)
        self.assertEqual(result["cardinality"], 1.0)
        self.assertEqual(result["assignments"], 1)


class TestEOTMetrics(unittest.TestCase):
    def test_gaussian_wasserstein_and_extent_error(self):
        mean = array([0.0, 0.0])
        cov = array(np.eye(2))
        self.assertEqual(gaussian_wasserstein_distance(mean, cov, mean, cov), 0.0)
        self.assertEqual(extent_wasserstein_distance(cov, cov), 0.0)
        npt.assert_allclose(
            gaussian_wasserstein_distance(mean, cov, array([1.0, 0.0]), cov), 1.0
        )
        npt.assert_allclose(extent_matrix_error(cov, 2.0 * cov), np.sqrt(2.0))
        npt.assert_allclose(extent_error(cov, 2.0 * cov, relative=True), 0.5)

    def test_iou_aliases(self):
        cross = Cross(2.0, 1.0, 2.0, 3.0)
        self.assertGreater(iou_polygon(cross, Star(0.5)), 0.05)
        self.assertGreater(iou_polygon(cross, Star(1.0)), iou_polygon(cross, Star(0.5)))
        self.assertEqual(iou_polygon(cross, cross), 1.0)
        self.assertEqual(eot_shape_iou(cross, cross), 1.0)


class TestIoU(unittest.TestCase):
    def test_iou_plolygon(self):
        cross = Cross(2.0, 1.0, 2.0, 3.0)
        self.assertGreater(iou_polygon(cross, Star(0.5)), 0.05)
        self.assertGreater(iou_polygon(cross, Star(1.0)), iou_polygon(cross, Star(0.5)))
        self.assertEqual(iou_polygon(cross, cross), 1.0)


if __name__ == "__main__":
    unittest.main()
