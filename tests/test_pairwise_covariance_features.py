import math
import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member,redefined-builtin
from pyrecest.backend import array, exp, log, zeros
from pyrecest.utils import (
    pairwise_covariance_shape_components,
    pairwise_mahalanobis_distances,
)


class TestPairwiseCovarianceFeatures(unittest.TestCase):
    def test_pairwise_mahalanobis_distances_use_summed_covariances(self):
        means_a = array([[0.0], [0.0]])
        means_b = array([[2.0, 0.0], [0.0, 2.0]])
        covariances_a = array([[[1.0], [0.0]], [[0.0], [1.0]]])
        covariances_b = array(
            [
                [[3.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 3.0]],
            ]
        )

        distances = pairwise_mahalanobis_distances(
            means_a,
            covariances_a,
            means_b,
            covariances_b,
        )

        self.assertEqual(distances.shape, (1, 2))
        npt.assert_allclose(distances, array([[1.0, 1.0]]))

    def test_pairwise_mahalanobis_distances_support_diagonal_regularization(self):
        means_a = array([[0.0], [0.0]])
        means_b = array([[2.0], [0.0]])
        zero_covariance = zeros((2, 2, 1))

        distances = pairwise_mahalanobis_distances(
            means_a,
            zero_covariance,
            means_b,
            zero_covariance,
            regularization=1.0,
        )

        npt.assert_allclose(distances, array([[2.0]]))

    def test_pairwise_mahalanobis_distances_return_empty_pairwise_shape(self):
        means_a = zeros((2, 0))
        means_b = zeros((2, 3))
        covariances_a = zeros((2, 2, 0))
        covariances_b = zeros((2, 2, 3))

        distances = pairwise_mahalanobis_distances(
            means_a,
            covariances_a,
            means_b,
            covariances_b,
        )

        self.assertEqual(distances.shape, (0, 3))

    def test_pairwise_covariance_shape_components_separate_shape_and_scale(self):
        covariances_a = array(
            [
                [[2.0], [0.0]],
                [[0.0], [1.0]],
            ]
        )
        covariances_b = array(
            [
                [[1.0, 2.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 2.0]],
            ]
        )

        shape_cost, logdet_cost, shape_similarity = (
            pairwise_covariance_shape_components(covariances_a, covariances_b)
        )

        npt.assert_allclose(shape_cost, array([[1.0 / 6.0, 1.0 / 6.0]]))
        npt.assert_allclose(logdet_cost, array([[log(2.0), log(2.0)]]))
        npt.assert_allclose(shape_similarity, exp(-shape_cost))

    def test_pairwise_covariance_shape_components_return_zero_for_equal_shapes(self):
        covariances_a = array(
            [
                [[1.0], [0.0]],
                [[0.0], [1.0]],
            ]
        )
        covariances_b = array(
            [
                [[2.0], [0.0]],
                [[0.0], [2.0]],
            ]
        )

        shape_cost, logdet_cost, shape_similarity = (
            pairwise_covariance_shape_components(covariances_a, covariances_b)
        )

        npt.assert_allclose(shape_cost, array([[0.0]]))
        npt.assert_allclose(logdet_cost, array([[math.log(4.0)]]))
        npt.assert_allclose(shape_similarity, array([[1.0]]))

    def test_pairwise_covariance_shape_components_support_empty_stacks(self):
        shape_cost, logdet_cost, shape_similarity = (
            pairwise_covariance_shape_components(zeros((2, 2, 0)), zeros((2, 2, 4)))
        )

        self.assertEqual(shape_cost.shape, (0, 4))
        self.assertEqual(logdet_cost.shape, (0, 4))
        self.assertEqual(shape_similarity.shape, (0, 4))

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            pairwise_mahalanobis_distances(
                array([[0.0]]),
                zeros((1, 1, 1)),
                array([[0.0]]),
                zeros((1, 1, 1)),
                regularization=-1.0,
            )

        with self.assertRaises(ValueError):
            pairwise_covariance_shape_components(
                zeros((2, 2, 1)),
                zeros((2, 2, 1)),
                epsilon=0.0,
            )

        with self.assertRaises(ValueError):
            pairwise_mahalanobis_distances(
                array([[0.0, 1.0]]),
                zeros((1, 1, 1)),
                array([[0.0]]),
                zeros((1, 1, 1)),
            )


if __name__ == "__main__":
    unittest.main()
