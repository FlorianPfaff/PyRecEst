import unittest
from math import sqrt

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, eye, linalg, ones, pi, random, stack, sum
from pyrecest.distributions import SO3ProductDiracDistribution
from pyrecest.distributions.cart_prod.hyperhemisphere_cart_prod_dirac_distribution import (
    HyperhemisphereCartProdDiracDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_dirac_distribution import (
    HyperhemisphericalDiracDistribution,
)


class SO3ProductDiracDistributionTest(unittest.TestCase):
    def setUp(self):
        self.identity = array([0.0, 0.0, 0.0, 1.0])
        self.z_ninety = array([0.0, 0.0, sqrt(0.5), sqrt(0.5)])
        self.x_ninety = array([sqrt(0.5), 0.0, 0.0, sqrt(0.5)])

    def test_constructor_normalizes_and_canonicalizes_quaternions(self):
        locations = array(
            [
                [
                    [0.0, 0.0, 0.0, 2.0],
                    [0.0, 0.0, sqrt(0.5), sqrt(0.5)],
                ],
                [
                    [0.0, 0.0, 0.0, -1.0],
                    [-sqrt(0.5), 0.0, 0.0, -sqrt(0.5)],
                ],
            ]
        )
        weights = array([1.0, 3.0])

        with self.assertWarns(RuntimeWarning):
            dist = SO3ProductDiracDistribution(locations, weights)

        self.assertEqual(dist.d.shape, (2, 2, 4))
        self.assertEqual(dist.num_rotations, 2)
        self.assertEqual(dist.dim, 6)
        self.assertTrue(dist.is_valid())
        npt.assert_allclose(linalg.norm(dist.d, axis=-1), ones((2, 2)))
        npt.assert_allclose(dist.w, array([0.25, 0.75]))
        self.assertGreaterEqual(float(dist.d[1, 0, -1]), 0.0)
        self.assertGreaterEqual(float(dist.d[1, 1, -1]), 0.0)

    def test_flattened_input_and_output(self):
        flat_locations = array(
            [
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    sqrt(0.5),
                    sqrt(0.5),
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    sqrt(0.5),
                    0.0,
                    0.0,
                    sqrt(0.5),
                ],
            ]
        )

        dist = SO3ProductDiracDistribution(flat_locations)

        self.assertEqual(dist.d.shape, (2, 2, 4))
        self.assertEqual(dist.as_flat_quaternions().shape, (2, 8))
        npt.assert_allclose(dist.as_flat_quaternions(), flat_locations)
        self.assertIsInstance(dist, HyperhemisphereCartProdDiracDistribution)
        self.assertEqual(dist.dim_hemisphere, 3)
        self.assertEqual(dist.n_hemispheres, 2)
        npt.assert_allclose(dist.as_component_array(), dist.as_quaternions())

    def test_single_particle_input_with_num_rotations(self):
        locations = array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, sqrt(0.5), sqrt(0.5)],
            ]
        )

        dist = SO3ProductDiracDistribution(locations, num_rotations=2)

        self.assertEqual(dist.d.shape, (1, 2, 4))
        npt.assert_allclose(dist.as_quaternions()[0], locations)

    def test_marginalize_rotations(self):
        dist = SO3ProductDiracDistribution(
            array(
                [
                    [self.identity, self.z_ninety],
                    [self.identity, self.x_ninety],
                ]
            ),
            array([0.4, 0.6]),
        )

        rotation_marginal = dist.marginalize_rotation(1)
        product_marginal = dist.marginalize_rotations([1])

        self.assertIsInstance(rotation_marginal, HyperhemisphericalDiracDistribution)
        self.assertEqual(rotation_marginal.d.shape, (2, 4))
        npt.assert_allclose(rotation_marginal.w, dist.w)
        self.assertIsInstance(product_marginal, SO3ProductDiracDistribution)
        self.assertEqual(product_marginal.d.shape, (2, 1, 4))
        npt.assert_allclose(product_marginal.d[:, 0, :], dist.d[:, 1, :])

    def test_geodesic_distances(self):
        dist = SO3ProductDiracDistribution(
            array(
                [
                    [self.identity, self.z_ninety],
                    [self.identity, self.identity],
                ]
            ),
            array([0.25, 0.75]),
        )
        target = array([self.identity, self.z_ninety])

        component_distances = dist.distance_to(target, reduce=False)
        summed_distances = dist.distance_to(target)

        npt.assert_allclose(component_distances[0], array([0.0, 0.0]), atol=1e-8)
        npt.assert_allclose(component_distances[1], array([0.0, pi / 2.0]), atol=1e-6)
        npt.assert_allclose(summed_distances, array([0.0, pi / 2.0]), atol=1e-6)
        npt.assert_allclose(dist.angular_error_mean(target), 0.75 * pi / 2.0, atol=1e-6)

    def test_mean_and_rotation_matrix_for_single_particle(self):
        dist = SO3ProductDiracDistribution(array([[self.identity, self.z_ninety]]))

        mean_quaternions = dist.mean_quaternion()
        rotation_matrices = dist.mean_rotation_matrices()
        expected_rotation_matrices = array(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                [
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            ]
        )

        npt.assert_allclose(mean_quaternions, array([self.identity, self.z_ninety]))
        npt.assert_allclose(rotation_matrices, expected_rotation_matrices, atol=1e-6)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "pytorch",
        reason="Rotation matrix conversion is not supported on the PyTorch backend.",
    )
    def test_rotation_matrix_roundtrip(self):
        z_matrix = array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        x_matrix = array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        rotations = stack(
            [stack([eye(3), z_matrix], axis=0), stack([eye(3), x_matrix], axis=0)],
            axis=0,
        )
        weights = array([0.2, 0.8])

        dist = SO3ProductDiracDistribution.from_rotation_matrices(rotations, w=weights)

        self.assertEqual(dist.d.shape, (2, 2, 4))
        self.assertEqual(dist.num_rotations, 2)
        npt.assert_allclose(dist.w, weights)
        npt.assert_allclose(
            SO3ProductDiracDistribution.as_rotation_matrices(dist.as_quaternions()),
            rotations,
            atol=1e-6,
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "pytorch",
        reason="Rotation matrix conversion is not supported on the PyTorch backend.",
    )
    def test_single_product_rotation_matrix_roundtrip(self):
        z_matrix = array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rotations = stack([eye(3), z_matrix], axis=0)

        dist = SO3ProductDiracDistribution.from_rotation_matrices(rotations)

        self.assertEqual(dist.d.shape, (1, 2, 4))
        npt.assert_allclose(
            SO3ProductDiracDistribution.as_rotation_matrices(dist.as_quaternions())[0],
            rotations,
            atol=1e-6,
        )

    def test_sampling(self):
        random.seed(0)
        dist = SO3ProductDiracDistribution(
            array(
                [
                    [self.identity, self.z_ninety],
                    [self.identity, self.x_ninety],
                ]
            )
        )

        samples = dist.sample(8)
        single_sample = dist.sample(1)

        self.assertEqual(samples.shape, (8, 2, 4))
        self.assertEqual(single_sample.shape, (1, 2, 4))
        npt.assert_allclose(linalg.norm(samples, axis=-1), ones((8, 2)))
        npt.assert_allclose(linalg.norm(single_sample, axis=-1), ones((1, 2)))
        npt.assert_allclose(sum(dist.w), 1.0)


if __name__ == "__main__":
    unittest.main()
