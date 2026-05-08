import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, pi
from pyrecest.distributions import SO3TangentGaussianDistribution, so3_helpers
from pyrecest.distributions._so3_helpers import (
    normalize_quaternions as private_normalize_quaternions,
)
from tests.distributions.so3_test_helpers import (
    ATOL,
    assert_matches_z_rotation,
    z_quaternion,
)


class SO3HelpersTest(unittest.TestCase):
    def test_public_helpers_normalize_and_private_import_remains_compatible(self):
        quaternion = array([0.0, 0.0, 0.0, -2.0])

        expected = array([[0.0, 0.0, 0.0, 1.0]])
        npt.assert_allclose(
            so3_helpers.canonicalize_quaternions(quaternion), expected, atol=ATOL
        )
        npt.assert_allclose(
            so3_helpers.normalize_quaternions(quaternion),
            private_normalize_quaternions(quaternion),
            atol=ATOL,
        )

    def test_exp_log_roundtrip_with_base_rotation(self):
        base = z_quaternion(pi / 3.0)
        tangent_vectors = array([[0.1, -0.2, 0.05], [0.0, 0.0, 0.0]])

        rotations = so3_helpers.exp_map(tangent_vectors, base=base)
        roundtrip = so3_helpers.log_map(rotations, base=base)

        npt.assert_allclose(roundtrip, tangent_vectors, atol=ATOL)
        npt.assert_allclose(
            rotations,
            SO3TangentGaussianDistribution.exp_map(tangent_vectors, base=base),
            atol=ATOL,
        )

    def test_quaternion_multiply_and_conjugate_cancel_rotation(self):
        rotation = z_quaternion(pi / 2.0)

        product = so3_helpers.quaternion_multiply(
            rotation, so3_helpers.quaternion_conjugate(rotation)
        )

        npt.assert_allclose(product, array([[0.0, 0.0, 0.0, 1.0]]), atol=ATOL)

    def test_rotation_matrix_alias_and_vector_rotation(self):
        rotation = z_quaternion(pi / 2.0)

        rotation_matrix = so3_helpers.rotation_matrices_from_quaternions(rotation)[0]
        rotated_vector = so3_helpers.rotate_vectors(rotation, array([1.0, 0.0, 0.0]))

        assert_matches_z_rotation(self, rotation_matrix, pi / 2.0)
        npt.assert_allclose(rotated_vector, array([[0.0, 1.0, 0.0]]), atol=ATOL)

    def test_geodesic_distance_respects_antipodal_equivalence(self):
        identity = array([0.0, 0.0, 0.0, 1.0])
        identity_antipodal = array([0.0, 0.0, 0.0, -1.0])
        quarter_turn = z_quaternion(pi / 2.0)

        npt.assert_allclose(
            so3_helpers.geodesic_distance(identity, identity_antipodal),
            array([0.0]),
            atol=ATOL,
        )
        npt.assert_allclose(
            so3_helpers.geodesic_distance(identity, quarter_turn),
            array([pi / 2.0]),
            atol=ATOL,
        )


if __name__ == "__main__":
    unittest.main()
