import unittest

import pyrecest.backend
from pyrecest.backend import array, eye
from pyrecest.utils.point_set_registration import AffineTransform, estimate_transform


class TestPointSetRegistrationVectorShapes(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_affine_transform_rejects_matrix_shaped_offsets(self):
        for offset in (array([[1.0, -2.0]]), array([[1.0], [-2.0]])):
            with self.subTest(offset_shape=offset.shape):
                with self.assertRaisesRegex(
                    ValueError, "offset must be one-dimensional"
                ):
                    AffineTransform(eye(2), offset)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_estimate_transform_rejects_matrix_shaped_weights(self):
        source = array([[0.0, 0.0], [1.0, 0.0]])
        target = source + array([1.0, -1.0])

        for weights in (array([[1.0, 1.0]]), array([[1.0], [1.0]])):
            with self.subTest(weights_shape=weights.shape):
                with self.assertRaisesRegex(
                    ValueError, "weights must be one-dimensional"
                ):
                    estimate_transform(
                        source,
                        target,
                        model="translation",
                        weights=weights,
                    )


if __name__ == "__main__":
    unittest.main()
