import unittest

import pyrecest.backend
from pyrecest.backend import array
from pyrecest.utils.point_set_registration import estimate_transform


class TestEstimateTransformWeightValidation(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_estimate_transform_rejects_nonfinite_weights(self):
        source = array([[0.0, 0.0], [1.0, 0.0]])
        target = source + array([1.0, -1.0])

        for bad_weight in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(bad_weight=bad_weight):
                with self.assertRaisesRegex(ValueError, "finite"):
                    estimate_transform(
                        source,
                        target,
                        model="translation",
                        weights=array([1.0, bad_weight]),
                    )


if __name__ == "__main__":
    unittest.main()
