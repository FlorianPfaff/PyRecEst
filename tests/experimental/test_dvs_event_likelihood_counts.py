import unittest

import numpy as np
from pyrecest.experimental.dvs.event_likelihood import PointProcessUpdateConfig


class TestDVSPointProcessCountValidation(unittest.TestCase):
    def test_rejects_fractional_contour_samples(self):
        with self.assertRaisesRegex(ValueError, "contour_samples"):
            PointProcessUpdateConfig(contour_samples=3.5)

    def test_rejects_text_contour_samples(self):
        with self.assertRaisesRegex(ValueError, "contour_samples"):
            PointProcessUpdateConfig(contour_samples="5")

    def test_normalizes_integer_like_contour_samples(self):
        config = PointProcessUpdateConfig(contour_samples=np.array(5.0))

        self.assertEqual(config.contour_samples, 5)
        self.assertIsInstance(config.contour_samples, int)


if __name__ == "__main__":
    unittest.main()
