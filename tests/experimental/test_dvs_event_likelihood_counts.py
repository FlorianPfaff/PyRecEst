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

    def test_rejects_invalid_map_count_parameters(self):
        invalid_values = (2.5, "2", True)
        for field_name in ("max_map_iterations", "shape_update_modes"):
            for value in invalid_values:
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaisesRegex(ValueError, field_name):
                        PointProcessUpdateConfig(**{field_name: value})

    def test_normalizes_integer_like_map_count_parameters(self):
        config = PointProcessUpdateConfig(
            max_map_iterations=np.array(3.0),
            shape_update_modes=np.int64(4),
        )

        self.assertEqual(config.max_map_iterations, 3)
        self.assertIsInstance(config.max_map_iterations, int)
        self.assertEqual(config.shape_update_modes, 4)
        self.assertIsInstance(config.shape_update_modes, int)

    def test_accepts_zero_map_count_parameters(self):
        config = PointProcessUpdateConfig(
            max_map_iterations=np.array(0.0),
            shape_update_modes=0,
        )

        self.assertEqual(config.max_map_iterations, 0)
        self.assertEqual(config.shape_update_modes, 0)


if __name__ == "__main__":
    unittest.main()
