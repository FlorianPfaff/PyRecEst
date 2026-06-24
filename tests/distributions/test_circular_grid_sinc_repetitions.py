import unittest

from pyrecest.backend import array, pi
from pyrecest.distributions.circle.circular_grid_distribution import (
    CircularGridDistribution,
)


class TestCircularGridSincRepetitions(unittest.TestCase):
    def _constant_grid_distribution(self):
        return CircularGridDistribution.from_function(
            lambda xs: xs * 0.0 + 1.0 / (2.0 * pi),
            9,
        )

    def test_pdf_via_sinc_accepts_positive_odd_integer_repetitions(self):
        dist = self._constant_grid_distribution()

        values = dist.pdf(array([0.0, pi]), use_sinc=True, sinc_repetitions=3)

        self.assertEqual(values.shape, (2,))

    def test_pdf_via_sinc_rejects_invalid_repetition_count(self):
        dist = self._constant_grid_distribution()

        invalid_repetitions = (True, False, 0, -1, 2, 3.0, array([3]), "3")
        for sinc_repetitions in invalid_repetitions:
            with self.subTest(sinc_repetitions=sinc_repetitions):
                with self.assertRaises(ValueError):
                    dist.pdf(
                        array([0.0]),
                        use_sinc=True,
                        sinc_repetitions=sinc_repetitions,
                    )


if __name__ == "__main__":
    unittest.main()
