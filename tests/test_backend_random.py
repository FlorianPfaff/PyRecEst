import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import random


class TestBackendRandom(unittest.TestCase):
    def test_randint_returns_integer_samples_in_bounds(self):
        if pyrecest.backend.__backend_name__ == "jax":
            samples = random.randint((64,), minval=0, maxval=5)
        elif pyrecest.backend.__backend_name__ == "pytorch":
            samples = random.randint(0, 5, (64,))
        else:
            samples = random.randint(0, 5, size=(64,))

        npt.assert_equal(samples.shape, (64,))
        self.assertIn(samples.dtype, (pyrecest.backend.int32, pyrecest.backend.int64))
        npt.assert_array_less(-1, samples)
        npt.assert_array_less(samples, 5)

    def test_choice_single_sample_preserves_sample_axis(self):
        values = pyrecest.backend.array([[0, 1], [2, 3], [4, 5]])

        sample = random.choice(values, 1)

        self.assertEqual(sample.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
