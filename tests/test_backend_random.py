import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import random


class TestBackendRandom(unittest.TestCase):
    def test_randint_returns_integer_samples_in_bounds(self):
        samples = random.randint(0, 5, size=(64,))

        npt.assert_equal(samples.shape, (64,))
        self.assertIn(samples.dtype, (pyrecest.backend.int32, pyrecest.backend.int64))
        npt.assert_array_less(-1, samples)
        npt.assert_array_less(samples, 5)

    def test_uniform_allows_degenerate_interval(self):
        samples = random.uniform(2.5, 2.5, size=(4,))

        self.assertEqual(tuple(pyrecest.backend.shape(samples)), (4,))
        npt.assert_allclose(pyrecest.backend.to_numpy(samples), [2.5, 2.5, 2.5, 2.5])

    def test_uniform_rejects_descending_interval(self):
        with self.assertRaises(ValueError):
            random.uniform(2.0, 1.0, size=(3,))

    def test_choice_single_sample_preserves_sample_axis(self):
        values = pyrecest.backend.array([[0, 1], [2, 3], [4, 5]])

        sample = random.choice(values, 1)

        self.assertEqual(sample.shape, (1, 2))

    def test_choice_accepts_python_list_population(self):
        samples = random.choice([10, 20, 30], size=(32,))

        self.assertEqual(samples.shape, (32,))
        npt.assert_array_less(9, samples)
        npt.assert_array_less(samples, 31)
        for sample in pyrecest.backend.to_numpy(samples).tolist():
            self.assertIn(sample, (10, 20, 30))

    def test_choice_accepts_integer_population(self):
        samples = random.choice(5, size=(32,))

        self.assertEqual(samples.shape, (32,))
        npt.assert_array_less(-1, samples)
        npt.assert_array_less(samples, 5)

        no_replacement = random.choice(5, size=5, replace=False)

        self.assertEqual(tuple(pyrecest.backend.shape(no_replacement)), (5,))
        self.assertEqual(
            len(set(pyrecest.backend.to_numpy(no_replacement).tolist())), 5
        )

    def test_choice_accepts_python_probability_sequence(self):
        values = pyrecest.backend.array([10, 20, 30])

        samples = random.choice(values, size=(32,), p=[0.2, 0.3, 0.5])

        self.assertEqual(samples.shape, (32,))
        for sample in pyrecest.backend.to_numpy(samples).tolist():
            self.assertIn(sample, (10, 20, 30))

    def test_choice_accepts_zero_sized_integer_population_sample(self):
        samples = random.choice(0, size=(0,))

        self.assertEqual(samples.shape, (0,))

    def test_choice_samples_matrix_values_along_requested_axis(self):
        values = pyrecest.backend.array([[0, 1, 2], [3, 4, 5]])

        sample = random.choice(values, size=2, replace=False, axis=1)

        self.assertEqual(tuple(pyrecest.backend.shape(sample)), (2, 2))
        sample_np = pyrecest.backend.to_numpy(sample)
        self.assertTrue(set(sample_np[0].tolist()).issubset({0, 1, 2}))
        self.assertTrue(set(sample_np[1].tolist()).issubset({3, 4, 5}))

    def test_multivariate_normal_accepts_python_sequences(self):
        samples = random.multivariate_normal(
            [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], size=(6,)
        )

        self.assertEqual(tuple(pyrecest.backend.shape(samples)), (6, 2))

    def test_multinomial_accepts_python_probability_sequence(self):
        sample = random.multinomial(12, [0.25, 0.75])

        self.assertEqual(sample.shape, (2,))
        self.assertEqual(int(pyrecest.backend.sum(sample)), 12)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ != "jax",
        "JAX-specific multinomial size support",
    )
    def test_jax_multinomial_accepts_size_argument(self):
        samples = random.multinomial(5, [0.25, 0.75], size=(2, 3))

        self.assertEqual(tuple(pyrecest.backend.shape(samples)), (2, 3, 2))
        npt.assert_array_equal(
            pyrecest.backend.to_numpy(pyrecest.backend.sum(samples, axis=-1)),
            [[5, 5, 5], [5, 5, 5]],
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ != "jax", "JAX-specific size validation"
    )
    def test_jax_random_rejects_invalid_size_arguments(self):
        invalid_sizes = (True, (2, True), 1.5, (2, 1.5), "3", -1, (2, -1))
        random_calls = (
            lambda size: random.rand(size=size),
            lambda size: random.uniform(size=size),
            lambda size: random.randint(0, 5, size=size),
            lambda size: random.normal(size=size),
            lambda size: random.choice(5, size=size),
            lambda size: random.multivariate_normal([0.0], [[1.0]], size=size),
            lambda size: random.multinomial(5, [0.5, 0.5], size=size),
        )

        for invalid_size in invalid_sizes:
            for random_call in random_calls:
                with self.subTest(size=invalid_size, random_call=random_call):
                    with self.assertRaises((TypeError, ValueError)):
                        random_call(invalid_size)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ != "jax",
        "JAX-specific multinomial validation",
    )
    def test_jax_multinomial_rejects_invalid_trial_count(self):
        for invalid_n in (True, 1.5, -1):
            with self.subTest(n=invalid_n):
                with self.assertRaises((TypeError, ValueError)):
                    random.multinomial(invalid_n, [0.5, 0.5])

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ != "jax", "JAX-specific RNG state contract"
    )
    def test_jax_multinomial_explicit_state_does_not_mutate_global_state(self):
        state = random.create_random_state(123)
        original_global_state = random.get_state()

        state_after, sample = random.multinomial(
            10, pyrecest.backend.array([0.25, 0.75]), state=state
        )

        self.assertEqual(sample.shape, (2,))
        self.assertEqual(int(pyrecest.backend.sum(sample)), 10)
        npt.assert_array_equal(random.get_state(), original_global_state)
        self.assertFalse(pyrecest.backend.all(random.get_state() == state_after))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ != "jax", "JAX-specific RNG state contract"
    )
    def test_jax_multinomial_uses_and_advances_global_state(self):
        random.seed(321)
        initial_state = random.get_state()

        sample = random.multinomial(8, pyrecest.backend.array([0.5, 0.5]))

        self.assertEqual(sample.shape, (2,))
        self.assertEqual(int(pyrecest.backend.sum(sample)), 8)
        self.assertFalse(pyrecest.backend.all(random.get_state() == initial_state))


if __name__ == "__main__":
    unittest.main()
