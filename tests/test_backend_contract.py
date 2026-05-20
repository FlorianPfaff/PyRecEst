import unittest

import numpy.testing as npt
import pyrecest.backend as backend
from pyrecest._backend import BACKEND_ATTRIBUTES
from pyrecest.backend import array, random, shape, to_numpy


class BackendContractTest(unittest.TestCase):
    def test_backend_attribute_lists_do_not_contain_duplicates(self):
        for module_name, attributes in BACKEND_ATTRIBUTES.items():
            with self.subTest(module=module_name or "root"):
                duplicates = sorted(
                    {name for name in attributes if attributes.count(name) > 1}
                )
                self.assertEqual(duplicates, [])

    def test_choice_supports_numpy_like_size_replace_and_probabilities(self):
        values = array([0, 1, 2, 3])
        weights = array([0.1, 0.2, 0.3, 0.4])

        random.seed(7)
        samples = random.choice(values, size=(2, 3), replace=True, p=weights)

        self.assertEqual(tuple(shape(samples)), (2, 3))
        samples_np = to_numpy(samples)
        npt.assert_array_less(samples_np, 4)
        npt.assert_array_less(-1, samples_np)

    def test_choice_without_replacement_returns_unique_values(self):
        values = array([0, 1, 2, 3])

        random.seed(11)
        samples = random.choice(values, size=values.shape[0], replace=False)

        self.assertEqual(tuple(shape(samples)), (values.shape[0],))
        self.assertEqual(len(set(to_numpy(samples).tolist())), values.shape[0])

    def test_normal_and_uniform_accept_none_size_for_scalar_sample(self):
        random.seed(13)

        normal_sample = random.normal(loc=1.0, scale=2.0, size=None)
        uniform_sample = random.uniform(low=0.0, high=1.0, size=None)

        float(normal_sample)
        self.assertTrue(0.0 <= float(uniform_sample) <= 1.0)

    @unittest.skipUnless(
        backend.__backend_name__ == "jax",
        reason="JAX-specific explicit RNG state contract",
    )
    def test_jax_explicit_state_returns_new_state_and_sample(self):
        state = random.get_state()

        new_state, sample = random.uniform(size=(3,), state=state)

        self.assertEqual(tuple(shape(sample)), (3,))
        self.assertNotEqual(str(state), str(new_state))


if __name__ == "__main__":
    unittest.main()
