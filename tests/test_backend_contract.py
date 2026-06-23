import unittest

import numpy as np
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

    @unittest.skipUnless(
        backend.__backend_name__ == "pytorch",
        reason="PyTorch-specific NumPy dtype normalization regression test",
    )
    def test_pytorch_array_accepts_numpy_integer_and_bool_dtypes(self):
        int_values = array([1, 2, 3], dtype=np.int64)
        bool_values = array([1, 0, 1], dtype=np.bool_)

        self.assertEqual(to_numpy(int_values).dtype, np.dtype("int64"))
        self.assertEqual(to_numpy(bool_values).dtype, np.dtype("bool"))
        npt.assert_array_equal(to_numpy(int_values), np.array([1, 2, 3]))
        npt.assert_array_equal(to_numpy(bool_values), np.array([True, False, True]))

    def test_convert_to_wider_dtype_preserves_matching_boolean_dtype(self):
        if backend.__backend_name__ not in {"autograd", "numpy"}:
            self.skipTest("shared NumPy dtype promotion regression test")

        first, second = backend.convert_to_wider_dtype(
            [array([True, False]), array([False, True])]
        )

        self.assertEqual(to_numpy(first).dtype, np.dtype("bool"))
        self.assertEqual(to_numpy(second).dtype, np.dtype("bool"))

    def test_assignment_with_empty_indices_is_a_noop(self):
        original = array([1.0, 2.0, 3.0])

        assigned = backend.assignment(original, 99.0, [])
        added = backend.assignment_by_sum(original, 99.0, [])

        npt.assert_allclose(to_numpy(assigned), [1.0, 2.0, 3.0])
        npt.assert_allclose(to_numpy(added), [1.0, 2.0, 3.0])

    def test_assignment_accepts_array_like_inputs(self):
        assigned = backend.assignment(
            [[0.0, 0.0], [0.0, 0.0]],
            [4.0, 5.0],
            [(0, 1), (1, 0)],
        )
        added = backend.assignment_by_sum([0.0, 0.0, 0.0], [2.0, 3.0], [0, 2])

        npt.assert_allclose(to_numpy(assigned), [[0.0, 4.0], [5.0, 0.0]])
        npt.assert_allclose(to_numpy(added), [2.0, 0.0, 3.0])

    def test_assignment_empty_indices_coerces_array_like_inputs(self):
        assigned = backend.assignment([1.0, 2.0, 3.0], 99.0, [])
        added = backend.assignment_by_sum([1.0, 2.0, 3.0], 99.0, [])

        self.assertEqual(tuple(shape(assigned)), (3,))
        self.assertEqual(tuple(shape(added)), (3,))
        npt.assert_allclose(to_numpy(assigned), [1.0, 2.0, 3.0])
        npt.assert_allclose(to_numpy(added), [1.0, 2.0, 3.0])

    def test_mat_from_diag_triu_tril_accepts_array_like_inputs(self):
        result = backend.mat_from_diag_triu_tril([1.0, 2.0], [3.0], [4.0])

        self.assertEqual(tuple(shape(result)), (2, 2))
        npt.assert_allclose(to_numpy(result), [[1.0, 3.0], [4.0, 2.0]])

    def test_atleast_helpers_accept_scalar_inputs(self):
        npt.assert_allclose(to_numpy(backend.atleast_1d(5.0)), np.array([5.0]))
        npt.assert_allclose(to_numpy(backend.atleast_2d(5.0)), np.array([[5.0]]))

    def test_expand_dims_accepts_tuple_axis(self):
        values = array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        expanded = backend.expand_dims(values, axis=(0, 2))

        self.assertEqual(tuple(shape(expanded)), (1, 2, 1, 3))
        npt.assert_allclose(
            to_numpy(expanded), np.expand_dims(to_numpy(values), axis=(0, 2))
        )

    def test_meshgrid_uses_numpy_default_xy_indexing(self):
        first, second = backend.meshgrid(array([1, 2]), array([3, 4]))

        npt.assert_array_equal(to_numpy(first), np.array([[1, 2], [1, 2]]))
        npt.assert_array_equal(to_numpy(second), np.array([[3, 3], [4, 4]]))

    def test_cross_accepts_numpy_axis_keyword(self):
        first = array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        second = array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])

        result = backend.cross(first, second, axis=0)

        npt.assert_allclose(
            to_numpy(result),
            np.array([[0.0, 0.0], [0.0, 0.0], [1.0, -1.0]]),
        )

    def test_cross_accepts_numpy_axis_triplet_keywords(self):
        first_np = np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            ]
        )
        second_np = np.array(
            [
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ]
        )

        result = backend.cross(
            array(first_np), array(second_np), axisa=2, axisb=2, axisc=0
        )

        npt.assert_allclose(
            to_numpy(result), np.moveaxis(np.cross(first_np, second_np), -1, 0)
        )

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
        backend.__backend_name__ == "numpy",
        reason="NumPy backend vmap randomness contract",
    )
    def test_numpy_vmap_rejects_unknown_randomness_option(self):
        with self.assertRaisesRegex(ValueError, "randomness"):
            backend.vmap(lambda value: value, randomness="same")

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
