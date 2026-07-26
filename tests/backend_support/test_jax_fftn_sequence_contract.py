import unittest

import numpy as np

jax_backend = None
try:
    from pyrecest._backend import jax as jax_backend
except ModuleNotFoundError:
    pass


@unittest.skipIf(jax_backend is None, "JAX is not installed")
class TestJaxFftnSequenceContract(unittest.TestCase):
    def test_fftn_accepts_scalar_array_entries_in_shape_and_axes(self):
        values = np.arange(4.0)
        actual = np.asarray(
            jax_backend.fft.fftn(values, s=(np.array(4),), axes=(np.array(0),))
        )
        expected = np.fft.fftn(values, s=(4,), axes=(0,))
        self.assertTrue(np.allclose(actual, expected))

    def test_ifftn_accepts_scalar_array_entries_in_shape_and_axes(self):
        spectrum = np.fft.fftn(np.arange(4.0))
        actual = np.asarray(
            jax_backend.fft.ifftn(spectrum, s=[np.array(4)], axes=[np.array(0)])
        )
        expected = np.fft.ifftn(spectrum, s=(4,), axes=(0,))
        self.assertTrue(np.allclose(actual, expected))

    def test_fftn_rejects_boolean_shape_entries(self):
        bad_shape = [bool(1)]
        with self.assertRaisesRegex(TypeError, "integer lengths"):
            jax_backend.fft.fftn(np.arange(4.0), s=bad_shape, axes=(0,))

    def test_fftn_rejects_boolean_axis_entries(self):
        bad_axes = (bool(1),)
        with self.assertRaisesRegex(TypeError, "integers"):
            jax_backend.fft.fftn(np.arange(4.0), s=(4,), axes=bad_axes)


if __name__ == "__main__":
    unittest.main()
