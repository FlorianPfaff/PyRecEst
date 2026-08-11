import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from pyrecest._backend.jax import fft  # noqa: E402


@pytest.mark.parametrize("shift", [fft.fftshift, fft.ifftshift])
def test_shift_accepts_numpy_integer_scalar_array_axis(shift):
    values = np.arange(12).reshape(3, 4)

    actual = np.asarray(shift(values, axes=np.asarray(1, dtype=np.int64)))
    expected = np.asarray(shift(values, axes=1))

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("shift", [fft.fftshift, fft.ifftshift])
def test_shift_accepts_jax_integer_axis_sequence(shift):
    values = np.arange(24).reshape(2, 3, 4)

    actual = np.asarray(shift(values, axes=jnp.asarray([0, 2])))
    expected = np.asarray(shift(values, axes=(0, 2)))

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("shift", [fft.fftshift, fft.ifftshift])
def test_shift_rejects_boolean_axes(shift):
    with pytest.raises(TypeError, match="axis must be an integer, not boolean"):
        shift(np.arange(4), axes=np.asarray(True))
