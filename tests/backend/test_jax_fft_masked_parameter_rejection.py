import numpy as np
import pytest

pytest.importorskip("jax")

from pyrecest._backend.jax import fft  # noqa: E402


@pytest.mark.parametrize("transform", [fft.rfft, fft.irfft])
def test_real_fft_rejects_masked_length(transform):
    with pytest.raises(TypeError, match="n must be an integer length"):
        transform(np.arange(4.0), n=np.ma.array(4, mask=True))


@pytest.mark.parametrize("transform", [fft.rfft, fft.irfft])
def test_real_fft_rejects_masked_axis(transform):
    with pytest.raises(TypeError, match="axis must be an integer"):
        transform(np.arange(4.0), axis=np.ma.array(0, mask=True))


@pytest.mark.parametrize("transform", [fft.fftn, fft.ifftn])
def test_complex_fft_rejects_masked_shape_entry(transform):
    with pytest.raises(TypeError, match="s entries must be integer lengths"):
        transform(np.ones((2, 2)), s=(np.ma.array(2, mask=True), 2))


@pytest.mark.parametrize("transform", [fft.fftn, fft.ifftn])
def test_complex_fft_rejects_masked_axis_entry(transform):
    with pytest.raises(TypeError, match="axes entries must be integers"):
        transform(np.ones((2, 2)), axes=(np.ma.array(0, mask=True), 1))


@pytest.mark.parametrize("shift", [fft.fftshift, fft.ifftshift])
def test_shift_rejects_masked_axis(shift):
    with pytest.raises(
        TypeError,
        match="axes must be None, an integer axis, or a sequence of integer axes",
    ):
        shift(np.arange(4), axes=np.ma.array(0, mask=True))


def test_unmasked_masked_integer_parameter_remains_supported():
    values = np.arange(4.0)

    actual = np.asarray(fft.rfft(values, n=np.ma.array(4, mask=False)))
    expected = np.asarray(fft.rfft(values, n=4))

    np.testing.assert_array_equal(actual, expected)
