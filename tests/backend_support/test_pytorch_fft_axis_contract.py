import numpy as np
import numpy.testing as npt
import pytest

pytest.importorskip("torch")

import pyrecest._backend.pytorch.fft as pytorch_fft  # noqa: E402


@pytest.mark.backend_portable
def test_raw_pytorch_fft_helpers_accept_numpy_axis_aliases():
    vector = np.arange(4.0)
    spectrum = pytorch_fft.rfft(vector.tolist(), axis=0)
    npt.assert_allclose(spectrum.numpy(), np.fft.rfft(vector, axis=0))
    npt.assert_allclose(
        pytorch_fft.irfft(spectrum, axis=0).numpy(),
        np.fft.irfft(np.fft.rfft(vector, axis=0), axis=0),
    )

    matrix = np.arange(6.0).reshape(2, 3)
    npt.assert_allclose(
        pytorch_fft.fftn(matrix.tolist(), axes=(0, 1)).numpy(),
        np.fft.fftn(matrix, axes=(0, 1)),
    )
    npt.assert_allclose(
        pytorch_fft.ifftn(pytorch_fft.fftn(matrix, axes=(0, 1)), axes=(0, 1)).numpy(),
        np.fft.ifftn(np.fft.fftn(matrix, axes=(0, 1)), axes=(0, 1)),
    )
    npt.assert_allclose(
        pytorch_fft.fftn(matrix, dim=0, axes=None).numpy(),
        np.fft.fftn(matrix, axes=(0,)),
    )

    npt.assert_array_equal(
        pytorch_fft.fftshift([0, 1, 2, 3], axes=0).numpy(),
        np.fft.fftshift(np.asarray([0, 1, 2, 3]), axes=0),
    )
    npt.assert_array_equal(
        pytorch_fft.ifftshift([2, 3, 0, 1], axes=0).numpy(),
        np.fft.ifftshift(np.asarray([2, 3, 0, 1]), axes=0),
    )
    npt.assert_array_equal(
        pytorch_fft.fftshift(matrix, dim=0, axes=None).numpy(),
        np.fft.fftshift(matrix, axes=0),
    )


@pytest.mark.backend_portable
@pytest.mark.parametrize("fft_func", [pytorch_fft.rfft, pytorch_fft.irfft])
def test_raw_pytorch_single_axis_fft_rejects_none_axis_alias(fft_func):
    with pytest.raises(TypeError):
        fft_func(np.arange(4.0), axis=None)


@pytest.mark.backend_portable
def test_raw_pytorch_single_axis_fft_rejects_conflicting_none_axis_alias():
    with pytest.raises(TypeError):
        pytorch_fft.rfft(np.arange(4.0), axis=None, dim=0)


@pytest.mark.backend_portable
def test_raw_pytorch_fft_none_axis_alias_preserves_explicit_dim():
    matrix = np.arange(6.0).reshape(2, 3)

    npt.assert_allclose(
        pytorch_fft.fftn(matrix.tolist(), axes=None, dim=(0,)).numpy(),
        np.fft.fftn(matrix, axes=(0,)),
    )
    npt.assert_allclose(
        pytorch_fft.ifftn(
            pytorch_fft.fftn(matrix, dim=(0,)),
            axes=None,
            dim=(0,),
        ).numpy(),
        np.fft.ifftn(np.fft.fftn(matrix, axes=(0,)), axes=(0,)),
    )


def test_raw_pytorch_fft_rejects_conflicting_axis_aliases():
    with pytest.raises(TypeError):
        pytorch_fft.rfft(np.arange(4.0), axis=0, dim=1)
