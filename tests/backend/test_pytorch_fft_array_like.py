import pytest


def test_pytorch_fft_helpers_accept_array_like_inputs():
    torch = pytest.importorskip("torch")

    from pyrecest._backend.pytorch import fft

    spectrum = fft.rfft([1.0, 2.0, 3.0, 4.0])
    expected_spectrum = torch.tensor(
        [10.0 + 0.0j, -2.0 + 2.0j, -2.0 + 0.0j], dtype=spectrum.dtype
    )
    assert torch.allclose(spectrum, expected_spectrum)

    reconstructed = fft.irfft(spectrum, n=4)
    assert torch.allclose(
        reconstructed,
        torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=reconstructed.dtype),
    )

    transformed = fft.fftn([[1.0, 2.0], [3.0, 4.0]], axes=(0, 1))
    expected_transformed = torch.tensor(
        [[10.0 + 0.0j, -2.0 + 0.0j], [-4.0 + 0.0j, 0.0 + 0.0j]],
        dtype=transformed.dtype,
    )
    assert torch.allclose(transformed, expected_transformed)

    assert fft.fftshift([0, 1, 2, 3]).tolist() == [2, 3, 0, 1]
    assert fft.ifftshift([2, 3, 0, 1]).tolist() == [0, 1, 2, 3]
