import numpy as np
from pyrecest._backend.numpy import linalg


def test_qr_mode_r_returns_single_batched_r_factor():
    matrices = np.stack(
        [
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]]),
            np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 4.0]]),
        ]
    )

    result = linalg.qr(matrices, mode="r")
    expected = np.linalg.qr(matrices, mode="r")

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, expected)


def test_qr_mode_raw_preserves_batched_numpy_contract():
    matrices = np.stack(
        [
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]]),
            np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 4.0]]),
        ]
    )

    h, tau = linalg.qr(matrices, mode="raw")
    expected_h, expected_tau = np.linalg.qr(matrices, mode="raw")

    np.testing.assert_allclose(h, expected_h)
    np.testing.assert_allclose(tau, expected_tau)
