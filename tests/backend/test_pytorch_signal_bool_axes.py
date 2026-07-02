from __future__ import annotations

import numpy as np
import pytest

pytorch_backend = pytest.importorskip("pyrecest._backend.pytorch")


@pytest.mark.parametrize(
    ("axes", "expected"),
    [
        (
            False,
            [[0.5, 2.0, 1.5], [3.5, 9.0, 7.5], [6.0, 10.0, 9.0]],
        ),
        (
            True,
            [[0.5, 2.0, 4.0, 4.0, 1.5], [6.0, 15.5, 25.0, 19.5, 9.0]],
        ),
    ],
)
def test_fftconvolve_accepts_python_bool_axes_like_scalar_axes(
    axes: bool,
    expected: list[list[float]],
) -> None:
    in1 = pytorch_backend.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    in2 = pytorch_backend.array([[0.5, 1.0, 0.5], [1.5, 2.0, 1.5]])

    result = pytorch_backend.signal.fftconvolve(in1, in2, axes=axes)

    assert pytorch_backend.allclose(result, pytorch_backend.array(expected))


def test_fftconvolve_rejects_numpy_bool_axis() -> None:
    in1 = pytorch_backend.ones((2, 3))
    in2 = pytorch_backend.ones((2, 3))

    with pytest.raises((TypeError, ValueError)):
        pytorch_backend.signal.fftconvolve(in1, in2, axes=np.bool_(True))
