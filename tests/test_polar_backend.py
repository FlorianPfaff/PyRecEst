import importlib.util

import numpy as np
import numpy.testing as npt
import pytest
from pyrecest._backend import numpy as numpy_backend

pytorch_backend = None
if importlib.util.find_spec("torch") is not None:
    from pyrecest._backend import pytorch as pytorch_backend


def test_numpy_polar_handles_rectangular_right_factor():
    value = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    unitary, positive = numpy_backend.linalg.polar(value)

    assert unitary.shape == value.shape
    assert positive.shape == (value.shape[1], value.shape[1])
    npt.assert_allclose(unitary @ positive, value, atol=1e-12)


def test_numpy_polar_handles_rectangular_left_factor():
    value = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    unitary, positive = numpy_backend.linalg.polar(value, side="left")

    assert unitary.shape == value.shape
    assert positive.shape == (value.shape[0], value.shape[0])
    npt.assert_allclose(positive @ unitary, value, atol=1e-12)


@pytest.mark.skipif(pytorch_backend is None, reason="PyTorch is not installed")
def test_pytorch_polar_handles_rectangular_right_factor():
    value = pytorch_backend.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=pytorch_backend.float64
    )

    unitary, positive = pytorch_backend.linalg.polar(value)

    assert tuple(unitary.shape) == tuple(value.shape)
    assert tuple(positive.shape) == (value.shape[1], value.shape[1])
    assert bool(pytorch_backend.allclose(unitary @ positive, value, atol=1e-10))


@pytest.mark.skipif(pytorch_backend is None, reason="PyTorch is not installed")
def test_pytorch_polar_handles_rectangular_left_factor():
    value = pytorch_backend.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=pytorch_backend.float64
    )

    unitary, positive = pytorch_backend.linalg.polar(value, side="left")

    assert tuple(unitary.shape) == tuple(value.shape)
    assert tuple(positive.shape) == (value.shape[0], value.shape[0])
    assert bool(pytorch_backend.allclose(positive @ unitary, value, atol=1e-10))
