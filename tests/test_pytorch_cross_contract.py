import numpy as np
import numpy.testing as npt
import pyrecest.backend as backend
import pytest


@pytest.mark.skipif(
    backend.__backend_name__ != "pytorch",
    reason="PyTorch-specific cross-product contract regression test",
)
def test_pytorch_cross_returns_scalar_z_component_for_2d_vectors():
    result = backend.cross([1.0, 2.0], [3.0, 4.0])

    npt.assert_allclose(backend.to_numpy(result), np.array(-2.0))

    batched = backend.cross(
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
    )

    npt.assert_allclose(backend.to_numpy(batched), np.array([-4.0, -4.0]))


@pytest.mark.skipif(
    backend.__backend_name__ != "pytorch",
    reason="PyTorch-specific cross-product contract regression test",
)
def test_pytorch_cross_accepts_numpy_axis_keywords():
    left = np.arange(24.0).reshape(3, 2, 4)
    right = np.ones_like(left)

    result = backend.cross(
        backend.asarray(left),
        backend.asarray(right),
        axisa=0,
        axisb=0,
        axisc=1,
    )

    expected = np.cross(left, right, axisa=0, axisb=0, axisc=1)
    npt.assert_allclose(backend.to_numpy(result), expected)
