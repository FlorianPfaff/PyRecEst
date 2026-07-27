import numpy as np
import numpy.testing as npt
import pyrecest.backend as backend
import pytest


def _to_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_pytorch_backend_exposes_dot_and_matvec():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific backend helper contract")

    first = backend.asarray([[1.0, 2.0], [3.0, 4.0]])
    second = backend.asarray([[5.0, 6.0], [7.0, 8.0]])

    assert _to_python(backend.dot(first, second)) == [[19.0, 22.0], [43.0, 50.0]]
    assert _to_python(backend.dot(first, [5.0, 6.0])) == [17.0, 39.0]
    assert _to_python(backend.outer([1.0, 2.0], [3.0, 4.0])) == [
        [3.0, 4.0],
        [6.0, 8.0],
    ]
    assert float(_to_python(backend.dot(backend.asarray(2.0), 3.0))) == pytest.approx(
        6.0
    )

    batched_matvec = backend.matvec(
        backend.asarray([[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 3.0]]]),
        first,
    )
    shared_vector_matvec = backend.matvec(
        backend.asarray([[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 3.0]]]),
        backend.asarray([1.0, 2.0]),
    )

    assert _to_python(batched_matvec) == [[1.0, 2.0], [6.0, 12.0]]
    assert _to_python(shared_vector_matvec) == [[1.0, 2.0], [2.0, 6.0]]


def test_raw_pytorch_transpose_accepts_numpy_axes_array():
    pytest.importorskip("torch")
    import pyrecest  # noqa: F401
    import pyrecest._backend.pytorch as raw_pytorch

    values = np.arange(24.0).reshape(2, 3, 4)
    axes = np.array([2, 0, 1])

    result = raw_pytorch.transpose(values, axes=axes)

    npt.assert_allclose(result.numpy(), np.transpose(values, axes=axes))


def test_pytorch_backend_transpose_accepts_numpy_axes_array():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific backend helper contract")

    values = np.arange(24.0).reshape(2, 3, 4)
    axes = np.array([2, 0, 1])

    result = backend.transpose(backend.array(values), axes=axes)

    npt.assert_allclose(backend.to_numpy(result), np.transpose(values, axes=axes))
