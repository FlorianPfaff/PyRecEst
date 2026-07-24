import pyrecest.backend as backend
import pyrecest.backend_support  # noqa: F401 ensure compatibility patches are installed
import pytest

pytorch_backend = pytest.importorskip("pyrecest._backend.pytorch")


def _as_list(value):
    return pytorch_backend.to_numpy(value).tolist()


def test_raw_pytorch_roll_accepts_array_like_inputs_with_numpy_contract():
    values = [[0, 1, 2], [3, 4, 5]]

    assert _as_list(pytorch_backend.roll(values, 1)) == [[5, 0, 1], [2, 3, 4]]
    assert _as_list(pytorch_backend.roll(values, shift=(1, 2), axis=(0, 1))) == [
        [4, 5, 3],
        [1, 2, 0],
    ]
    assert _as_list(pytorch_backend.roll(values, shift=(1, 2), axis=0)) == [
        [3, 4, 5],
        [0, 1, 2],
    ]
    assert _as_list(pytorch_backend.roll(values, shifts=1, dims=1)) == [
        [2, 0, 1],
        [5, 3, 4],
    ]


def test_public_pytorch_roll_accepts_array_like_inputs_when_active():
    if getattr(backend, "__backend_name__", None) != "pytorch":
        pytest.skip("public PyTorch backend is not active")

    values = [[0, 1, 2], [3, 4, 5]]
    assert backend.to_numpy(backend.roll(values, 1)).tolist() == [[5, 0, 1], [2, 3, 4]]
    assert backend.to_numpy(
        backend.roll(values, shift=(1, 2), axis=(0, 1))
    ).tolist() == [
        [4, 5, 3],
        [1, 2, 0],
    ]
