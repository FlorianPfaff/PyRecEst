import pyrecest.backend as backend
import pytest
from pyrecest._backend.capabilities import get_unsupported_functions
from pyrecest.backend import array, linalg

_MATRIX = array([[1.0, 0.0], [0.0, 1.0]])


def _linalg_call_args(name):
    return {
        "fractional_matrix_power": (_MATRIX, 0.5),
        "is_single_matrix_pd": (_MATRIX,),
        "logm": (_MATRIX,),
        "quadratic_assignment": (_MATRIX, _MATRIX, {}),
        "solve_sylvester": (_MATRIX, _MATRIX, _MATRIX),
    }[name]


def test_declared_linalg_unsupported_functions_raise_not_implemented():
    unsupported = get_unsupported_functions(backend.__backend_name__, "linalg")
    if not unsupported:
        pytest.skip("active backend has no declared unsupported linalg functions")

    for name in unsupported:
        with pytest.raises(NotImplementedError):
            getattr(linalg, name)(*_linalg_call_args(name))


def test_pytorch_to_numpy_detaches_tensors_requiring_grad():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific backend behavior")

    import torch

    tensor = torch.tensor([1.0, 2.0], requires_grad=True)
    converted = backend.to_numpy(tensor)

    assert converted.tolist() == [1.0, 2.0]
