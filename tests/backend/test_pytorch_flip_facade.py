from tests.support.backend_runner import run_backend_code


def test_pytorch_flip_matches_numpy_default_and_scalar_axis_contracts():
    code = """
import numpy as np
import pyrecest.backend as backend

values = backend.reshape(backend.arange(6), (2, 3))

def as_list(value):
    converted = backend.to_numpy(value)
    if hasattr(converted, "tolist"):
        return converted.tolist()
    return converted

assert as_list(backend.flip(values)) == [[5, 4, 3], [2, 1, 0]]
assert as_list(backend.flip(values, axis=None)) == [[5, 4, 3], [2, 1, 0]]
assert as_list(backend.flip(values, axis=np.int64(0))) == [[3, 4, 5], [0, 1, 2]]
assert as_list(backend.flip(values, axis=(0, 1))) == [[5, 4, 3], [2, 1, 0]]
assert as_list(backend.flip(values, axis=())) == [[0, 1, 2], [3, 4, 5]]
"""
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr
