import pytest

from tests.support.backend_runner import run_backend_code


def test_pytorch_flip_matches_numpy_default_and_scalar_axis_contracts():
    pytest.importorskip("torch")

    code = (
        "import numpy as np\n"
        "import pyrecest.backend as backend\n"
        "values = backend.reshape(backend.arange(6), (2, 3))\n"
        "def as_list(value):\n"
        "    converted = backend.to_numpy(value)\n"
        "    if hasattr(converted, 'tolist'):\n"
        "        return converted.tolist()\n"
        "    return converted\n"
        "assert as_list(backend.flip(values)) == [[5, 4, 3], [2, 1, 0]]\n"
        "assert as_list(backend.flip(values, axis=None)) == [[5, 4, 3], [2, 1, 0]]\n"
        "assert as_list(backend.flip(values, axis=np.int64(0))) == [[3, 4, 5], [0, 1, 2]]\n"
        "assert as_list(backend.flip(values, axis=(0, 1))) == [[5, 4, 3], [2, 1, 0]]\n"
        "assert as_list(backend.flip(values, axis=())) == [[0, 1, 2], [3, 4, 5]]\n"
    )
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr
