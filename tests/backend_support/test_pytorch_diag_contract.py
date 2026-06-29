import importlib.util
import sys
from contextlib import contextmanager

import pytest


@contextmanager
def _isolated_pytorch_backend(monkeypatch):
    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "pyrecest" or name.startswith("pyrecest.")
    }
    for name in previous_modules:
        sys.modules.pop(name, None)

    monkeypatch.setenv("PYRECEST_BACKEND", "pytorch")
    try:
        import pyrecest._backend.pytorch as raw_pytorch_backend
        import pyrecest.backend as backend

        yield backend, raw_pytorch_backend
    finally:
        for name in list(sys.modules):
            if name == "pyrecest" or name.startswith("pyrecest."):
                sys.modules.pop(name, None)
        sys.modules.update(previous_modules)


@pytest.mark.backend_portable
def test_pytorch_diag_accepts_arraylike_and_numpy_k_keyword(monkeypatch):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    with _isolated_pytorch_backend(monkeypatch) as (backend, raw_pytorch_backend):
        vector = [1, 2, 3]
        upper = backend.diag(vector, k=1)
        assert backend.to_numpy(upper).tolist() == [
            [0, 1, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 3],
            [0, 0, 0, 0],
        ]

        lower = raw_pytorch_backend.diag(vector, k=-1)
        assert raw_pytorch_backend.to_numpy(lower).tolist() == [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 3, 0],
        ]

        matrix = [[0, 4, 0], [1, 0, 5], [0, 2, 0]]
        assert backend.to_numpy(backend.diag(matrix, k=-1)).tolist() == [1, 2]
        assert raw_pytorch_backend.to_numpy(
            raw_pytorch_backend.diag(matrix, k=1)
        ).tolist() == [4, 5]
