import numpy as np
import pyrecest.backend as backend
import pytest


def test_common_helpers_ignore_late_backend_env_change(monkeypatch):
    if backend.__backend_name__ not in {"numpy", "autograd"}:
        pytest.skip("NumPy-style backend regression test")
    torch = pytest.importorskip("torch")

    monkeypatch.setenv("PYRECEST_BACKEND", "pytorch")

    result = backend.outer([1.0, 2.0], [3.0, 4.0])

    assert not torch.is_tensor(result)
    np.testing.assert_allclose(result, np.array([[3.0, 4.0], [6.0, 8.0]]))
