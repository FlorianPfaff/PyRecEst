import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


def test_public_pytorch_argsort_rejects_kind_stable_conflict():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    code = """
import numpy as np
import pyrecest.backend as backend

values = backend.asarray([[2, 1], [3, 0]])
stable_values = (True, False, np.bool_(True), np.bool_(False), 1, 0, backend.asarray(True), backend.asarray(False))
for kind in ("quicksort", "heapsort", "stable", "mergesort"):
    for stable in stable_values:
        try:
            backend.argsort(values, axis=None, kind=kind, stable=stable)
        except ValueError as exc:
            assert "kind" in str(exc)
            assert "stable" in str(exc)
        else:
            raise AssertionError("accepted simultaneous argsort kind and stable")

assert backend.to_numpy(backend.argsort(values, axis=None, kind="stable")).tolist() == [3, 1, 0, 2]
assert backend.to_numpy(backend.argsort(values, axis=None, stable=True)).tolist() == [3, 1, 0, 2]
print("ok")
"""

    result = run_backend_code("pytorch", code)

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
