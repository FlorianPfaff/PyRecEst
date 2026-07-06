import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_reductions_reject_boolean_axis_arguments():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import pyrecest.backend as backend

values = backend.reshape(backend.arange(6), (2, 3))
boolean_values = values > 2

calls = [
    ("sum", lambda: backend.sum(values, axis=True)),
    ("sum dim", lambda: backend.sum(values, dim=True)),
    ("mean", lambda: backend.mean(values, axis=True)),
    ("std", lambda: backend.std(values, axis=True)),
    ("argmax", lambda: backend.argmax(values, axis=True)),
    ("argmin", lambda: backend.argmin(values, axis=True)),
    ("count_nonzero", lambda: backend.count_nonzero(values, axis=True)),
    ("any", lambda: backend.any(boolean_values, axis=True)),
    ("all", lambda: backend.all(boolean_values, axis=True)),
    ("max", lambda: backend.max(values, axis=True)),
    ("tuple axis", lambda: backend.sum(values, axis=(False,))),
]

for name, call in calls:
    try:
        call()
    except TypeError:
        pass
    else:
        raise AssertionError(f"{name} accepted a boolean reduction axis")

axis_result = backend.sum(values, axis=1, keepdims=True)
assert tuple(axis_result.shape) == (2, 1)
assert backend.to_numpy(axis_result).tolist() == [[3], [12]]
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
