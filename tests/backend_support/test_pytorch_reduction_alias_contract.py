import importlib.util

import pytest

from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_public_reductions_accept_dim_keepdim_aliases():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import pyrecest.backend as backend

values = backend.array([[1, 2, 3], [4, 5, 6]])

assert backend.to_numpy(backend.max(values, dim=1, keepdim=True)).tolist() == [[3], [6]]
assert backend.to_numpy(backend.min(values, dim=0, keepdim=True)).tolist() == [[1, 2, 3]]
assert backend.to_numpy(backend.sum(values, dim=1, keepdim=True)).tolist() == [[6], [15]]
assert backend.to_numpy(backend.prod(values, dim=1, keepdim=True)).tolist() == [[6], [120]]
assert backend.to_numpy(backend.count_nonzero(values, dim=0, keepdim=True)).tolist() == [[2, 2, 2]]

assert backend.to_numpy(backend.any(values > 4, dim=1, keepdim=True)).tolist() == [[False], [True]]
assert backend.to_numpy(backend.all(values > 0, dim=0, keepdim=True)).tolist() == [[True, True, True]]

mean_result = backend.mean(values, dim=0, keepdim=True)
std_result = backend.std(values, dim=0, keepdim=True)
quantile_result = backend.quantile(values, 0.5, dim=0, keepdim=True)

assert backend.allclose(mean_result, backend.array([[2.5, 3.5, 4.5]]))
assert backend.allclose(std_result, backend.array([[1.5, 1.5, 1.5]]))
assert backend.allclose(quantile_result, backend.array([[2.5, 3.5, 4.5]]))

try:
    backend.max(values, axis=0, dim=1)
except TypeError as exc:
    assert "axis" in str(exc)
    assert "dim" in str(exc)
else:
    raise AssertionError("conflicting axis/dim arguments should fail")

try:
    backend.sum(values, axis=0, keepdims=True, keepdim=False)
except TypeError as exc:
    assert "keepdims" in str(exc)
    assert "keepdim" in str(exc)
else:
    raise AssertionError("conflicting keepdims/keepdim arguments should fail")

print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
