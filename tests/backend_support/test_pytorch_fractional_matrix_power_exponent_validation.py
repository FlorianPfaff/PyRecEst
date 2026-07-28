import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_fractional_matrix_power_validates_exponents():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import numpy as np
import torch
from pyrecest.backend import linalg
import pyrecest._backend.pytorch.linalg as raw_linalg

matrix = [[4.0, 0.0], [0.0, 9.0]]
invalid_exponents = [
    True,
    np.bool_(False),
    np.array([0.5]),
    np.array([[0.5]]),
    np.timedelta64(2, "ns"),
    np.datetime64("1970-01-01T00:00:00.000000002"),
    np.array(np.timedelta64(2, "ns"), dtype=object),
    np.array(
        np.datetime64("1970-01-01T00:00:00.000000002"), dtype=object
    ),
    "0.5",
    0.5 + 0.0j,
    torch.tensor(True),
    torch.tensor([0.5]),
    torch.tensor(0.5 + 0.0j),
]

for exponent in invalid_exponents:
    for linalg_module in (linalg, raw_linalg):
        try:
            linalg_module.fractional_matrix_power(matrix, exponent)
        except TypeError as exc:
            assert "real scalar" in str(exc)
        else:
            raise AssertionError(
                "fractional_matrix_power accepted invalid exponent "
                f"{exponent!r} via {linalg_module!r}"
            )

nonfinite_exponents = [
    np.nan,
    np.inf,
    -np.inf,
    np.array(np.nan),
    torch.tensor(float("nan")),
    torch.tensor(float("inf")),
]
for exponent in nonfinite_exponents:
    for linalg_module in (linalg, raw_linalg):
        try:
            linalg_module.fractional_matrix_power(matrix, exponent)
        except ValueError as exc:
            assert "finite" in str(exc)
        else:
            raise AssertionError(
                "fractional_matrix_power accepted nonfinite exponent "
                f"{exponent!r} via {linalg_module!r}"
            )

expected = torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=torch.float64)
for exponent in (0.5, np.float64(0.5), np.array(0.5), torch.tensor(0.5)):
    result = raw_linalg.fractional_matrix_power(
        torch.tensor(matrix, dtype=torch.float64), exponent
    )
    assert result.dtype == torch.float64
    assert torch.allclose(result, expected, atol=1e-10, rtol=1e-10)

empty_batch = torch.empty((0, 2, 2), dtype=torch.float64)
for linalg_module in (linalg, raw_linalg):
    try:
        linalg_module.fractional_matrix_power(
            empty_batch, np.timedelta64(2, "ns")
        )
    except TypeError as exc:
        assert "real scalar" in str(exc)
    else:
        raise AssertionError("empty batches bypassed exponent validation")

print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
