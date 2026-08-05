import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_fractional_matrix_power_rejects_masked_exponents():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import numpy as np
from pyrecest.backend import linalg
import pyrecest._backend.pytorch.linalg as raw_linalg

matrix = [[4.0, 0.0], [0.0, 9.0]]
for exponent in (np.ma.array(0.5, mask=True), np.ma.masked):
    for linalg_module in (linalg, raw_linalg):
        try:
            linalg_module.fractional_matrix_power(matrix, exponent)
        except TypeError as exc:
            assert "real scalar" in str(exc)
        else:
            raise AssertionError(
                "fractional_matrix_power accepted masked exponent "
                f"{exponent!r} via {linalg_module!r}"
            )

print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
