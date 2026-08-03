"""Regressions for masked JAX linear-algebra axis arguments."""

import pytest

from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_jax_linalg_norm_rejects_masked_axes():
    pytest.importorskip("jax")

    result = run_backend_code(
        "jax",
        """
import numpy as np
import pytest
import pyrecest.backend as backend

values = backend.array([[3.0, 4.0], [5.0, 12.0]])
masked_axes = (
    np.ma.masked,
    np.ma.array(1, mask=True),
    np.ma.array([1], mask=[True]),
    [np.ma.array(1, mask=True)],
    (np.ma.array(1, mask=True),),
)

for axis in masked_axes:
    with pytest.raises(
        TypeError,
        match="axis must be None, an integer, or a tuple of integers",
    ):
        backend.linalg.norm(values, axis=axis)

unmasked_axis = [np.ma.array(1, mask=False)]
result = backend.linalg.norm(values, axis=unmasked_axis)
np.testing.assert_allclose(backend.to_numpy(result), [5.0, 13.0])
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
