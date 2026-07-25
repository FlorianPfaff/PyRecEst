import numpy as np
import pyrecest.smoothers.abstract_smoother as abstract_smoother_module
from pyrecest.smoothers.abstract_smoother import AbstractSmoother


def test_matrix_sequence_falls_back_after_backend_runtime_error(monkeypatch):
    matrices = [np.eye(2), 2.0 * np.eye(2)]
    backend_asarray = abstract_smoother_module.asarray

    def asarray_with_outer_conversion_failure(value):
        if value is matrices:
            raise RuntimeError("backend cannot stack the matrix sequence")
        return backend_asarray(value)

    monkeypatch.setattr(
        abstract_smoother_module,
        "asarray",
        asarray_with_outer_conversion_failure,
    )

    normalized = AbstractSmoother._normalize_matrix_sequence(
        matrices,
        length=2,
        name="transition_matrices",
        matrix_dim=2,
    )

    np.testing.assert_allclose(normalized[0], matrices[0])
    np.testing.assert_allclose(normalized[1], matrices[1])
