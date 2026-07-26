import numpy as np
import pyrecest.backend
import pytest
from pyrecest.backend import array, isinf
from pyrecest.evaluation.determine_all_deviations import determine_all_deviations

pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ == "jax",
    reason="determine_all_deviations is not supported on the JAX backend",
)


def test_nonfinite_distance_is_treated_as_failed_run():
    groundtruths = np.empty((1, 1), dtype=object)
    groundtruths[0, 0] = array([0.0])
    results = [[array([0.0])]]

    with pytest.warns(UserWarning, match="apparently failed 1 times"):
        deviations = determine_all_deviations(
            results,
            None,
            lambda estimate, expected: float("nan"),
            groundtruths,
        )

    assert bool(isinf(deviations[0, 0]))
