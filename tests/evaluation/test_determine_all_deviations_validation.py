import numpy as np
import pyrecest.backend
import pytest
from pyrecest.evaluation.determine_all_deviations import determine_all_deviations

pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ == "jax",
    reason="determine_all_deviations is not supported on the JAX backend",
)


def test_determine_all_deviations_rejects_empty_groundtruths():
    groundtruths = np.empty((0, 0), dtype=object)

    with pytest.raises(ValueError, match="non-empty"):
        determine_all_deviations(
            [],
            None,
            lambda estimate, truth: 0.0,
            groundtruths,
        )


def test_determine_all_deviations_rejects_mismatched_result_run_count():
    groundtruths = np.empty((2, 1), dtype=object)
    groundtruths[0, 0] = np.asarray([0.0])
    groundtruths[1, 0] = np.asarray([1.0])
    results = [[np.asarray([0.0])]]

    with pytest.raises(ValueError, match="one entry per groundtruth run"):
        determine_all_deviations(
            results,
            None,
            lambda estimate, truth: float(np.linalg.norm(estimate - truth)),
            groundtruths,
        )
