import numpy as np
import pytest

import pyrecest.backend
from pyrecest.evaluation import summarize_filter_results


@pytest.mark.skipif(
    pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
    reason="Not supported on this backend",
)
def test_summarize_filter_results_defaults_missing_mtt_flag():
    n_runs = 2
    groundtruths = np.empty((n_runs, 1), dtype=object)
    for run in range(n_runs):
        groundtruths[run, 0] = np.array([float(run), 0.0])

    last_estimates = np.array([[[0.0, 0.0], [1.0, 0.0]]])
    runtimes = np.ones((1, n_runs))
    run_failed = np.zeros((1, n_runs), dtype=bool)
    filter_configs = [{"name": "kf", "parameter": None}]

    summarized = summarize_filter_results(
        scenario_config={"manifold": "Euclidean"},
        filter_configs=filter_configs,
        runtimes=runtimes,
        groundtruths=groundtruths,
        run_failed=run_failed,
        last_estimates=last_estimates,
    )

    assert summarized is filter_configs
    assert summarized[0]["error_mean"] == 0
    assert summarized[0]["failure_rate"] == 0
