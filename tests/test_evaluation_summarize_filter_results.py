import unittest
import warnings
from unittest.mock import patch

import numpy as np

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.evaluation import summarize_filter_results


class TestSummarizeFilterResultsWarnings(unittest.TestCase):
    def test_rejects_jax_backend_explicitly(self):
        with patch.object(pyrecest.backend, "__backend_name__", "jax"):
            with self.assertRaisesRegex(NotImplementedError, "JAX"):
                summarize_filter_results(
                    scenario_config={},
                    filter_configs=[],
                    runtimes=np.empty((0, 0)),
                    groundtruths=np.empty((0, 0), dtype=object),
                    run_failed=np.empty((0, 0), dtype=bool),
                    last_estimates=np.empty((0, 0), dtype=object),
                )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_accepts_last_estimates_without_last_filter_states(self):
        groundtruths = np.empty((1, 1), dtype=object)
        groundtruths[0, 0] = np.array([1.0, 2.0])
        last_estimates = np.array([[[1.0, 4.0]]])
        filter_configs = [{"name": "estimate-only"}]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            summarized = summarize_filter_results(
                scenario_config={"manifold": "Euclidean", "mtt": False},
                filter_configs=filter_configs,
                runtimes=np.array([[0.25]]),
                groundtruths=groundtruths,
                run_failed=np.array([[False]]),
                last_estimates=last_estimates,
            )

        self.assertIs(summarized, filter_configs)
        self.assertAlmostEqual(summarized[0]["error_mean"], 2.0)
        self.assertAlmostEqual(summarized[0]["error_std"], 0.0)
        self.assertAlmostEqual(summarized[0]["time_mean"], 0.25)
        self.assertAlmostEqual(summarized[0]["failure_rate"], 0.0)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_run_count_warning_uses_run_axis(self):
        n_runs = 1000
        n_timesteps = 2
        groundtruths = np.empty((n_runs, n_timesteps), dtype=object)
        for index in np.ndindex(groundtruths.shape):
            groundtruths[index] = np.zeros(2)

        last_filter_states = np.zeros((1, n_runs, 2))
        runtimes = np.ones((1, n_runs))
        run_failed = np.zeros((1, n_runs), dtype=bool)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            summarize_filter_results(
                scenario_config={"manifold": "Euclidean", "mtt": False},
                filter_configs=[{"name": "kf", "parameter": None}],
                runtimes=runtimes,
                groundtruths=groundtruths,
                run_failed=run_failed,
                last_filter_states=last_filter_states,
            )

        warning_messages = [str(warning.message) for warning in caught]
        self.assertFalse(
            any("Using less than 1000 runs" in message for message in warning_messages)
        )


if __name__ == "__main__":
    unittest.main()
