import numpy as np

from pyrecest.evaluation.diagnostic_summaries import worst_time_windows


def test_worst_time_windows_stays_finite_for_large_finite_errors():
    records = [
        {"time_s": 0.0, "error": -1.0e308},
        {"time_s": 1.0, "error": 1.0e308},
    ]

    with np.errstate(over="raise", invalid="raise"):
        rows = worst_time_windows(records, window_s=5.0, top_n=1)

    assert len(rows) == 1
    row = rows[0]
    assert np.isfinite(row["rmse"])
    assert np.isfinite(row["mae"])
    assert np.isfinite(row["p95"])
    np.testing.assert_allclose(row["rmse"], 1.0e308, rtol=1.0e-15)
    np.testing.assert_allclose(row["mae"], 1.0e308, rtol=1.0e-15)
    np.testing.assert_allclose(row["p95"], 9.0e307, rtol=1.0e-15)
