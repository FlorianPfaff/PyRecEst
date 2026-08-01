from __future__ import annotations

from pathlib import Path

SOURCE_PATH = Path("src/pyrecest/calibration/time_offset.py")
TEST_PATH = Path("tests/calibration/test_time_offset_extreme_metrics.py")
WORKFLOW_PATH = Path(
    ".github/workflows/one-shot-time-offset-extreme-metrics.yml"
)
SCRIPT_PATH = Path("tools/apply_time_offset_extreme_metrics_patch.py")


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(
            f"expected exactly one replacement target, found {text.count(old)}"
        )
    return text.replace(old, new, 1)


def main() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    source = replace_once(
        source,
        '''    errors = np.linalg.norm(\n        measurement_values[valid] - reference_at_query[valid], axis=1\n    )\n''',
        '''    errors = np.hypot.reduce(\n        measurement_values[valid] - reference_at_query[valid],\n        axis=1,\n        initial=0.0,\n    )\n''',
    )
    source = replace_once(
        source,
        '''def _aggregate_summary_metric(\n    key: str, values: np.ndarray, counts: np.ndarray\n) -> float:\n    valid = np.isfinite(values) & (counts > 0.0)\n    if not valid.any():\n        return float("nan")\n    if key == "rmse":\n        return float(np.sqrt(np.average(values[valid] ** 2, weights=counts[valid])))\n    if key == "max":\n        return float(np.max(values[valid]))\n    return float(np.average(values[valid], weights=counts[valid]))\n\n\ndef _aggregate_std_metric(\n    stds: np.ndarray, means: np.ndarray, counts: np.ndarray\n) -> float:\n    valid = np.isfinite(stds) & np.isfinite(means) & (counts > 0.0)\n    if not valid.any():\n        return float("nan")\n    weights = counts[valid]\n    pooled_mean = float(np.average(means[valid], weights=weights))\n    second_moment = float(\n        np.average(stds[valid] ** 2 + means[valid] ** 2, weights=weights)\n    )\n    return float(np.sqrt(max(0.0, second_moment - pooled_mean**2)))\n''',
        '''def _stable_weighted_mean(\n    values: np.ndarray, weights: np.ndarray | None = None\n) -> float:\n    values = np.asarray(values, dtype=float).reshape(-1)\n    scale = float(np.max(np.abs(values)))\n    if scale == 0.0:\n        return 0.0\n    scaled_values = values / scale\n    if weights is None:\n        mean = float(np.mean(scaled_values))\n    else:\n        weights = np.asarray(weights, dtype=float).reshape(-1)\n        weight_scale = float(np.max(weights))\n        scaled_weights = weights / weight_scale\n        mean = float(np.average(scaled_values, weights=scaled_weights))\n    return float(scale * mean)\n\n\ndef _stable_root_mean_square(\n    values: np.ndarray, weights: np.ndarray | None = None\n) -> float:\n    values = np.asarray(values, dtype=float).reshape(-1)\n    scale = float(np.max(np.abs(values)))\n    if scale == 0.0:\n        return 0.0\n    squared = (values / scale) ** 2\n    if weights is None:\n        mean_square = float(np.mean(squared))\n    else:\n        weights = np.asarray(weights, dtype=float).reshape(-1)\n        weight_scale = float(np.max(weights))\n        scaled_weights = weights / weight_scale\n        mean_square = float(np.average(squared, weights=scaled_weights))\n    return float(scale * np.sqrt(mean_square))\n\n\ndef _stable_standard_deviation(values: np.ndarray) -> float:\n    values = np.asarray(values, dtype=float).reshape(-1)\n    scale = float(np.max(np.abs(values)))\n    if scale == 0.0:\n        return 0.0\n    return float(scale * np.std(values / scale))\n\n\ndef _aggregate_summary_metric(\n    key: str, values: np.ndarray, counts: np.ndarray\n) -> float:\n    valid = np.isfinite(values) & (counts > 0.0)\n    if not valid.any():\n        return float("nan")\n    if key == "rmse":\n        return _stable_root_mean_square(values[valid], counts[valid])\n    if key == "max":\n        return float(np.max(values[valid]))\n    return _stable_weighted_mean(values[valid], counts[valid])\n\n\ndef _aggregate_std_metric(\n    stds: np.ndarray, means: np.ndarray, counts: np.ndarray\n) -> float:\n    valid = np.isfinite(stds) & np.isfinite(means) & (counts > 0.0)\n    if not valid.any():\n        return float("nan")\n    stds = stds[valid]\n    means = means[valid]\n    weights = counts[valid]\n    scale = float(max(np.max(np.abs(stds)), np.max(np.abs(means))))\n    if scale == 0.0:\n        return 0.0\n    scaled_stds = stds / scale\n    scaled_means = means / scale\n    pooled_mean = _stable_weighted_mean(scaled_means, weights)\n    pooled_variance = _stable_weighted_mean(\n        scaled_stds**2 + (scaled_means - pooled_mean) ** 2,\n        weights,\n    )\n    return float(scale * np.sqrt(max(0.0, pooled_variance)))\n''',
    )
    source = replace_once(
        source,
        '''        "mean": float(np.mean(errors)),\n        "std": float(np.std(errors)),\n        "rmse": float(np.sqrt(np.mean(errors**2))),\n''',
        '''        "mean": _stable_weighted_mean(errors),\n        "std": _stable_standard_deviation(errors),\n        "rmse": _stable_root_mean_square(errors),\n''',
    )
    SOURCE_PATH.write_text(source, encoding="utf-8")

    TEST_PATH.write_text(
        '''import unittest\n\nimport numpy as np\nimport numpy.testing as npt\nfrom pyrecest.calibration.time_offset import (\n    aggregate_time_offset_sweeps,\n    time_offset_error_summary,\n)\n\n\nclass TimeOffsetExtremeMetricStabilityTest(unittest.TestCase):\n    def test_summary_preserves_extreme_finite_residuals(self):\n        expected = np.hypot(1e308, 1e308)\n\n        with np.errstate(over="raise", invalid="raise"):\n            summary = time_offset_error_summary(\n                np.array([0.0, 1.0]),\n                np.array([[1e308, 1e308], [1e308, 1e308]]),\n                np.array([0.0, 1.0]),\n                np.zeros((2, 2)),\n                0.0,\n            )\n\n        self.assertEqual(summary["count"], 2.0)\n        self.assertEqual(summary["coverage"], 1.0)\n        npt.assert_allclose(summary["mean"], expected, rtol=1e-15)\n        npt.assert_allclose(summary["std"], 0.0, atol=0.0)\n        npt.assert_allclose(summary["rmse"], expected, rtol=1e-15)\n        npt.assert_allclose(summary["p95"], expected, rtol=1e-15)\n        npt.assert_allclose(summary["max"], expected, rtol=1e-15)\n\n    def test_aggregation_preserves_extreme_finite_metrics(self):\n        summary = {\n            "time_offset_s": 0.0,\n            "count": 2.0,\n            "mean": 1e308,\n            "std": 0.0,\n            "rmse": 1e308,\n            "p95": 1e308,\n            "max": 1e308,\n        }\n\n        with np.errstate(over="raise", invalid="raise"):\n            aggregated = aggregate_time_offset_sweeps([[summary]])\n\n        self.assertEqual(len(aggregated), 1)\n        self.assertEqual(aggregated[0]["count"], 2.0)\n        npt.assert_allclose(aggregated[0]["mean"], 1e308, rtol=1e-15)\n        npt.assert_allclose(aggregated[0]["std"], 0.0, atol=0.0)\n        npt.assert_allclose(aggregated[0]["rmse"], 1e308, rtol=1e-15)\n        npt.assert_allclose(aggregated[0]["p95"], 1e308, rtol=1e-15)\n        npt.assert_allclose(aggregated[0]["max"], 1e308, rtol=1e-15)\n\n\nif __name__ == "__main__":\n    unittest.main()\n''',
        encoding="utf-8",
    )

    WORKFLOW_PATH.unlink(missing_ok=True)
    SCRIPT_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
