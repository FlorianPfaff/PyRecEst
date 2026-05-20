#!/usr/bin/env python3
"""Small deterministic benchmark/regression scenarios for PyRecEst."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from pyrecest.backend import array, diag, to_numpy
from pyrecest.filters import KalmanFilter


def _jsonable(value):
    try:
        value = to_numpy(value)
    except Exception:  # pragma: no cover - backend fallback
        pass
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def run_linear_kalman(iterations: int = 200) -> dict[str, object]:
    """Run a deterministic one-dimensional constant-velocity Kalman scenario."""
    dt = 1.0
    system_matrix = array([[1.0, dt], [0.0, 1.0]])
    measurement_matrix = array([[1.0, 0.0]])
    system_noise_cov = diag(array([0.05, 0.01]))
    measurement_noise_cov = array([[0.25]])

    kalman_filter = KalmanFilter((array([0.0, 1.0]), diag(array([1.0, 1.0]))))

    start = perf_counter()
    for step in range(iterations):
        measurement = array([float(step + 1)])
        kalman_filter.predict_linear(system_matrix, system_noise_cov)
        kalman_filter.update_linear(
            measurement, measurement_matrix, measurement_noise_cov
        )
    elapsed = perf_counter() - start

    return {
        "name": "linear_kalman",
        "iterations": iterations,
        "elapsed_seconds": elapsed,
        "final_estimate": _jsonable(kalman_filter.get_point_estimate()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = {"benchmarks": [run_linear_kalman(args.iterations)]}
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
