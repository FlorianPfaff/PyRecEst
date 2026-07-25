#!/usr/bin/env python3
"""Benchmarks for low-rank hypertoroidal Fourier filtering."""

from __future__ import annotations

import argparse
import json
from math import prod
from pathlib import Path
from time import perf_counter

import numpy as np
from pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution import (
    HypertoroidalFourierDistribution,
)
from pyrecest.distributions.hypertorus.low_rank_hypertoroidal_fourier_distribution import (
    LowRankHypertoroidalFourierDistribution,
)
from pyrecest.filters.low_rank_hypertoroidal_fourier_filter import (
    LowRankHypertoroidalFourierFilter,
)


def _one_dimensional_identity_coefficients(axis_size: int, scale: float) -> np.ndarray:
    center = axis_size // 2
    coeff = np.zeros(axis_size, dtype=np.complex128)
    coeff[center] = 1.0 / (2.0 * np.pi)
    if axis_size > 1:
        coeff[center - 1] = scale * (0.01 + 0.02j)
        coeff[center + 1] = np.conjugate(coeff[center - 1])
    return coeff


def _separable_identity_coefficients(
    dim: int, axis_size: int, scale: float
) -> np.ndarray:
    coeff_1d = _one_dimensional_identity_coefficients(axis_size, scale)
    coeff = coeff_1d
    for _ in range(dim - 1):
        coeff = np.multiply.outer(coeff, coeff_1d)
    return coeff


def _time_call(function, repeat: int) -> float:
    start = perf_counter()
    for _ in range(repeat):
        function()
    return perf_counter() - start


def _storage_row(dim: int, axis_size: int, rank: int) -> dict[str, object]:
    shape = (axis_size,) * dim
    dense_entries = prod(shape)
    tt_storage = axis_size
    if dim > 1:
        tt_storage += (dim - 2) * rank * axis_size * rank
        tt_storage += rank * axis_size
    return {
        "dim": dim,
        "axis_size": axis_size,
        "nominal_rank": rank,
        "dense_entries": dense_entries,
        "tt_storage_entries": int(tt_storage),
        "dense_to_tt_storage_ratio": dense_entries / float(tt_storage),
    }


def run_storage_scaling(
    *, axis_size: int, rank: int, dimensions: tuple[int, ...]
) -> dict[str, object]:
    return {
        "name": "low_rank_hypertoroidal_fourier_storage_scaling",
        "axis_size": axis_size,
        "nominal_rank": rank,
        "rows": [_storage_row(dim, axis_size, rank) for dim in dimensions],
    }


def run_low_rank_filter(
    *, dim: int, axis_size: int, max_rank: int, iterations: int
) -> dict[str, object]:
    shape = (axis_size,) * dim
    prior = HypertoroidalFourierDistribution(
        _separable_identity_coefficients(dim, axis_size, scale=1.0), "identity"
    )
    noise = LowRankHypertoroidalFourierDistribution.from_dense(
        HypertoroidalFourierDistribution(
            _separable_identity_coefficients(dim, axis_size, scale=0.5), "identity"
        ),
        max_rank=max_rank,
        rtol=0.0,
        atol=0.0,
    )
    measurement = np.zeros(dim)
    query_point = np.full(dim, 0.1)

    filt = LowRankHypertoroidalFourierFilter(
        shape, "identity", max_rank=max_rank, rtol=0.0, atol=0.0
    )
    filt.filter_state = prior

    predict_seconds = _time_call(lambda: filt.predict_identity(noise), iterations)
    update_seconds = _time_call(
        lambda: filt.update_identity(noise, measurement), iterations
    )
    pdf_seconds = _time_call(lambda: filt.filter_state.pdf(query_point), iterations)
    shift_seconds = _time_call(lambda: filt.filter_state.shift(measurement), iterations)

    ranks = filt.filter_state.tt_ranks
    return {
        "name": "low_rank_hypertoroidal_fourier_filter",
        "dim": dim,
        "axis_size": axis_size,
        "iterations": iterations,
        "max_rank": max_rank,
        "tt_ranks": list(ranks),
        "max_observed_rank": max(ranks),
        "dense_entries": prod(shape),
        "tt_storage_entries": filt.filter_state.coefficients.storage_size,
        "integral": filt.filter_state.integrate(),
        "timings_seconds": {
            "predict_identity": predict_seconds,
            "update_identity": update_seconds,
            "pdf": pdf_seconds,
            "shift": shift_seconds,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis-size", type=int, default=3)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = {
        "benchmarks": [
            run_storage_scaling(
                axis_size=args.axis_size,
                rank=args.rank,
                dimensions=(2, 4, 8, 12),
            ),
            run_low_rank_filter(
                dim=6,
                axis_size=args.axis_size,
                max_rank=args.rank,
                iterations=args.iterations,
            ),
        ]
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
