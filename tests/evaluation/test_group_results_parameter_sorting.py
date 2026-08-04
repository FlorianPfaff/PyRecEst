"""Regression tests for robust result-parameter sorting."""

import warnings

import numpy as np

from pyrecest.evaluation.group_results_by_filter import group_results_by_filter


def test_masked_and_nonscalar_parameters_sort_without_conversion_warnings():
    data = [
        {"name": "pf", "parameter": None, "label": "none"},
        {
            "name": "pf",
            "parameter": np.ma.array(99.0, mask=True),
            "label": "masked",
        },
        {"name": "pf", "parameter": np.array([1.0]), "label": "vector"},
        {
            "name": "pf",
            "parameter": np.ma.array(0.25, mask=False),
            "label": "clear-mask",
        },
        {"name": "pf", "parameter": 0.5, "label": "scalar"},
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        grouped = group_results_by_filter(data)

    labels = grouped["pf"]["label"]
    assert labels[:3] == ["none", "clear-mask", "scalar"]
    assert set(labels[3:]) == {"masked", "vector"}
