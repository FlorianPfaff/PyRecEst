from pyrecest.evaluation.group_results_by_filter import group_results_by_filter


def test_group_results_by_filter_aligns_sparse_fields():
    data = [
        {
            "name": "pf",
            "parameter": 2,
            "error_mean": 0.2,
            "time_std": 0.02,
        },
        {
            "name": "pf",
            "parameter": 1,
            "error_mean": 0.1,
        },
        {
            "name": "pf",
            "parameter": 3,
            "time_std": 0.03,
        },
    ]

    grouped = group_results_by_filter(data)

    assert grouped == {
        "pf": {
            "parameter": [1, 2, 3],
            "error_mean": [0.1, 0.2, None],
            "time_std": [None, 0.02, 0.03],
        }
    }
