import numpy as np
import pytest
from pyrecest.utils.track_completion import enumerate_fragment_completion_paths


def candidates(*_args):
    return [1]


def test_max_path_length_rejects_fractional_values():
    with pytest.raises(ValueError, match="max_path_length"):
        enumerate_fragment_completion_paths(
            [[0, None, None]],
            direction="suffix",
            candidate_provider=candidates,
            max_path_length=1.5,
        )


def test_max_paths_per_fragment_rejects_fractional_values():
    with pytest.raises(ValueError, match="max_paths_per_fragment"):
        enumerate_fragment_completion_paths(
            [[0, None, None]],
            direction="suffix",
            candidate_provider=candidates,
            max_paths_per_fragment=np.array(1.5),
        )


def test_integer_like_limits_are_accepted():
    paths = enumerate_fragment_completion_paths(
        [[0, None, None]],
        direction="suffix",
        candidate_provider=candidates,
        max_path_length=np.array(1.0),
        max_paths_per_fragment=np.float64(1.0),
    )
    assert len(paths) == 1
