import numpy as np
import pytest
from pyrecest.backend_support._pytorch_sort_numpy_contract import (
    resolve_sort_stability,
    sort_axis_none,
)


class _SortResult:
    def __init__(self, values):
        self.values = values


class _FakeBackend:
    @staticmethod
    def array(values):
        return np.asarray(values)


class _FakeTorch:
    bool = np.bool_
    last_options = None

    @staticmethod
    def is_tensor(_value):
        return False

    @staticmethod
    def flatten(values):
        return np.asarray(values).reshape(-1)

    @classmethod
    def sort(cls, values, *, dim, descending, stable):
        cls.last_options = {
            "dim": dim,
            "descending": descending,
            "stable": stable,
        }
        ordered = np.sort(np.asarray(values), axis=dim)
        if descending:
            ordered = np.flip(ordered, axis=dim)
        return _SortResult(ordered)


@pytest.mark.parametrize("stable", ["false", "true", 0, 1, np.array(True)])
def test_sort_stability_rejects_nonboolean_values(stable):
    with pytest.raises(TypeError, match="stable must be a boolean"):
        resolve_sort_stability(None, stable)


@pytest.mark.parametrize("descending", ["false", "true", 0, 1, np.array(False)])
def test_sort_rejects_nonboolean_descending_values(descending):
    with pytest.raises(TypeError, match="descending must be a boolean"):
        sort_axis_none(
            _FakeBackend,
            _FakeTorch,
            [2, 1],
            axis=None,
            descending=descending,
        )


def test_sort_accepts_numpy_boolean_options_without_truthiness_coercion():
    result = sort_axis_none(
        _FakeBackend,
        _FakeTorch,
        [2, 1],
        axis=None,
        stable=np.bool_(False),
        descending=np.bool_(True),
    )

    assert result.tolist() == [2, 1]
    assert _FakeTorch.last_options == {
        "dim": 0,
        "descending": True,
        "stable": False,
    }
