from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pyrecest.utils import plotting


class _Axes:
    def __init__(self) -> None:
        self.surface: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def plot_surface(self, x: Any, y: Any, z: Any, **_kwargs: Any) -> None:
        self.surface = (np.asarray(x), np.asarray(y), np.asarray(z))


class _Figure:
    def __init__(self, axes: _Axes) -> None:
        self.axes = axes

    def add_subplot(self, *_args: Any, **_kwargs: Any) -> _Axes:
        return self.axes


def test_plot_ellipsoid_3d_scaling_preserves_center(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    axes = _Axes()
    monkeypatch.setattr(plotting.plt, "figure", lambda: _Figure(axes))
    monkeypatch.setattr(plotting.plt, "show", lambda: None)

    plotting.plot_ellipsoid_3d(
        np.array([3.0, 4.0, 5.0]),
        np.eye(3),
        scaling_factor=2.0,
    )

    assert axes.surface is not None
    _, _, z = axes.surface
    np.testing.assert_allclose(z[:, 0], 7.0)
    np.testing.assert_allclose(z[:, -1], 3.0)
