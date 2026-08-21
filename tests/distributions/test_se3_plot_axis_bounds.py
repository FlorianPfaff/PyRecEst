import unittest
from unittest.mock import patch

import numpy as np
import numpy.testing as npt

from pyrecest.backend import array
from pyrecest.distributions.abstract_se3_distribution import AbstractSE3Distribution


class _RecordingAxes:
    def __init__(self, xlim, ylim, zlim):
        self._xlim = xlim
        self._ylim = ylim
        self._zlim = zlim
        self.xlim_updates = []
        self.ylim_updates = []
        self.zlim_updates = []

    @staticmethod
    def quiver(*_args, **_kwargs):
        return object()

    def get_xlim(self):
        return self._xlim

    def get_ylim(self):
        return self._ylim

    def get_zlim(self):
        return self._zlim

    def set_xlim(self, limits):
        self.xlim_updates.append(np.asarray(limits, dtype=float))

    def set_ylim(self, limits):
        self.ylim_updates.append(np.asarray(limits, dtype=float))

    def set_zlim(self, limits):
        self.zlim_updates.append(np.asarray(limits, dtype=float))


class _RecordingFigure:
    def __init__(self, axes):
        self.axes = axes

    def add_subplot(self, *_args, **_kwargs):
        return self.axes


class SE3PlotAxisBoundsTest(unittest.TestCase):
    @staticmethod
    def _plot(point, axes):
        with patch(
            "pyrecest.distributions.abstract_se3_distribution.plt.figure",
            return_value=_RecordingFigure(axes),
        ):
            AbstractSE3Distribution.plot_point(array(point))

    def test_keeps_axis_limits_that_already_contain_pose(self):
        axes = _RecordingAxes(
            xlim=(-100.0, 100.0),
            ylim=(-100.0, 100.0),
            zlim=(-100.0, 100.0),
        )

        self._plot([1.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0], axes)

        self.assertEqual(axes.xlim_updates, [])
        self.assertEqual(axes.ylim_updates, [])
        self.assertEqual(axes.zlim_updates, [])

    def test_expands_axis_when_rotated_body_axes_straddle_current_limits(self):
        axes = _RecordingAxes(
            xlim=(0.0, 1.0),
            ylim=(-100.0, 100.0),
            zlim=(-100.0, 100.0),
        )
        half_angle = np.pi / 8.0
        point = [
            np.cos(half_angle),
            0.0,
            0.0,
            np.sin(half_angle),
            0.5,
            0.5,
            0.5,
        ]

        self._plot(point, axes)

        self.assertEqual(len(axes.xlim_updates), 1)
        npt.assert_allclose(axes.xlim_updates[0], np.array([-5.0, 5.0]))
        self.assertEqual(axes.ylim_updates, [])
        self.assertEqual(axes.zlim_updates, [])


if __name__ == "__main__":
    unittest.main()
