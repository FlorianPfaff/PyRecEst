import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pyrecest.backend
import pytest
from pyrecest.backend import array
from pyrecest.distributions.abstract_se2_distribution import AbstractSE2Distribution

plt.switch_backend("Agg")


class _PlotStateFixture:
    _get_2d_axes = staticmethod(AbstractSE2Distribution._get_2d_axes)
    _matplotlib_color = staticmethod(AbstractSE2Distribution._matplotlib_color)
    plot_state = AbstractSE2Distribution.plot_state

    @staticmethod
    def linear_covariance():
        return array([[4.0, 0.0], [0.0, 9.0]])

    @staticmethod
    def hybrid_moment():
        return array([1.0, 0.0, 1.5, -2.0])


@pytest.mark.skipif(
    pyrecest.backend.__backend_name__ == "jax",
    reason="plot_state is not supported for the JAX backend",
)
def test_plot_state_uses_standard_deviation_axes():
    fig, ax = plt.subplots()
    plt.sca(ax)
    try:
        contour, *_ = _PlotStateFixture().plot_state()
        x_data = np.asarray(contour.get_xdata(), dtype=float)
        y_data = np.asarray(contour.get_ydata(), dtype=float)

        npt.assert_allclose(x_data[0] - 1.5, 2.0, atol=1e-10)
        npt.assert_allclose(y_data[0] + 2.0, 0.0, atol=1e-10)
    finally:
        plt.close(fig)
