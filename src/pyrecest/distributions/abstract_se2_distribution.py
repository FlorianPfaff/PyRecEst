import matplotlib.pyplot as plt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    arctan2,
    array,
    column_stack,
    cos,
    hstack,
    linalg,
    linspace,
    mod,
    pi,
    sin,
    sum,
    vstack,
)

from .cart_prod.abstract_hypercylindrical_distribution import (
    AbstractHypercylindricalDistribution,
)


class AbstractSE2Distribution(AbstractHypercylindricalDistribution):
    def __init__(self):
        AbstractHypercylindricalDistribution.__init__(self, 1, 2)

    @staticmethod
    def _get_2d_axes():
        ax = plt.gca()
        if getattr(ax, "name", "") == "3d":
            fig = plt.figure()
            ax = fig.add_subplot(111)
        return ax

    @staticmethod
    def _matplotlib_color(color):
        if isinstance(color, str):
            return color
        return [float(channel) for channel in color]

    # pylint: disable=too-many-locals
    def plot_state(self, scaling_factor=1, circle_color=None, angle_color=None):
        assert (
            pyrecest.backend.__backend_name__ != "jax"
        ), "plot_state is not supported for the JAX backend."

        if circle_color is None:
            circle_color = array([0, 0.4470, 0.7410])

        if angle_color is None:
            angle_color = array([0.8500, 0.3250, 0.0980])

        circle_color = self._matplotlib_color(circle_color)
        angle_color = self._matplotlib_color(angle_color)

        linear_covmat = self.linear_covariance()
        hybrid_moment = self.hybrid_moment()
        linear_mean = hybrid_moment[2:4]
        periodic_mean = arctan2(hybrid_moment[1], hybrid_moment[0])
        periodic_var = 1 - linalg.norm(hybrid_moment[0:2])

        ax = self._get_2d_axes()
        hold_status = ax.get_label() == "hold"
        if not hold_status:
            ax.set_label("hold")

        xs = hstack((linspace(0, 2 * pi, 100), array([0.0])))
        ps = scaling_factor * linear_covmat @ vstack((cos(xs), sin(xs)))
        (h1,) = ax.plot(
            ps[0, :] + linear_mean[0], ps[1, :] + linear_mean[1], color=circle_color
        )

        plot_ang_range = 0.1 * periodic_var * pi
        xs = linspace(
            periodic_mean - plot_ang_range, periodic_mean + plot_ang_range, 100
        )
        ps = scaling_factor * linear_covmat @ vstack((cos(xs), sin(xs)))
        scaled_mean_vec = (
            scaling_factor
            * linear_covmat
            @ array([cos(periodic_mean), sin(periodic_mean)])
        )
        h2 = ax.quiver(
            linear_mean[0],
            linear_mean[1],
            scaled_mean_vec[0],
            scaled_mean_vec[1],
            scale_units="xy",
            angles="xy",
            scale=1,
            color=angle_color,
        )
        (h3,) = ax.plot(
            linear_mean[0] + ps[0, :],
            linear_mean[1] + ps[1, :],
            linestyle="--",
            color=angle_color,
        )
        (h4,) = ax.plot(
            [linear_mean[0], linear_mean[0] + ps[0, -1]],
            [linear_mean[1], linear_mean[1] + ps[1, -1]],
            linestyle="-",
            color=angle_color,
        )
        (h5,) = ax.plot(
            [linear_mean[0], linear_mean[0] + ps[0, 0]],
            [linear_mean[1], linear_mean[1] + ps[1, 0]],
            linestyle="-",
            color=angle_color,
        )

        if not hold_status:
            ax.set_label("")
            plt.show()

        return [h1, h2, h3, h4, h5]

    @staticmethod
    def angle_pos_to_dual_quaternion(x):
        rot = x[:, 0]
        trans = x[:, 1:3]
        dq_real = column_stack((cos(rot / 2), sin(rot / 2)))
        dq_tmp = column_stack((-dq_real[:, 1], dq_real[:, 0]))
        dq_dual = 0.5 * column_stack(
            (sum(dq_real * trans, axis=1), sum(dq_tmp * trans, axis=1))
        )
        dq = column_stack((dq_real, dq_dual))
        return dq

    @staticmethod
    def dual_quaternion_to_angle_pos(dq):
        angle_or_angle_pos = mod(2 * arctan2(dq[:, 1], dq[:, 0]), 2 * pi)
        q_1 = column_stack((dq[:, 0], -dq[:, 1]))
        q_2 = column_stack((dq[:, 1], dq[:, 0]))
        pos = 2 * column_stack(
            (sum(q_1 * dq[:, 2:], axis=1), sum(q_2 * dq[:, 2:], axis=1))
        )

        return angle_or_angle_pos, pos

    @staticmethod
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    def plot_trajectory(
        periodic_states,
        lin_states,
        animate=False,
        delay=0.05,
        arrow_scaling=1,
        pos_color=None,
        angle_color=None,
        fade=False,
    ):
        import numpy as _np

        lin_states = lin_states.T
        if pos_color is None:
            pos_color = [0.4660, 0.6740, 0.1880]

        if angle_color is None:
            angle_color = [0.4660, 0.6740, 0.1880]

        pos_color = AbstractSE2Distribution._matplotlib_color(pos_color)
        angle_color = AbstractSE2Distribution._matplotlib_color(angle_color)

        ax = AbstractSE2Distribution._get_2d_axes()
        hold_status = ax.get_label() == "hold"
        if not hold_status:
            ax.set_label("hold")

        if fade:
            rgbtmp = pos_color
            r_range_pos = _np.linspace(rgbtmp[0], 1, lin_states.shape[1] + 1)
            g_range_pos = _np.linspace(rgbtmp[1], 1, lin_states.shape[1] + 1)
            b_range_pos = _np.linspace(rgbtmp[2], 1, lin_states.shape[1] + 1)
            rgbtmp = angle_color
            r_range_angle = _np.linspace(rgbtmp[0], 1, lin_states.shape[1] + 1)
            g_range_angle = _np.linspace(rgbtmp[1], 1, lin_states.shape[1] + 1)
            b_range_angle = _np.linspace(rgbtmp[2], 1, lin_states.shape[1] + 1)
        else:
            r_range_pos = _np.full(lin_states.shape[1] + 1, pos_color[0])
            g_range_pos = _np.full(lin_states.shape[1] + 1, pos_color[1])
            b_range_pos = _np.full(lin_states.shape[1] + 1, pos_color[2])

            r_range_angle = _np.full(lin_states.shape[1] + 1, angle_color[0])
            g_range_angle = _np.full(lin_states.shape[1] + 1, angle_color[1])
            b_range_angle = _np.full(lin_states.shape[1] + 1, angle_color[2])
        h = []
        for i in range(lin_states.shape[1]):
            if arrow_scaling != 0:
                h_curr = ax.quiver(
                    lin_states[0, i],
                    lin_states[1, i],
                    arrow_scaling * cos(periodic_states[i]),
                    arrow_scaling * sin(periodic_states[i]),
                    scale_units="xy",
                    angles="xy",
                    scale=1,
                    color=[
                        r_range_angle[-(i + 1)],
                        g_range_angle[-(i + 1)],
                        b_range_angle[-(i + 1)],
                    ],
                )
                h.append(h_curr)
            h_curr = ax.scatter(
                lin_states[0, i],
                lin_states[1, i],
                c=[
                    [
                        r_range_pos[-(i + 1)],
                        g_range_pos[-(i + 1)],
                        b_range_pos[-(i + 1)],
                    ]
                ],
            )
            h.append(h_curr)
            plt.show()
            if animate:
                plt.pause(delay)

        return h
