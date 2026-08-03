import matplotlib.pyplot as plt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    array,
    column_stack,
    cos,
    linalg,
    linspace,
    ones,
    outer,
    pi,
    reshape,
    sin,
    sqrt,
)


def plot_ellipsoid(center, shape_matrix, scaling_factor=1, color="blue"):
    if center.shape[0] == 2:
        plot_ellipsoid_2d(center, shape_matrix, scaling_factor, color)
    elif center.shape[0] == 3:
        plot_ellipsoid_3d(center, shape_matrix, scaling_factor, color)
    else:
        raise ValueError("Only 2D and 3D ellipsoids are supported.")


def plot_ellipsoid_2d(center, shape_matrix, scaling_factor=1, color="blue"):
    xs = linspace(0, 2 * pi, 100)
    ps = scaling_factor * shape_matrix @ column_stack((cos(xs), sin(xs))).T
    plt.plot(ps[0] + center[0], ps[1] + center[1], color=color)
    plt.show()


def plot_ellipsoid_3d(center, shape_matrix, scaling_factor=1, color="blue"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    u = linspace(0, 2 * pi, 100)
    v = linspace(0, pi, 100)
    x = outer(cos(u), sin(v))
    y = outer(sin(u), sin(v))
    z = outer(ones(u.shape[0]), cos(v))

    D, V = linalg.eigh(shape_matrix)
    all_coords = scaling_factor * (V * sqrt(D)) @ array(
        [x.ravel(), y.ravel(), z.ravel()]
    ) + center.reshape(-1, 1)
    x = reshape(all_coords[0], x.shape)
    y = reshape(all_coords[1], y.shape)
    z = reshape(all_coords[2], z.shape)

    ax.plot_surface(
        x,
        y,
        z,
        color=color,
        alpha=0.7,
        linewidth=0,
        antialiased=False,
    )
    plt.show()
