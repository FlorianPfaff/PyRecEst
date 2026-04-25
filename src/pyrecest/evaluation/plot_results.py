import warnings

import matplotlib.pyplot as plt
from beartype import beartype
from numpy import load
from pyrecest.backend import any, asarray, isnan, shape, size, zeros

from .get_axis_label import get_axis_label
from .group_results_by_filter import group_results_by_filter
from .summarize_filter_results import summarize_filter_results


# pylint: disable=too-many-branches,too-many-locals,too-many-statements
def plot_results(
    filename=None, plot_log=False, plot_stds: bool = False, omit_slow: bool = False
):
    """
    Plot results of filter evaluations.

    Args:
        filenames (str or list): File names to load results from.
        plot_log (np.array of bool): Whether to plot in log scale for each axis.
        plot_stds (bool): Whether to plot standard deviations.
        omit_slow (bool): Whether to omit slow filters from plotting.

    Returns:
        param_time_and_error_per_filter (dict): Dictionary containing performance statistics.
    """
    not_all_warnings_shown = warnings.filters != "always"

    if not_all_warnings_shown:
        print("Not all warnings are enabled.")

    # Expand plot_log to handle all plots (pad it)
    if size(plot_log) == 1:
        plot_log = zeros((2, 3), dtype=bool)
    else:
        assert shape(plot_log) == (2, 3)
    plot_random_filter = True

    if filename is None:
        # Prompt for filename
        filename = input("Please enter the filename to load: ")

    data = load(filename, allow_pickle=True).item()
    # Get the mean errors and mean times for each filter configuration
    results_summarized = summarize_filter_results(**data)
    [min_param, max_param] = get_min_max_param(results_summarized)
    # Group the configurations, the mean errors and mean times for each filter
    results_grouped = group_results_by_filter(results_summarized)

    if not plot_random_filter:
        if "random" in results_grouped:
            del results_grouped["random"]

    # To convert to ms per time step
    times_factor = 1000 / data["groundtruths"].shape[1]
    state_dim = data["scenario_config"]["initial_prior"].dim
    # Initialize plots and axis
    figs = []
    axes_list = []
    for i in range(3):
        fig = plt.figure(i)
        ax = fig.add_subplot(1, 1, 1)
        figs.append(fig)
        axes_list.append(ax)

    for curr_filter_name in results_grouped.keys():
        # Iterate over all possible names and plot the lines for those that were evaluated
        color, style_marker, style_line = get_plot_style_for_filter(curr_filter_name)

        params = asarray(results_grouped[curr_filter_name]["parameter"])
        errors_mean = asarray(results_grouped[curr_filter_name]["error_mean"])
        errors_std = asarray(results_grouped[curr_filter_name]["error_std"])
        times_mean = asarray(results_grouped[curr_filter_name]["time_mean"])

        if curr_filter_name.startswith("ff") or curr_filter_name == "htgf":
            params = params**state_dim
        elif "shf" in curr_filter_name:
            params = (params + 1) ** 2

        # Plot errors
        plt.figure(0)
        if params[0] is not None and not any(isnan(params)) and size(params) > 1:
            if plot_stds:
                plt.plot(
                    params,
                    errors_mean - errors_std * 0.05,
                    color=color,
                    label=long_name_to_short_name(curr_filter_name) + " - std",
                )
                plt.plot(
                    params,
                    errors_mean + errors_std * 0.05,
                    color=color,
                    label=long_name_to_short_name(curr_filter_name) + " + std",
                )

            plt.plot(
                params,
                errors_mean,
                style_marker + style_line,
                color=color,
                label=long_name_to_short_name(curr_filter_name),
            )
        else:  # If not parametric like wn or twn filter
            plt.plot(
                [min_param, max_param],
                [errors_mean, errors_mean],
                style_line,
                color=color,
                label=long_name_to_short_name(curr_filter_name),
            )
        plt.xlabel("Number of grid points/particles/coefficients")
        plt.ylabel(get_axis_label(data["scenario_config"]["manifold"]))

        # Plot times
        plt.figure(1)
        if params[0] is not None and not any(isnan(params)) and size(params) > 1:
            plt.plot(
                params,
                times_factor * times_mean,
                style_marker + style_line,
                color=color,
                label=long_name_to_short_name(curr_filter_name),
            )
        else:
            if omit_slow:
                raise NotImplementedError(
                    "Omitting slow filters is not implemented yet."
                )

            plt.plot(
                [min_param, max_param],
                [times_mean, times_mean],
                style_line,
                color=color,
                label=long_name_to_short_name(curr_filter_name),
            )
        plt.xlabel("Number of grid points/particles/coefficients")
        plt.ylabel("Time taken in ms per time step")

        # Plot errors over time
        plt.figure(2)
        if params[0] is not None and not any(isnan(params)) and size(params) > 1:
            plt.plot(
                times_factor * times_mean,
                errors_mean,
                style_marker + style_line,
                color=color,
                label=long_name_to_short_name(curr_filter_name),
            )
        else:
            if omit_slow:
                raise NotImplementedError(
                    "Omitting slow filters is not implemented yet."
                )

            plt.plot(
                times_mean,
                errors_mean,
                style_marker,
                color=color,
                label=long_name_to_short_name(curr_filter_name),
            )
        plt.xlabel("Time taken in ms per time step")
        plt.ylabel(get_axis_label(data["scenario_config"]["manifold"]))

    if not_all_warnings_shown:
        print("-----------Reminder: Not all warnings were enabled.-----------")

    plt.show()
    # Apply log scales (if chosen to do so)
    apply_log_scale_to_axes(axes_list, plot_log)
    return figs, axes_list


def get_plot_style_for_filter(filter_name):
    """
    Get plot style for a given filter name.

    Args:
        filter_name (str): Name of the filter.

    Returns:
        color (str or list): Color for the plot.
        style_marker (str): Marker style for the plot.
        style_line (str): Line style for the plot.
    """

    switcher = {
        "iff": ([0.4660, 0.6740, 0.1880], "o", "--"),
        "ishf": ([0.4660, 0.6740, 0.1880], "o", "--"),
        "discrete": ([0.8500, 0.3250, 0.0980], "", "-"),
        "bingham": ([0.8500, 0.3250, 0.0980], "", "-"),
        "sqff": ([0, 0.4470, 0.7410], "d", "-"),
        "sqshf": ([0, 0.4470, 0.7410], "d", "-"),
        "pf": ([0.9290, 0.6940, 0.1250], "p", "-."),
        "htpf": ([0.9290, 0.6940, 0.1250], "p", "-."),
        "wn": ([0.4940, 0.1840, 0.5560], "d", "-"),
        "vm": ([0.4940, 0.1840, 0.5560], "d", "-"),
        "vmf": ([0.4940, 0.1840, 0.5560], "d", "-"),
        "hhgf": ([0.4940, 0.1840, 0.5560], "d", "-"),
        "se2bf": ([0.4940, 0.1840, 0.5560], "d", "-"),
        "random": ([0, 0, 0], "x", "-"),
        "dummy": ([0, 0, 0], "x", "-"),
        "kf": ([0.3010, 0.7450, 0.9330], "*", "-"),
        "se2iukf": ([0.3010, 0.7450, 0.9330], "*", "-"),
        "hgfSymm": ([0.3010, 0.7450, 0.9330], "", "-"),
        "figResetOnPred": ([0.3010, 0.7450, 0.9330], "", "-"),
        "sgf": ([0.6350, 0.0780, 0.1840], "*", "-"),
        "hgf": ([0.6350, 0.0780, 0.1840], "*", "-"),
        "s3f": ([0.6350, 0.0780, 0.1840], "*", "-"),
        "fig": ([0.6350, 0.0780, 0.1840], "*", "-"),
        "htgf": ([0.6350, 0.0780, 0.1840], "*", "-."),
    }

    color, style_marker, style_line = switcher.get(filter_name, ("k", "", "-"))

    return color, style_marker, style_line


@beartype
def long_name_to_short_name(short_name: str) -> str:
    """Get short name from long name."""

    switcher = {
        "se2ukfm": "Unscented KF for Manifolds",
        "se2bf": "(Progressive) SE(2) Bingham filter",
        "s3f": "State space subdivision filter",
        "grid": "Grid filter",
        "iff": "Fourier identity filter",
        "sqff": "Fourier square root filter",
        "pf": "Particle filter",
        "htpf": "Particle filter",
        "vmf": "Von Mises--Fisher filter",
        "bingham": "Bingham filter",
        "wn": "Wrapped normal filter",
        "vm": "Von Mises filter",
        "twn": "Bivariate WN filter",
        "kf": "Kalman filter",
        "ishf": "Spherical hamonics identity filter",
        "sqshf": "Spherical hamonics square root filter",
        "htgf": "Hypertoroidal grid filter",
        "sgf": "Spherical grid filter",
        "hgf": "Hyperspherical grid filter",
        "hgfSymm": "Symmetric hyperspherical grid filter",
        "hhgf": "Hyperhemispherical grid filter",
        "randomTorus": "Random filter",
        "randomSphere": "Random filter",
        "fig": "Fourier-interpreted grid filter",
        "figResetOnPred": "FIG-Filter with resetting on prediction",
        "gnn": "Global Nearest Neighbor",
        "GNN": "Global Nearest Neighbor",
    }
    return switcher.get(short_name, short_name)


def get_min_max_param(results_summarized):
    """Get min and max parameter value."""

    valid_parameters = [
        res["parameter"] for res in results_summarized if res["parameter"] is not None
    ]
    # Finding min and max values
    min_parameter = min(valid_parameters)
    max_parameter = max(valid_parameters)
    return min_parameter, max_parameter


def apply_log_scale_to_axes(axes_list, log_array):
    # Ensure the provided array has the correct shape
    assert log_array.shape == (2, 3), "Invalid shape for log_array"
    assert len(axes_list) == 3, "Expecting 3 axes for 3 plots"

    for plot_idx, ax in enumerate(axes_list):
        if log_array[0, plot_idx]:  # Check if x-axis should be log scale
            ax.set_xscale("log")

        if log_array[1, plot_idx]:  # Check if y-axis should be log scale
            ax.set_yscale("log")
