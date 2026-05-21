import time
import warnings

import numpy as _np
from pyrecest.backend import any, array, atleast_2d, empty_like, squeeze

from .configure_for_filter import configure_for_filter


def _empty_estimates_like(groundtruth):
    """Create an estimate container matching dense or object groundtruth arrays."""
    if isinstance(groundtruth, _np.ndarray) and groundtruth.dtype == object:
        return _np.empty_like(groundtruth)
    return empty_like(groundtruth)


def _update_with_likelihood(filter_obj, likelihood_for_filter, measurement):
    """Apply a measurement-specific likelihood update to a compatible filter."""
    if likelihood_for_filter is None:
        raise ValueError(
            "use_likelihood=True requires scenario_config['likelihood'] to be set."
        )

    update_likelihood = getattr(filter_obj, "update_nonlinear_using_likelihood", None)
    if update_likelihood is None:
        raise NotImplementedError(
            f"{type(filter_obj).__name__} does not support likelihood-based updates."
        )

    if hasattr(likelihood_for_filter, "pdf"):
        update_likelihood(likelihood_for_filter)
    else:
        update_likelihood(likelihood_for_filter, measurement=squeeze(measurement))


def _store_estimate(all_estimates, time_index, estimate):
    """Store ``estimate`` in dense or object-array estimate containers."""
    try:
        all_estimates[time_index, :] = estimate
    except IndexError:
        all_estimates[time_index] = estimate


def _last_stored_estimate(all_estimates):
    """Return the final estimate from dense or object-array estimate containers."""
    try:
        return all_estimates[-1, :]
    except IndexError:
        return all_estimates[-1]


# pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-positional-arguments
def perform_predict_update_cycles(
    scenario_config,
    filter_config,
    groundtruth,
    measurements,
    precalculated_params=None,
    extract_all_estimates=False,
    cumulated_updates_preferred=None,
):
    if extract_all_estimates:
        all_estimates = _empty_estimates_like(groundtruth)
    else:
        all_estimates = None

    # Configure filter
    filter_obj, prediction_routine, likelihood_for_filter, meas_noise_for_filter = (
        configure_for_filter(filter_config, scenario_config, precalculated_params)
    )

    # Check conditions for cumulative updates
    perform_cumulative_updates = cumulated_updates_preferred and any(
        array(scenario_config["n_meas_at_individual_time_step"]) > 1
    )
    if (
        cumulated_updates_preferred
        and any(array(scenario_config["n_meas_at_individual_time_step"]) > 1)
        and scenario_config.get("plot", False)
    ):
        warnings.warn("When plotting, measurements are fused sequentially.")
        perform_cumulative_updates = False

    # Start timer
    start_time = time.perf_counter()

    # Perform predict and update cycles
    for t in range(scenario_config["n_timesteps"]):
        # Update
        if scenario_config.get("mtt", False):
            raise NotImplementedError("MTT not implemented yet.")
        if perform_cumulative_updates:
            raise NotImplementedError("Cumulative updates not implemented yet.")

        all_meas_curr_time_step = atleast_2d(array(measurements[t]))
        n_updates = all_meas_curr_time_step.shape[0]

        if scenario_config.get("eot", False):
            meas_matrix = scenario_config.get("eot_meas_matrix")
            filter_obj.update(
                all_meas_curr_time_step.T, meas_matrix, meas_noise_for_filter
            )
        else:
            for m in range(n_updates):
                curr_meas = all_meas_curr_time_step[m, :]

                if scenario_config.get("use_likelihood", False):
                    _update_with_likelihood(
                        filter_obj, likelihood_for_filter, curr_meas
                    )
                else:
                    filter_obj.update_identity(
                        meas_noise=meas_noise_for_filter, measurement=squeeze(curr_meas)
                    )

                # If plotting is required
                if scenario_config.get("plot", False):
                    raise NotImplementedError("Plotting is not implemented yet.")

        # Save results only if required (takes time)
        if extract_all_estimates:
            _store_estimate(all_estimates, t, filter_obj.get_point_estimate())

        # Predict
        if scenario_config["apply_sys_noise_times"][t]:
            if scenario_config.get("inputs", None) is None:
                prediction_routine()
            else:
                prediction_routine(scenario_config["inputs"][:, t])

            # If plotting is required
            if (
                scenario_config.get("plot", False)
                and t != scenario_config["n_timesteps"] - 1
            ):
                raise NotImplementedError("Plotting is not implemented yet.")

    # End timer
    time_elapsed = time.perf_counter() - start_time

    # Get the final filter state and estimate
    last_filter_state = filter_obj.filter_state
    if all_estimates is not None:
        last_estimate = _last_stored_estimate(all_estimates)
    else:
        last_estimate = filter_obj.get_point_estimate()

    return last_filter_state, time_elapsed, last_estimate, all_estimates
