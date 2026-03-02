import time
import warnings

import numpy as np

from .configure_for_filter import configure_for_filter


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
        all_estimates = np.empty_like(groundtruth)
    else:
        all_estimates = None
    if cumulated_updates_preferred is None:
        # Turn on cumulated updates if eot or mtt is turned on
        cumulated_updates_preferred = scenario_config.get("eot", False) or scenario_config.get("mtt", False)
    # Configure filter
    filter_obj, prediction_routine, _, meas_noise_for_filter = configure_for_filter(
        filter_config, scenario_config, precalculated_params
    )

    n_meas_at_individual_time_step = [1 if elem.ndim == 1 else elem.shape[0] for elem in measurements]

    # Check conditions for cumulative updates
    perform_cumulative_updates = (cumulated_updates_preferred
                                  and any(np.array(n_meas_at_individual_time_step) > 1)
                                  or scenario_config.get("eot", False))
    if (
        perform_cumulative_updates
        and scenario_config.get("plot", False)
    ):
        warnings.warn("When plotting, measurements are fused sequentially.")
        perform_cumulative_updates = False

    # Start timer
    start_time = time.time()

    # Perform predict and update cycles
    for t in range(scenario_config["n_timesteps"]):
        # Update
        if scenario_config.get("mtt", False):
            raise NotImplementedError("MTT not implemented yet.")
        if scenario_config.get("eot", False):
            assert perform_cumulative_updates

        all_meas_curr_time_step = np.atleast_2d(measurements[t])
        n_updates = all_meas_curr_time_step.shape[0]
        if perform_cumulative_updates:
            if scenario_config.get("eot", False):
                assert scenario_config["kinematic_state_to_pos_matrix"] is not None
                filter_obj.update_linear(all_meas_curr_time_step, meas_noise_cov=meas_noise_for_filter)
            else:
                raise NotImplementedError("Cumulative updates not implemented yet.")
        else:
            for m in range(n_updates):
                curr_meas = all_meas_curr_time_step[m, :]

                if not scenario_config.get("use_likelihood", False):
                    filter_obj.update_identity(
                        meas_noise=meas_noise_for_filter, measurement=np.squeeze(curr_meas)
                    )

                # If plotting is required
                if scenario_config.get("plot", False):
                    raise NotImplementedError("Plotting is not implemented yet.")

        # Save results only if required (takes time)
        if extract_all_estimates:
            all_estimates[t, :] = filter_obj.get_point_estimate()

        # Predict
        if scenario_config["apply_sys_noise_times"][t]:
            if scenario_config.get("inputs", None) is None:
                prediction_routine()
            else:
                prediction_routine(scenario_config["inputs"][:, t])

            # If plotting is required
            if scenario_config.get("plot", False) and t != scenario_config["timesteps"]:
                raise NotImplementedError("Plotting is not implemented yet.")

    # End timer
    time_elapsed = time.time() - start_time

    # Get the final filter state and estimate
    last_filter_state = filter_obj.filter_state
    if all_estimates is not None:
        last_estimate = all_estimates[-1, :]
    else:
        last_estimate = filter_obj.get_point_estimate()

    return last_filter_state, time_elapsed, last_estimate, all_estimates
