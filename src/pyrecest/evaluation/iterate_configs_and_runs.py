import warnings
from typing import Any, Dict

from pyrecest.backend import empty, zeros

from .perform_predict_update_cycles import perform_predict_update_cycles


# pylint: disable=R0913,R0914,W0718,R0912
def iterate_configs_and_runs(
    groundtruths,
    measurements,
    scenario_config,
    filter_configs: list[Dict[str, None | int | list[Any]]],
    evaluation_config,
):
    if evaluation_config["extract_all_point_estimates"]:
        warnings.warn(
            "Extracting all point estimates can have a massive impact on the run time. Use this for debugging only"
        )
        raise NotImplementedError("This is not implemented yet.")

    n_configs = len(filter_configs)
    n_runs = groundtruths.shape[0]
    run_times = empty((n_configs, n_runs))
    run_failed = zeros((n_configs, n_runs), dtype=bool)

    if evaluation_config["convert_to_point_estimate_during_runtime"]:
        raise NotImplementedError("This is not implemented yet.")

    last_filter_states = empty((n_configs, n_runs), dtype=object)

    for run in range(n_runs):
        for config_no, filter_config in enumerate(filter_configs):
            try:
                if (
                    not evaluation_config["convert_to_point_estimate_during_runtime"]
                    and not evaluation_config["extract_all_point_estimates"]
                ):
                    (
                        last_filter_states[config_no, run],
                        run_times[config_no, run],
                        *_,
                    ) = perform_predict_update_cycles(
                        scenario_config,
                        filter_config=filter_config,
                        groundtruth=groundtruths[run, :],
                        measurements=measurements[run, :],
                    )

                elif (
                    not evaluation_config["convert_to_point_estimate_during_runtime"]
                    and evaluation_config["extract_all_point_estimates"]
                ):
                    raise NotImplementedError("This is not implemented yet.")
                elif (
                    evaluation_config["convert_to_point_estimate_during_runtime"]
                    and not evaluation_config["extract_all_point_estimates"]
                ):
                    raise NotImplementedError("This is not implemented yet.")
                elif (
                    evaluation_config["convert_to_point_estimate_during_runtime"]
                    and evaluation_config["extract_all_point_estimates"]
                ):
                    raise NotImplementedError("This is not implemented yet.")
                else:
                    raise ValueError("This should not happen.")

            except Exception as err:
                if not evaluation_config["tolerate_failure"]:
                    raise err
                run_failed[config_no, run] = True
                run_times[config_no, run] = float("nan")
                if evaluation_config["auto_warning_on_off"]:
                    warnings.warn(
                        f"Filter {config_no} config {filter_config['parameter']} run {run} FAILED: {str(err)}"
                    )

    return last_filter_states, run_times, run_failed, groundtruths, measurements
