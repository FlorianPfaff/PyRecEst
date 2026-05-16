import os
from typing import Any

import numpy as np

from .evaluate_for_variables import evaluate_for_variables


# jscpd:ignore-start
# pylint: disable=R0913,R0914,too-many-positional-arguments
def evaluate_for_file(
    input_file_name: str,
    filter_configs: list[dict[str, Any]],
    scenario_config,
    save_folder: str = ".",
    plot_each_step: bool = False,
    convert_to_point_estimate_during_runtime: bool = False,
    extract_all_point_estimates: bool = False,
    tolerate_failure: bool = False,
    auto_warning_on_off: bool = False,
    # jscpd:ignore-end
) -> tuple[
    dict,
    list[dict],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    data = np.load(input_file_name, allow_pickle=True).item()

    if "name" not in scenario_config:
        scenario_config["name"] = os.path.splitext(os.path.basename(input_file_name))[0]
    if "n_timesteps" in scenario_config:
        assert (
            scenario_config["n_timesteps"] == data["groundtruths"].shape[1]
        ), "n_timesteps in scenario_config does not match the number of measurements in the file"
    else:
        scenario_config["n_timesteps"] = data["groundtruths"].shape[1]

    n_meas_at_individual_time_step = np.zeros(data["measurements"].shape[1], dtype=int)

    for idx, inner_array in enumerate(data["measurements"][0]):
        if inner_array.ndim == 2:
            n_meas_at_individual_time_step[idx] = inner_array.shape[0]
        elif inner_array.ndim == 1:
            n_meas_at_individual_time_step[idx] = 1

    scenario_config.setdefault(
        "n_meas_at_individual_time_step", n_meas_at_individual_time_step
    )
    scenario_config.setdefault(
        "apply_sys_noise_times",
        np.concatenate(
            [np.ones(scenario_config["n_timesteps"] - 1, dtype=bool), [False]]
        ),
    )

    return evaluate_for_variables(
        data["groundtruths"],
        data["measurements"],
        filter_configs,
        scenario_config,
        save_folder=save_folder,
        plot_each_step=plot_each_step,
        convert_to_point_estimate_during_runtime=convert_to_point_estimate_during_runtime,
        extract_all_point_estimates=extract_all_point_estimates,
        tolerate_failure=tolerate_failure,
        auto_warning_on_off=auto_warning_on_off,
    )
