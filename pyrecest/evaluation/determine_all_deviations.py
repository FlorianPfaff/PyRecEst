import warnings

import pyrecest.backend
from beartype import beartype
from beartype.typing import Callable
from pyrecest.backend import any, empty, is_array, isinf, sum


@beartype
def determine_all_deviations(
    results,
    extract_mean,
    distance_function: Callable,
    groundtruths,
    mean_calculation_symm: str = "",
):
    assert (
        pyrecest.backend.__backend_name__ != "jax"  # pylint: disable=no-member
    ), "Not supported for the JAX backend."

    if mean_calculation_symm != "":
        raise NotImplementedError("Not implemented yet")

    assert groundtruths.ndim == 2 and groundtruths[0, 0].ndim in (
        1,
        2,
    ), "Assuming groundtruths to be a 2-D array of shape (n_runs, n_timesteps) composed arrays of shape (n_dim,) or (n_targets,n_dim)."

    all_deviations_last_mat = empty((len(results), groundtruths.shape[0]))

    for config_no, result_curr_config in enumerate(results):
        for run in range(len(groundtruths)):
            if is_array(result_curr_config[run]):
                # If estimates are already given as an array, use it
                final_estimate = result_curr_config[run]
            elif callable(extract_mean):
                # Otherwise, use extract_mean to obtain the estimate
                final_estimate = extract_mean(result_curr_config[run])
            else:
                raise ValueError("No compatible mean extraction function given.")

            if final_estimate is not None:
                all_deviations_last_mat[config_no][run] = distance_function(
                    final_estimate, groundtruths[run, -1]
                )
            else:
                warnings.warn("No estimate for this filter, setting error to inf.")
                all_deviations_last_mat[config_no][run] = float("inf")

        if any(isinf(all_deviations_last_mat[config_no])):
            print(
                f"Warning: {result_curr_config['filterName']} with {result_curr_config['filterParams']} "
                f"parameters apparently failed {sum(isinf(all_deviations_last_mat[config_no]))} "
                "times. Check if this is plausible."
            )

    return all_deviations_last_mat
