import warnings

import pyrecest.backend
from beartype import beartype
from beartype.typing import Callable
from pyrecest.backend import any, empty, is_array, isfinite, sum


def _shape_size(container):
    shape = getattr(container, "shape", None)
    if shape is None:
        return len(container)
    n_entries = 1
    for dimension in shape:
        n_entries *= int(dimension)
    return n_entries


@beartype
def determine_all_deviations(
    results,
    extract_mean,
    distance_function: Callable,
    groundtruths,
    mean_calculation_symm: str = "",
):
    if pyrecest.backend.__backend_name__ == "jax":  # pylint: disable=no-member
        raise NotImplementedError("Not supported for the JAX backend.")

    if mean_calculation_symm != "":
        raise NotImplementedError("Not implemented yet")

    groundtruths_ndim = getattr(groundtruths, "ndim", None)
    groundtruths_size = _shape_size(groundtruths) if groundtruths_ndim == 2 else 0
    first_groundtruth_ndim = (
        getattr(groundtruths[0, 0], "ndim", None) if groundtruths_size != 0 else None
    )
    if (
        groundtruths_ndim != 2
        or groundtruths_size == 0
        or first_groundtruth_ndim not in (1, 2)
    ):
        raise ValueError(
            "Assuming groundtruths to be a non-empty 2-D array of shape "
            "(n_runs, n_timesteps) composed arrays of shape (n_dim,) or "
            "(n_targets,n_dim)."
        )

    n_runs = groundtruths.shape[0]
    all_deviations_last_mat = empty((len(results), n_runs))

    for config_no, result_curr_config in enumerate(results):
        if len(result_curr_config) != n_runs:
            raise ValueError(
                "Each result configuration must contain one entry per groundtruth run."
            )
        for run in range(n_runs):
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

        failed_mask = ~isfinite(all_deviations_last_mat[config_no])
        if any(failed_mask):
            all_deviations_last_mat[config_no][failed_mask] = float("inf")
            warnings.warn(
                f"Filter result {config_no} apparently failed "
                f"{int(sum(failed_mask))} times. Check if this is plausible."
            )

    return all_deviations_last_mat
