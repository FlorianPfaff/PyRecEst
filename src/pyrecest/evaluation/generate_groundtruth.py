import numpy as np

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    array,
    atleast_2d,
    empty_like,
    get_default_dtype,
    reshape,
)


def _validate_initial_state_shape(x0, simulation_param):
    n_targets = simulation_param["n_targets"]
    if n_targets == 1:
        if x0.ndim == 1:
            return
        if x0.ndim == 2 and x0.shape[0] == 1:
            return
    elif x0.ndim == 2 and x0.shape[0] == n_targets:
        return

    raise ValueError("Mismatch in number of targets.")


def _as_state_noise_vector(noise_sample, state_dim):
    """Normalize one system-noise draw without coordinate broadcasting."""
    sample_shape = tuple(np.shape(noise_sample))
    vector_shape = (state_dim,)
    batched_shape = (1, state_dim)

    if sample_shape == vector_shape:
        return noise_sample
    if sample_shape == batched_shape:
        return reshape(noise_sample, vector_shape)
    if state_dim == 1 and sample_shape == ():
        return reshape(array(noise_sample), vector_shape)

    raise ValueError(
        "sys_noise.sample(1) must return one state-noise vector with shape "
        f"{batched_shape} or {vector_shape}; got {sample_shape}."
    )


# pylint: disable=too-many-branches
def generate_groundtruth(simulation_param, x0=None):
    """
    Generate ground truth based on the given scenario parameters.

    Parameters:
        simulation_param (dict): Dictionary containing scenario parameters.
        x0 (ndarray): Starting point (optional).

    Returns:
        groundtruth (np.ndarray[np.ndarray]): Generated ground truth as an
        array of arrays (!) because the size of the ground truth is not
        necessarily the same over time (e.g., if the number of targets changes)
    """
    if x0 is None:
        x0 = simulation_param["initial_prior"].sample(simulation_param["n_targets"])

    _validate_initial_state_shape(x0, simulation_param)

    # Initialize ground truth
    groundtruth = np.empty(simulation_param["n_timesteps"], dtype=object)

    has_inputs = "inputs" in simulation_param and simulation_param["inputs"] is not None
    if has_inputs:
        if (
            simulation_param["inputs"].ndim != 2
            or simulation_param["inputs"].shape[1]
            != simulation_param["n_timesteps"] - 1
        ):
            raise ValueError("Mismatch in number of timesteps.")

    groundtruth[0] = atleast_2d(x0)
    state_dim = groundtruth[0].shape[1]
    state_dtype = groundtruth[0].dtype
    dtype_kind = getattr(state_dtype, "kind", None)
    integer_or_bool_dtype = dtype_kind in ("b", "i", "u")
    if dtype_kind is None:
        integer_or_bool_dtype = (
            getattr(state_dtype, "is_floating_point", None) is False
            and getattr(state_dtype, "is_complex", None) is False
        )
    if integer_or_bool_dtype:
        state_dtype = get_default_dtype()

    for t in range(1, simulation_param["n_timesteps"]):
        groundtruth[t] = empty_like(groundtruth[0], dtype=state_dtype)
        for target_no in range(simulation_param["n_targets"]):
            previous_state = groundtruth[t - 1][target_no, :]
            if "gen_next_state_with_noise" in simulation_param:
                if not has_inputs:
                    groundtruth[t][target_no, :] = simulation_param[
                        "gen_next_state_with_noise"
                    ](previous_state)
                else:
                    groundtruth[t][target_no, :] = simulation_param[
                        "gen_next_state_with_noise"
                    ](
                        previous_state,
                        simulation_param["inputs"][:, t - 1],
                    )

            elif "sys_noise" in simulation_param:
                if "gen_next_state_without_noise" in simulation_param:
                    if not has_inputs:
                        state_to_add_noise_to = simulation_param[
                            "gen_next_state_without_noise"
                        ](previous_state)
                    else:
                        state_to_add_noise_to = simulation_param[
                            "gen_next_state_without_noise"
                        ](
                            previous_state,
                            simulation_param["inputs"][:, t - 1],
                        )
                else:
                    if has_inputs:
                        raise ValueError(
                            "No inputs accepted for the identity system model."
                        )
                    state_to_add_noise_to = previous_state

                sys_noise_sample = _as_state_noise_vector(
                    simulation_param["sys_noise"].sample(1), state_dim
                )
                groundtruth[t][target_no, :] = state_to_add_noise_to + sys_noise_sample
            else:
                raise ValueError("Cannot generate groundtruth.")

    if groundtruth[0].shape[0] != simulation_param["n_targets"]:
        raise RuntimeError("Generated groundtruth has the wrong number of targets.")
    if groundtruth[0].shape[1] != simulation_param["initial_prior"].dim:
        raise RuntimeError("Generated groundtruth has the wrong state dimension.")
    for t in range(simulation_param["n_timesteps"]):
        if simulation_param["n_targets"] == 1:
            groundtruth[t] = reshape(groundtruth[t], (-1,))

    return groundtruth
