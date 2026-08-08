import numpy as np

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    asarray,
    atleast_2d,
    empty_like,
    get_default_dtype,
    reshape,
    squeeze,
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


def _as_transition_state(value, expected_dim, transition_name):
    """Normalize one callback result without allowing NumPy broadcasting."""
    state = asarray(value)
    shape = tuple(state.shape)
    if shape == (expected_dim,):
        return state
    if shape == (1, expected_dim):
        return reshape(state, (expected_dim,))
    if expected_dim == 1 and shape == ():
        return reshape(state, (1,))
    raise ValueError(
        f"{transition_name} must return a state vector with shape "
        f"({expected_dim},); got {shape}."
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
                    next_state = simulation_param["gen_next_state_with_noise"](
                        previous_state
                    )
                else:
                    next_state = simulation_param["gen_next_state_with_noise"](
                        previous_state,
                        simulation_param["inputs"][:, t - 1],
                    )
                groundtruth[t][target_no, :] = _as_transition_state(
                    next_state,
                    previous_state.shape[0],
                    "gen_next_state_with_noise",
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
                    state_to_add_noise_to = _as_transition_state(
                        state_to_add_noise_to,
                        previous_state.shape[0],
                        "gen_next_state_without_noise",
                    )
                else:
                    if has_inputs:
                        raise ValueError(
                            "No inputs accepted for the identity system model."
                        )
                    state_to_add_noise_to = previous_state

                sys_noise_sample = squeeze(simulation_param["sys_noise"].sample(1))
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
