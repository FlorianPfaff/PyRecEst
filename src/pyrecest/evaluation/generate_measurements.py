import copy

import numpy as np
from beartype import beartype

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import array, dot, get_backend_name, mod, pi, squeeze, tile, zeros
from pyrecest.distributions import (
    AbstractHypertoroidalDistribution,
    GaussianDistribution,
    VonMisesFisherDistribution,
    WatsonDistribution,
)
from pyrecest.evaluation.eot_shape_database import PolygonWithSampling
from scipy.stats import poisson
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon


def _as_shapely_scalar(value):
    if np.isscalar(value):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            pass
    return float(np.asarray(value).reshape(-1)[0])


def _as_nonnegative_measurement_count(value, name="measurement count") -> int:
    value_array = np.asarray(value)
    if value_array.shape != () or value_array.dtype == np.bool_:
        raise ValueError(f"{name} must be a non-negative integer.")

    scalar = value_array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer.")
    if isinstance(scalar, (int, np.integer)):
        count = int(scalar)
    else:
        try:
            scalar_float = float(scalar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a non-negative integer.") from exc
        if not np.isfinite(scalar_float) or not scalar_float.is_integer():
            raise ValueError(f"{name} must be a non-negative integer.")
        count = int(scalar_float)

    if count < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return count


def _mtt_state_at(groundtruth, t, target_no, n_targets):
    """Return one target state from dense or per-step object ground truth."""
    state_at_t = np.asarray(groundtruth[t])
    if n_targets == 1:
        if target_no != 0:
            raise IndexError("target_no out of bounds for single-target ground truth")
        if state_at_t.ndim == 0:
            return state_at_t.reshape(1)
        if state_at_t.ndim == 1:
            return state_at_t
    if state_at_t.ndim != 2:
        raise ValueError(
            "MTT groundtruth entries must have shape (n_targets, dim); "
            f"got {state_at_t.shape}."
        )
    return state_at_t[target_no, :]


# pylint: disable=too-many-branches,too-many-locals,too-many-statements
def generate_measurements(groundtruth, simulation_config):
    """
    Generate measurements based on the given groundtruth and scenario parameters.

    Parameters:
        groundtruth (ndarray): Ground truth data.
        simulation_config (dict): Dictionary containing scenario parameters.

    Returns:
        measurements (list): List of generated measurements at each time step.
        Is a list because the number of measurements can vary at each time step.
        Comprises timesteps elements, each of which is a numpy array of shape
        (n_meas_at_individual_time_step[t], n_dim).
    """
    if "n_meas_at_individual_time_step" in simulation_config and np.shape(
        simulation_config["n_meas_at_individual_time_step"]
    ) != (simulation_config["n_timesteps"],):
        raise ValueError(
            "n_meas_at_individual_time_step must have shape (n_timesteps,)"
        )
    measurements = np.empty(simulation_config["n_timesteps"], dtype=object)

    if simulation_config.get("mtt", False) and simulation_config.get("eot", False):
        raise NotImplementedError(
            "Multiple extended object tracking is currently not supported."
        )

    if simulation_config.get("eot", False):
        if "target_shape" not in simulation_config:
            raise ValueError("target_shape must be in simulation_config for EOT")
        if "eot_sampling_style" not in simulation_config:
            raise ValueError("eot_sampling_style must be in simulation_config for EOT")
        if ("intensity_lambda" in simulation_config) == (
            "n_meas_at_individual_time_step" in simulation_config
        ):
            raise ValueError(
                "Must either give intensity_lambda or n_meas_at_individual_time_step for EOT"
            )
        shape = simulation_config["target_shape"]
        eot_sampling_style = simulation_config["eot_sampling_style"]
        if not isinstance(shape, Polygon):
            raise TypeError(
                "Currently only StarConvexPolygon (based on shapely Polygons) are supported as target shapes."
            )

        for t in range(simulation_config["n_timesteps"]):
            curr_groundtruth = groundtruth[t]
            if curr_groundtruth.shape[-1] == 2:
                x = _as_shapely_scalar(curr_groundtruth[..., 0])
                y = _as_shapely_scalar(curr_groundtruth[..., 1])
                curr_shape = translate(shape, xoff=x, yoff=y)
            elif curr_groundtruth.shape[-1] == 3:
                angle = _as_shapely_scalar(curr_groundtruth[..., 0])
                x = _as_shapely_scalar(curr_groundtruth[..., 1])
                y = _as_shapely_scalar(curr_groundtruth[..., 2])
                curr_shape = rotate(
                    translate(shape, xoff=x, yoff=y),
                    angle=angle,
                    origin="centroid",
                    use_radians=True,
                )
            else:
                raise ValueError(
                    "Currently only R^2 and SE(2) scenarios are supported."
                )
            if not isinstance(curr_shape, PolygonWithSampling):
                curr_shape.__class__ = (
                    PolygonWithSampling  # Evil class surgery to add sampling methods
                )

            if "n_meas_at_individual_time_step" in simulation_config:
                if "intensity_lambda" in simulation_config:
                    raise ValueError(
                        "Cannot use both intensity_lambda and "
                        "n_meas_at_individual_time_step."
                    )
                n_meas_curr = _as_nonnegative_measurement_count(
                    simulation_config["n_meas_at_individual_time_step"][t],
                    name=f"n_meas_at_individual_time_step[{t}]",
                )
            else:
                if eot_sampling_style == "boundary":
                    n_meas_curr = generate_n_measurements_PPP(
                        curr_shape.length, simulation_config["intensity_lambda"]
                    )
                elif eot_sampling_style == "within":
                    n_meas_curr = generate_n_measurements_PPP(
                        curr_shape.area, simulation_config["intensity_lambda"]
                    )
                else:
                    raise ValueError(
                        "eot_sampling_style must be either 'boundary' or 'within'."
                    )

            if eot_sampling_style == "boundary":
                measurements[t] = curr_shape.sample_on_boundary(n_meas_curr)
            elif eot_sampling_style == "within":
                measurements[t] = curr_shape.sample_within(n_meas_curr)
            else:
                raise ValueError(
                    "eot_sampling_style must be either 'boundary' or 'within'."
                )

    elif simulation_config.get("mtt", False):
        if get_backend_name() == "jax":
            raise NotImplementedError(
                "generate_measurements does not support MTT scenarios with the JAX backend."
            )
        if simulation_config["clutter_rate"] != 0:
            raise NotImplementedError("Clutter currently not supported.")

        n_observations = np.random.binomial(
            1,
            simulation_config["detection_probability"],
            (simulation_config["n_timesteps"], simulation_config["n_targets"]),
        )

        measurement_dim = simulation_config["meas_matrix_for_each_target"].shape[0]
        for t in range(simulation_config["n_timesteps"]):
            n_meas_at_t = int(np.sum(n_observations[t, :]))
            measurements[t] = float("NaN") * zeros((n_meas_at_t, measurement_dim))

            meas_no = 0
            for target_no in range(simulation_config["n_targets"]):
                if n_observations[t, target_no] == 1:
                    meas_no += 1
                    target_state = _mtt_state_at(
                        groundtruth,
                        t,
                        target_no,
                        simulation_config["n_targets"],
                    )
                    measurements[t][meas_no - 1, :] = dot(
                        simulation_config["meas_matrix_for_each_target"],
                        array(target_state),
                    ) + squeeze(simulation_config["meas_noise"].sample(1))
                else:
                    if n_observations[t, target_no] != 0:
                        raise NotImplementedError(
                            "Multiple measurements currently not supported."
                        )

            if meas_no != n_meas_at_t:
                raise RuntimeError("Mismatch in number of measurements.")

    else:
        if "meas_generator" in simulation_config:
            raise NotImplementedError(
                "Scenarios based on a 'measGenerator' are currently not supported."
            )
        for t in range(simulation_config["n_timesteps"]):
            n_meas = _as_nonnegative_measurement_count(
                simulation_config["n_meas_at_individual_time_step"][t],
                name=f"n_meas_at_individual_time_step[{t}]",
            )
            meas_noise = simulation_config["meas_noise"]

            if isinstance(meas_noise, AbstractHypertoroidalDistribution):
                noise_samples = meas_noise.sample(n_meas)
                measurements[t] = mod(
                    tile(
                        groundtruth[t],
                        (
                            n_meas,
                            1,
                        ),
                    )
                    + noise_samples,
                    2.0 * pi,
                )

            elif isinstance(
                meas_noise, (VonMisesFisherDistribution, WatsonDistribution)
            ):
                curr_dist = copy.deepcopy(meas_noise)
                curr_dist.mu = groundtruth[t]
                measurements[t] = curr_dist.sample(n_meas)
            elif isinstance(meas_noise, GaussianDistribution):
                noise_samples = meas_noise.sample(n_meas)
                measurements[t] = tile(
                    groundtruth[t],
                    (
                        n_meas,
                        1,
                    ),
                ) + noise_samples
            else:
                raise TypeError(
                    "meas_noise must be a supported PyRecEst distribution; "
                    f"got {type(meas_noise).__name__}."
                )

    return measurements


@beartype
def generate_n_measurements_PPP(area: float, intensity_lambda: float) -> int:
    # Compute the expected number of points
    expected_num_points = intensity_lambda * area
    # Get the actual number of points to generate as a realization from a Poisson distribution
    return poisson.rvs(expected_num_points)
