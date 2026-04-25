import warnings
from typing import Optional

from beartype import beartype

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import eye, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.evaluation.eot_shape_database import PolygonWithSampling


@beartype
def simulation_database(
    scenario_name: str = "custom", scenario_customization_params: Optional[dict] = None
) -> dict:
    simulation_param: dict = {
        "n_timesteps": None,
        "all_seeds": None,
        "eot": False,
        "mtt": False,
        "n_targets": 1,
    }

    if scenario_name == "custom":
        warnings.warn(
            "Scenario not recognized. Assuming scenario_customization_params contains all parameters."
        )
        assert scenario_customization_params is not None
        simulation_param.update(scenario_customization_params)
    elif scenario_name == "R2randomWalk":
        simulation_param["manifold"] = "Euclidean"
        simulation_param["n_timesteps"] = 10
        simulation_param["initial_prior"] = GaussianDistribution(zeros(2), 0.5 * eye(2))
        simulation_param["meas_noise"] = GaussianDistribution(zeros(2), 0.5 * eye(2))
        simulation_param["sys_noise"] = GaussianDistribution(zeros(2), 0.5 * eye(2))
        simulation_param["gen_next_state_without_noise_is_vectorized"] = True
    elif scenario_name == "R2randomWalkEOT":
        simulation_param["eot"] = True
        simulation_param["manifold"] = "Euclidean"
        simulation_param["n_timesteps"] = 10
        simulation_param["initial_prior"] = GaussianDistribution(zeros(2), 0.5 * eye(2))
        simulation_param["sys_noise"] = GaussianDistribution(zeros(2), 0.1 * eye(2))
        simulation_param["gen_next_state_without_noise_is_vectorized"] = True
        simulation_param["target_shape"] = PolygonWithSampling(
            [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
        )
        simulation_param["eot_sampling_style"] = "boundary"
        simulation_param["n_meas_at_individual_time_step"] = [5] * simulation_param[
            "n_timesteps"
        ]
        simulation_param["eot_initial_extent"] = eye(2)
        simulation_param["eot_meas_matrix"] = eye(2)
        simulation_param["eot_meas_noise_cov"] = 0.1 * eye(2)
        simulation_param["eot_dt"] = 1.0
        simulation_param["eot_tau"] = 10.0
        simulation_param["eot_system_matrix"] = eye(2)
        simulation_param["eot_process_noise"] = 0.1 * eye(2)
    else:
        raise ValueError("Scenario not recognized.")

    return simulation_param
