import warnings
from typing import Optional

from beartype import beartype

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import eye, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.evaluation.eot_shape_database import Cross

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
    elif scenario_name in ("R2randomWalk", "R2randomWalkEOT"):
        simulation_param["manifold"] = "Euclidean"
        simulation_param["n_timesteps"] = 10
        simulation_param["initial_prior"] = GaussianDistribution(zeros(2), 0.5 * eye(2))
        simulation_param["meas_noise"] = GaussianDistribution(zeros(2), 0.5 * eye(2))
        simulation_param["sys_noise"] = GaussianDistribution(zeros(2), 0.5 * eye(2))
        simulation_param["gen_next_state_without_noise_is_vectorized"] = True
        simulation_param["eot"] = "EOT" in scenario_name
        if simulation_param["eot"]:
            simulation_param["intensity_lambda"] = 0.2
            simulation_param["target_shape"] = Cross(2, 1, 2, 3)
            simulation_param["eot_sample_from"] = "within"
            simulation_param["initial_extent_matrix"] = eye(2)
            simulation_param["kinematic_state_to_pos_matrix"] = eye(2)
            simulation_param["sys_mat"] = eye(2)
        
    else:
        raise ValueError("Scenario not recognized.")

    return simulation_param
