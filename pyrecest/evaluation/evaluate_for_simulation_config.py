import random
from typing import Any, Optional

from beartype import beartype

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import size

from .evaluate_for_variables import evaluate_for_variables
from .generate_simulated_scenarios import generate_simulated_scenarios
from .simulation_database import simulation_database


# jscpd:ignore-start
# pylint: disable=R0913,R0914,too-many-positional-arguments
def evaluate_for_simulation_config(
    simulation_config: str | dict[str, Any],
    filter_configs: list[dict[str, Any]],
    n_runs: int,
    n_timesteps: Optional[int] = None,
    initial_seed: Optional[int] = None,
    consecutive_seed: bool = False,
    save_folder: str = ".",
    scenario_customization_params: Optional[dict] = None,
    plot_each_step: bool = False,
    convert_to_point_estimate_during_runtime: bool = False,
    extract_all_point_estimates: bool = False,
    tolerate_failure: bool = False,
    auto_warning_on_off: bool = False,
    # jscpd:ignore-end
) -> tuple[
    dict,
    list[dict],
    Any,
    Any,
    Any,
    Any,
    Any,
]:
    if isinstance(simulation_config, str):
        simulation_name = simulation_config
        simulation_config = simulation_database(
            simulation_config, scenario_customization_params
        )
        simulation_config["name"] = simulation_name  # type: ignore
    else:
        simulation_config["name"] = "custom"

    simulation_config["all_seeds"] = get_all_seeds(  # type: ignore
        n_runs, initial_seed, consecutive_seed
    )
    if n_timesteps is None:
        assert (
            "n_timesteps" in simulation_config
        ), "n_steps must be provided in simulation_config or as an argument."
    else:
        simulation_config["n_timesteps"] = n_timesteps  # type: ignore

    groundtruths, measurements = generate_simulated_scenarios(simulation_config)

    return evaluate_for_variables(
        groundtruths,
        measurements,
        filter_configs,
        simulation_config,
        save_folder,
        plot_each_step,
        convert_to_point_estimate_during_runtime,
        extract_all_point_estimates,
        tolerate_failure,
        auto_warning_on_off,
    )


@beartype
def get_all_seeds(n_runs: int, seed_input=None, consecutive_seed: bool = True):
    if seed_input is None:
        seed_input = random.randint(1, 0xFFFFFFFF)  # nosec

    if size(seed_input) == n_runs:
        all_seeds = seed_input
    elif size(seed_input) == 1 and n_runs > 1:
        seed_value = int(seed_input)
        if consecutive_seed:
            all_seeds = list(range(seed_value, seed_value + n_runs))
        else:
            random.seed(seed_value)
            all_seeds = [random.randint(1, 0xFFFFFFFF) for _ in range(n_runs)]  # nosec
    else:
        raise ValueError(
            "The number of seeds provided must be either 1 or equal to the number of runs."
        )

    return all_seeds
