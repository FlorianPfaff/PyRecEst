from pyrecest.distributions import AbstractManifoldSpecificDistribution


def _expand_meas_per_step(simulation_param):
    if "n_meas_at_individual_time_step" in simulation_param:
        return
    if "meas_per_step" not in simulation_param:
        return

    if not isinstance(simulation_param["meas_per_step"], int):
        raise TypeError("meas_per_step must be an integer")
    if simulation_param["meas_per_step"] <= 0:
        raise ValueError("meas_per_step must be positive")

    simulation_param["n_meas_at_individual_time_step"] = [
        simulation_param["meas_per_step"]
    ] * simulation_param["n_timesteps"]
    del simulation_param["meas_per_step"]


def _validate_measurement_counts(simulation_param):
    counts = simulation_param["n_meas_at_individual_time_step"]
    if len(counts) != simulation_param["n_timesteps"]:
        raise ValueError(
            "n_meas_at_individual_time_step must have one entry per time step"
        )
    if not all(isinstance(x, int) for x in counts):
        raise TypeError("n_meas_at_individual_time_step must contain integer values")
    if not all(x > 0 for x in counts):
        raise ValueError("n_meas_at_individual_time_step must contain positive values")


def check_and_fix_config(simulation_param):
    # Initialize default values if they are not present
    simulation_param.setdefault("use_transition", False)
    simulation_param.setdefault("use_likelihood", False)
    simulation_param.setdefault("eot", False)
    simulation_param.setdefault("mtt", False)
    simulation_param.setdefault("n_targets", 1)
    # Check for 'timesteps'
    timesteps = simulation_param["n_timesteps"]
    if timesteps is not None:
        if not isinstance(timesteps, int):
            raise TypeError("n_timesteps must be an integer")
        if timesteps <= 0:
            raise ValueError("n_timesteps must be positive")

    simulation_param.setdefault(
        "apply_sys_noise_times",
        [True] * (simulation_param["n_timesteps"] - 1) + [False],
    )

    if "intensity_lambda" in simulation_param:
        if not (simulation_param["mtt"] or simulation_param["eot"]):
            raise ValueError("Intensity lambda can only be used with MTT or EOT.")
        if simulation_param["intensity_lambda"] <= 0:
            raise ValueError("Intensity lambda must be positive.")

    if simulation_param["mtt"] and simulation_param["eot"]:
        raise ValueError("MTT and EOT cannot be used together at the moment.")

    if simulation_param["eot"]:
        if (
            "n_meas_at_individual_time_step" in simulation_param
            and "meas_per_step" in simulation_param
        ):
            raise ValueError(
                "Do not provide n_meas_at_individual_time_step and meas_per_step at the same time."
            )
        _expand_meas_per_step(simulation_param)
        if (
            sum(
                key in simulation_param
                for key in [
                    "n_meas_at_individual_time_step",
                    "intensity_lambda",
                ]
            )
            != 1
        ):
            raise ValueError(
                "Must use precisely one parameter to modulate the number of measurements "
                "(n_meas_at_individual_time_step or intensity_lambda)"
            )
        if "n_meas_at_individual_time_step" in simulation_param:
            _validate_measurement_counts(simulation_param)
    elif simulation_param["mtt"]:
        if (
            "n_meas_at_individual_time_step" in simulation_param
            or "meas_per_step" in simulation_param
        ):
            raise ValueError(
                "MTT scenarios must not provide n_meas_at_individual_time_step or "
                "meas_per_step."
            )
        if "detectionProbability" in simulation_param:
            if (
                "detection_probability" in simulation_param
                and simulation_param["detection_probability"]
                != simulation_param["detectionProbability"]
            ):
                raise ValueError(
                    "Use only one detection probability value for MTT scenarios."
                )
            simulation_param["detection_probability"] = simulation_param[
                "detectionProbability"
            ]
        simulation_param.setdefault("detection_probability", 1)
        simulation_param.setdefault("clutter_rate", 0)

        if (
            "observed_area" not in simulation_param
            and simulation_param["clutter_rate"] != 0
        ):
            raise ValueError("Can only add clutter if observed_area is set.")

    elif (
        "n_meas_at_individual_time_step" in simulation_param
        and "meas_per_step" in simulation_param
    ):
        raise ValueError(
            "Do not provide n_meas_at_individual_time_step and meas_per_step at the same time."
        )

    elif (
        "n_meas_at_individual_time_step" not in simulation_param
        and "meas_per_step" in simulation_param
    ):
        _expand_meas_per_step(simulation_param)
        _validate_measurement_counts(simulation_param)

    elif "n_meas_at_individual_time_step" not in simulation_param:
        simulation_param["n_meas_at_individual_time_step"] = [1] * simulation_param[
            "n_timesteps"
        ]
    else:
        _validate_measurement_counts(simulation_param)

    if not isinstance(
        simulation_param["initial_prior"], AbstractManifoldSpecificDistribution
    ):
        raise TypeError("initial_prior must be an AbstractManifoldSpecificDistribution")

    return simulation_param
