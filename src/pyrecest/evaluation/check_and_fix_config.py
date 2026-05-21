from pyrecest.distributions import AbstractManifoldSpecificDistribution


def _expand_meas_per_step(simulation_param):
    if "n_meas_at_individual_time_step" in simulation_param:
        return
    if "meas_per_step" not in simulation_param:
        return

    assert isinstance(
        simulation_param["meas_per_step"], int
    ), "meas_per_step must be an integer"
    assert simulation_param["meas_per_step"] > 0, "meas_per_step must be positive"

    simulation_param["n_meas_at_individual_time_step"] = [
        simulation_param["meas_per_step"]
    ] * simulation_param["n_timesteps"]
    del simulation_param["meas_per_step"]


def _validate_measurement_counts(simulation_param):
    assert all(
        x > 0 for x in simulation_param["n_meas_at_individual_time_step"]
    ), "n_meas_at_individual_time_step must contain positive values"
    assert all(
        isinstance(x, int) for x in simulation_param["n_meas_at_individual_time_step"]
    ), "n_meas_at_individual_time_step must contain integer values"


def check_and_fix_config(simulation_param):
    # Initialize default values if they are not present
    simulation_param.setdefault("use_transition", False)
    simulation_param.setdefault("use_likelihood", False)
    simulation_param.setdefault("eot", False)
    simulation_param.setdefault("mtt", False)
    simulation_param.setdefault("n_targets", 1)
    simulation_param.setdefault(
        "apply_sys_noise_times",
        [True] * (simulation_param["n_timesteps"] - 1) + [False],
    )

    # Check for 'timesteps'
    timesteps = simulation_param["n_timesteps"]
    if timesteps is not None:
        assert timesteps > 0 and isinstance(timesteps, int)

    if "intensity_lambda" in simulation_param:
        assert (
            simulation_param["mtt"] or simulation_param["eot"]
        ), "Intensity lambda can only be used with MTT or EOT."
        assert (
            simulation_param["intensity_lambda"] > 0
        ), "Intensity lambda must be positive."

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
        assert (
            sum(
                key in simulation_param
                for key in [
                    "n_meas_at_individual_time_step",
                    "intensity_lambda",
                ]
            )
            == 1
        ), "Must use precisely one parameter to modulate the number of measurements (n_meas_at_individual_time_step or lambda_intensity)"
        if "n_meas_at_individual_time_step" in simulation_param:
            _validate_measurement_counts(simulation_param)
    elif simulation_param["mtt"]:
        assert (
            "n_meas_at_individual_time_step" not in simulation_param
            and "meas_per_step" not in simulation_param
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

        assert (
            "observed_area" in simulation_param or simulation_param["clutter_rate"] == 0
        ), "Can only add clutter if observed_area is set."

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

    assert isinstance(
        simulation_param["initial_prior"], AbstractManifoldSpecificDistribution
    )

    return simulation_param
