from pyrecest.filters import (
    EuclideanParticleFilter,
    HypertoroidalParticleFilter,
    KalmanFilter,
)
from pyrecest.filters.random_matrix_tracker import RandomMatrixTracker


# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
def configure_for_filter(filter_config, scenario_config, precalculated_params=None):
    if precalculated_params is not None:
        raise NotImplementedError(
            "No filters using precalculated parameters have been implemented so far."
        )

    # Check for likelihood and measurement noise in scenario parameters
    likelihood_for_filter = scenario_config.get("likelihood", None)
    meas_noise_for_filter = scenario_config.get("meas_noise", None)

    # Switch-case based on filter name
    filter_name = filter_config["name"]

    if filter_name == "kf":
        # Implement your KalmanFilter class and its methods
        filter_obj = KalmanFilter(scenario_config["initial_prior"])
        meas_noise_for_filter = meas_noise_for_filter.covariance()
        if scenario_config.get("inputs") is None:

            def prediction_routine():  # type: ignore
                return filter_obj.predict_identity(
                    scenario_config["sys_noise"].covariance()
                )

        else:

            def prediction_routine(curr_input):  # type: ignore
                return filter_obj.predict_identity(
                    scenario_config["sys_noise"], curr_input
                )

    elif filter_name == "twn":
        raise NotImplementedError("ToroidalWNFilter not implemented yet")

    elif filter_name in ["iff", "sqff"]:
        raise NotImplementedError("IF and SQFF not implemented yet")

    elif filter_name == "htgf":
        raise NotImplementedError("HTGF not implemented yet")

    elif filter_name == "fig":
        raise NotImplementedError("FIG not implemented yet")

    elif filter_name == "ishf":
        raise NotImplementedError("ISHF not implemented yet")

    elif filter_name == "pf":
        no_particles = filter_config["parameter"]

        if no_particles is None or no_particles == 0:
            raise ValueError("Using zero particles does not make sense")

        manifold = scenario_config["manifold"]

        if manifold in ["circle", "hypertorus"]:
            assert (
                scenario_config.get("inputs") is None
            ), "Inputs currently not supported for the current setting."

            filter_obj = HypertoroidalParticleFilter(
                no_particles, scenario_config["initial_prior"].dim
            )
            filter_obj.filter_state = scenario_config["initial_prior"]

            if "gen_next_state_with_noise" in scenario_config:

                def prediction_routine():  # type: ignore
                    return filter_obj.predict_nonlinear(
                        scenario_config["gen_next_state_with_noise"],
                        None,
                        scenario_config.get("genNextStateWithNoiseIsVectorized", False),
                    )

            elif "sys_noise" in scenario_config:

                def prediction_routine():  # type: ignore
                    return filter_obj.predict_nonlinear(
                        lambda x: x, scenario_config["sys_noise"], True
                    )

        elif manifold in [
            "hypersphere",
            "hypersphere_general",
            "hypersphere_symmetric",
        ]:
            assert (
                scenario_config.get("inputs") is None
            ), "Inputs currently not supported for the current setting."
            raise NotImplementedError(
                "HypersphericalParticleFilter not implemented yet"
            )
        elif manifold in [
            "euclidean",
            "Euclidean",
        ]:
            filter_obj = EuclideanParticleFilter(
                no_particles, scenario_config["initial_prior"].dim
            )
            if scenario_config.get("inputs") is None:

                def prediction_routine():  # type: ignore[misc]
                    return filter_obj.predict_identity(scenario_config["sys_noise"])

            else:

                def prediction_routine(curr_input):  # type: ignore[misc]
                    return filter_obj.predict_identity(
                        scenario_config["sys_noise"].shift(curr_input)
                    )

        else:
            raise NotImplementedError("Manifold not supported yet")

    elif filter_name in ["sgf", "hgf"]:
        raise NotImplementedError("SGF and HGF not implemented yet")

    elif filter_name == "vmf":
        raise NotImplementedError("VMF not implemented yet")

    elif filter_name == "s3f":
        raise NotImplementedError("S3F not implemented yet")

    elif filter_name == "se2bf":
        raise NotImplementedError("SE2BF not implemented yet")

    elif filter_name == "se2ukfm":
        raise NotImplementedError("SE2UKFM not implemented yet")

    elif filter_name == "dummy":
        raise NotImplementedError("Dummy not implemented yet")

    elif filter_name == "gnn":
        raise NotImplementedError("GNN not implemented yet")

    elif filter_name == "random_matrix":
        
        filter_obj = RandomMatrixTracker(scenario_config["initial_prior"].mean(), scenario_config["initial_prior"].covariance(),
                                         scenario_config["initial_extent_matrix"], scenario_config["kinematic_state_to_pos_matrix"])
        def prediction_routine():  # type: ignore[misc]
            return filter_obj.predict_linear(scenario_config["sys_noise_cov"], scenario_config["sys_mat"])
        
        
    else:
        raise ValueError("Filter currently unsupported")

    return (
        filter_obj,
        prediction_routine,
        likelihood_for_filter,
        meas_noise_for_filter,
    )
