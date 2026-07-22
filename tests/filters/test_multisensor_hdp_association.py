import unittest

import numpy as np
from pyrecest.filters.multisensor_hdp_association import (
    HDPAssociationLabel,
    multisensor_hdp_association,
    predict_survival_weighted_hdp_masses,
)


class MultisensorHDPAssociationTest(unittest.TestCase):
    def test_sensor_counts_reinforce_existing_target_atoms(self):
        results = multisensor_hdp_association(
            {
                "radar": np.log([[1.0, 1.0]]),
                "camera": np.log([[1.0, 4.0]]),
            },
            global_target_weights=np.array([1.0, 1.0]),
            global_birth_weight=1.0,
            sensor_target_counts={"radar": np.array([5.0, 0.0]), "camera": 0.0},
            sensor_concentrations={"radar": 1.0, "camera": 1.0},
            clutter_log_likelihoods=-10.0,
            clutter_weights=0.1,
        )

        radar_probabilities = results["radar"].target_probability_matrix()
        camera_probabilities = results["camera"].target_probability_matrix()

        self.assertGreater(radar_probabilities[0, 0], radar_probabilities[1, 0])
        self.assertGreater(camera_probabilities[1, 0], camera_probabilities[0, 0])
        self.assertEqual(results["radar"].labels[0], HDPAssociationLabel("target", 0))
        self.assertEqual(results["radar"].labels[-2], HDPAssociationLabel("birth"))
        self.assertEqual(results["radar"].labels[-1], HDPAssociationLabel("clutter"))

    def test_birth_and_clutter_handle_no_existing_targets(self):
        results = multisensor_hdp_association(
            {"radar": np.empty((2, 0))},
            global_target_weights=np.empty((0,)),
            global_birth_weight=2.0,
            birth_log_likelihoods={"radar": np.array([0.0, -10.0])},
            clutter_log_likelihoods={"radar": np.array([-10.0, 0.0])},
            clutter_weights={"radar": 2.0},
        )

        result = results["radar"]
        self.assertEqual(result.target_probability_matrix().shape, (0, 2))
        self.assertTrue(np.allclose(result.probabilities.sum(axis=1), 1.0))
        self.assertEqual(result.best_assignments()[0].label.kind, "birth")
        self.assertEqual(result.best_assignments()[1].label.kind, "clutter")

    def test_target_cost_matrix_is_negative_log_target_probability(self):
        result = multisensor_hdp_association(
            {"radar": np.log([[0.8, 0.2], [0.1, 0.9]])},
            global_target_weights=np.array([2.0, 1.0]),
            global_birth_weight=0.5,
            clutter_log_likelihoods=-20.0,
            clutter_weights=1.0,
        )["radar"]

        target_probabilities = result.target_probability_matrix()
        self.assertEqual(target_probabilities.shape, (2, 2))
        self.assertTrue(
            np.allclose(
                result.target_cost_matrix(),
                -np.log(np.clip(target_probabilities, 1e-300, 1.0)),
            )
        )

    def test_survival_prediction_discounts_target_masses(self):
        predicted = predict_survival_weighted_hdp_masses(
            np.array([2.0, 4.0]),
            np.array([0.5, 0.25]),
        )

        self.assertTrue(np.allclose(predicted, np.array([1.0, 1.0])))

    def test_validation_rejects_shape_and_probability_errors(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            multisensor_hdp_association(
                {"radar": np.zeros((1, 3))},
                global_target_weights=np.ones(2),
            )

        with self.assertRaisesRegex(ValueError, "probabilities"):
            multisensor_hdp_association(
                {"radar": np.zeros((1, 2))},
                global_target_weights=np.ones(2),
                detection_probabilities={"radar": np.array([1.1, 0.9])},
            )


if __name__ == "__main__":
    unittest.main()
