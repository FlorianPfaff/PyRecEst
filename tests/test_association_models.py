import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member,redefined-builtin
from pyrecest.backend import (
    all,
    argmax,
    argmin,
    array,
    column_stack,
    concatenate,
    isfinite,
    max,
    mean,
    min,
    ones,
    random,
    vstack,
    zeros,
)
from pyrecest.utils import (
    CalibratedPairwiseAssociationModel,
    LogisticPairwiseAssociationModel,
    NamedPairwiseFeatureSchema,
    pairwise_feature_tensor,
)


class TestLogisticPairwiseAssociationModel(unittest.TestCase):
    def setUp(self):
        random.seed(7)
        positive_examples = column_stack(
            [
                0.2 + 0.08 * random.normal(size=(80,)),
                0.85 + 0.06 * random.normal(size=(80,)),
            ]
        )
        negative_examples = column_stack(
            [
                2.4 + 0.25 * random.normal(size=(80,)),
                0.12 + 0.05 * random.normal(size=(80,)),
            ]
        )
        self.training_features = vstack([positive_examples, negative_examples])
        self.training_labels = concatenate(
            [
                ones(len(positive_examples), dtype=int),
                zeros(len(negative_examples), dtype=int),
            ]
        )

    def test_fit_separates_match_and_nonmatch_examples(self):
        model = LogisticPairwiseAssociationModel(class_weight="balanced")
        model.fit(self.training_features, self.training_labels)

        positive_probabilities = model.predict_match_probability(
            self.training_features[:80]
        )
        negative_probabilities = model.predict_match_probability(
            self.training_features[80:]
        )

        self.assertTrue(model.converged_)
        self.assertGreater(mean(positive_probabilities), 0.95)
        self.assertLess(mean(negative_probabilities), 0.05)
        self.assertLess(max(negative_probabilities), min(positive_probabilities))

    def test_pairwise_cost_matrix_prefers_more_match_like_pairs(self):
        model = LogisticPairwiseAssociationModel(class_weight="balanced")
        model.fit(self.training_features, self.training_labels)

        pairwise_features = array(
            [
                [[0.15, 0.92], [2.2, 0.15]],
                [[2.6, 0.08], [0.18, 0.88]],
            ]
        )

        costs = model.pairwise_cost_matrix(pairwise_features)
        probabilities = model.predict_match_probability(pairwise_features)

        self.assertEqual(costs.shape, (2, 2))
        npt.assert_array_equal(argmin(costs, axis=1), array([0, 1]))
        npt.assert_array_equal(argmax(probabilities, axis=1), array([0, 1]))
        self.assertLess(costs[0, 0], costs[0, 1])
        self.assertLess(costs[1, 1], costs[1, 0])

    def test_flattened_pairwise_training_tensor_is_supported(self):
        model = LogisticPairwiseAssociationModel(class_weight="balanced")

        training_tensor = array(
            [
                [[0.1, 0.95], [2.1, 0.10], [2.4, 0.12]],
                [[2.3, 0.06], [0.2, 0.90], [2.0, 0.20]],
            ]
        )
        label_matrix = array([[1, 0, 0], [0, 1, 0]])

        model.fit(training_tensor, label_matrix)

        probabilities = model.predict_match_probability(training_tensor)
        npt.assert_array_equal(argmax(probabilities, axis=1), array([0, 1]))

    def test_constant_feature_is_handled_during_standardization(self):
        model = LogisticPairwiseAssociationModel(class_weight="balanced")

        constant_augmented_features = column_stack(
            [self.training_features, ones(self.training_features.shape[0])]
        )
        model.fit(constant_augmented_features, self.training_labels)

        probabilities = model.predict_match_probability(constant_augmented_features)
        self.assertTrue(all(isfinite(probabilities)))
        self.assertGreater(mean(probabilities[:80]), 0.9)
        self.assertLess(mean(probabilities[80:]), 0.1)

    def test_fit_without_intercept_is_supported(self):
        model = LogisticPairwiseAssociationModel(
            class_weight="balanced", fit_intercept=False
        )
        model.fit(self.training_features, self.training_labels)

        probabilities = model.predict_match_probability(self.training_features)
        self.assertTrue(all(isfinite(probabilities)))
        self.assertGreater(mean(probabilities[:80]), 0.9)
        self.assertLess(mean(probabilities[80:]), 0.1)

    def test_invalid_labels_raise(self):
        model = LogisticPairwiseAssociationModel()
        with self.assertRaises(ValueError):
            model.fit(
                self.training_features, zeros(self.training_labels.shape, dtype=int) + 2
            )

    def test_pairwise_feature_tensor_stacks_named_components_and_sanitizes(self):
        components = {
            "distance": array([[0.0, float("inf")], [float("nan"), -float("inf")]]),
            "overlap": array([[0.9, 0.2], [0.1, 0.8]]),
        }

        features = pairwise_feature_tensor(
            components,
            ("distance", "one_minus_overlap"),
            transforms={"one_minus_overlap": lambda item: 1.0 - item["overlap"]},
        )

        self.assertEqual(features.shape, (2, 2, 2))
        npt.assert_allclose(
            features[:, :, 0], array([[0.0, 1.0e6], [0.0, -1.0e6]])
        )
        npt.assert_allclose(features[:, :, 1], array([[0.1, 0.8], [0.9, 0.2]]))

    def test_named_pairwise_feature_schema_validates_feature_layout(self):
        schema = NamedPairwiseFeatureSchema(("distance", "similarity"))
        self.assertEqual(schema.feature_index("similarity"), 1)

        with self.assertRaises(KeyError):
            schema.build_tensor({"distance": array([[1.0]])})
        with self.assertRaises(ValueError):
            pairwise_feature_tensor(
                {
                    "distance": array([[1.0, 2.0]]),
                    "similarity": array([[1.0], [2.0]]),
                },
                ("distance", "similarity"),
            )

    def test_calibrated_pairwise_association_model_uses_named_components(self):
        model = LogisticPairwiseAssociationModel(class_weight="balanced")
        model.fit(self.training_features, self.training_labels)
        calibrated_model = CalibratedPairwiseAssociationModel(
            model, feature_names=("distance", "similarity")
        )
        components = {
            "distance": array([[0.15, 2.2], [2.6, 0.18]]),
            "similarity": array([[0.92, 0.15], [0.08, 0.88]]),
        }

        probabilities = calibrated_model.pairwise_probability_matrix_from_components(
            components
        )
        costs = calibrated_model.pairwise_cost_matrix_from_components(components)

        self.assertEqual(probabilities.shape, (2, 2))
        self.assertEqual(costs.shape, (2, 2))
        npt.assert_array_equal(argmax(probabilities, axis=1), array([0, 1]))
        npt.assert_array_equal(argmin(costs, axis=1), array([0, 1]))


if __name__ == "__main__":
    unittest.main()
