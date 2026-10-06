"""Checks for metric values, row alignment, and invalid predictions."""

import unittest

import pandas as pd

from powerlifting_forecaster.evaluation import regression_metrics


class RegressionMetricTests(unittest.TestCase):
    def test_known_errors_and_delta_total_equivalence(self):
        actual, predicted = pd.Series([500, 520, 540]), pd.Series([510, 500, 540])
        scores = regression_metrics(actual, predicted)
        self.assertEqual(scores["mae_kg"], 10)
        self.assertAlmostEqual(scores["rmse_kg"], (500 / 3) ** 0.5)
        self.assertAlmostEqual(scores["r2_total"], 0.375)
        previous = pd.Series([490, 510, 525])
        delta_scores = regression_metrics(actual - previous, predicted - previous)
        self.assertEqual(scores["mae_kg"], delta_scores["mae_kg"])
        self.assertEqual(scores["rmse_kg"], delta_scores["rmse_kg"])

    def test_constant_targets_have_undefined_r_squared(self):
        self.assertIsNone(regression_metrics(pd.Series([500, 500]), pd.Series([490, 510]))["r2_total"])

    def test_misaligned_missing_and_infinite_predictions_are_rejected(self):
        actual = pd.Series([500, 520])
        for predicted in [pd.Series([500, 520], index=[1, 0]), pd.Series([500, None]),
                          pd.Series([500, float("inf")])]:
            with self.subTest(predicted=predicted.tolist()):
                with self.assertRaises(ValueError):
                    regression_metrics(actual, predicted)


if __name__ == "__main__":
    unittest.main()
