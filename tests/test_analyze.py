"""Checks for diagnostic alignment, complete groups, and fixed-model permutation scoring."""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from powerlifting_forecaster.analyze import (
    GAP_LABELS, align_predictions, error_summary, gap_groups, grouped_errors,
    permutation_importance, training_progression,
)
from powerlifting_forecaster.baselines import experience_groups
from powerlifting_forecaster.features import FEATURE_COLUMNS


def make_data():
    frame = pd.DataFrame(1.0, index=range(20), columns=FEATURE_COLUMNS)
    frame["sex"], frame["equipment"] = "F", "Raw"
    frame["federation"] = np.tile(["A", "B"], 10)
    frame["previous_total"] = 400.0 + np.arange(20) * 10
    frame["prior_meets"] = np.arange(1, 21)
    frame["days_since_previous_meet"] = np.tile([1, 90, 91, 180, 181, 365, 366, 730, 731, 1000], 2)
    frame["target_date"] = "2025-01-01"
    frame["delta_total"] = frame["previous_total"] / 10 + np.where(frame["federation"].eq("A"), 5, -5)
    frame["target_total"] = frame["previous_total"] + frame["delta_total"]
    error = np.tile([-10, 20], 10)
    predictions = pd.DataFrame({
        "source_row": frame.index, "target_date": frame["target_date"],
        "actual_total": frame["target_total"], "actual_delta": frame["delta_total"],
        "previous_total": frame["previous_total"], "predicted_delta": frame["delta_total"] + error,
        "lightgbm": frame["target_total"] + error, "experience_drift": frame["target_total"] + 30,
        "global_drift": frame["previous_total"] + 10, "persistence": frame["previous_total"],
    })
    return frame, predictions


class AnalysisTests(unittest.TestCase):
    def test_saved_predictions_must_match_rows_dates_targets_and_reconstruction(self):
        frame, predictions = make_data()
        diagnostic = align_predictions(frame, predictions)
        np.testing.assert_array_equal(diagnostic["error"], np.tile([-10, 20], 10))
        for invalid in [predictions.iloc[::-1], predictions.iloc[:-1], predictions.assign(source_row=0),
                        predictions.assign(target_date="2026-01-01"), predictions.assign(actual_total=999),
                        predictions.assign(actual_delta=999), predictions.assign(previous_total=999),
                        predictions.assign(predicted_delta=999), predictions.assign(lightgbm=np.inf)]:
            with self.subTest(rows=len(invalid)), self.assertRaises(ValueError):
                align_predictions(frame, invalid)

    def test_gap_boundaries_and_group_denominators_cover_every_meet(self):
        days = pd.Series([1, 90, 91, 180, 181, 365, 366, 730, 731, 100000])
        self.assertEqual(gap_groups(days).tolist(), [label for label in GAP_LABELS for _ in range(2)])
        for invalid in [0, -1, 1.5, np.nan, np.inf]:
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                gap_groups(pd.Series([invalid]))
        frame, predictions = make_data()
        diagnostic = align_predictions(frame, predictions)
        for groups in [experience_groups(frame["prior_meets"]), gap_groups(frame["days_since_previous_meet"])]:
            rows = grouped_errors(diagnostic, groups)
            self.assertEqual(sum(row["rows"] for row in rows), len(frame))
            weighted_mae = sum(row["rows"] * row["lightgbm_mae_kg"] for row in rows) / len(frame)
            self.assertEqual(weighted_mae, 15)
        with self.assertRaises(ValueError):
            grouped_errors(diagnostic, pd.Series([None] * len(frame)))

    def test_error_summary_and_training_only_progression(self):
        frame, predictions = make_data()
        diagnostic = align_predictions(frame, predictions)
        summary = error_summary(diagnostic)
        self.assertEqual(summary["lightgbm_mae_kg"], 15)
        self.assertAlmostEqual(summary["lightgbm_rmse_kg"], np.sqrt(250))
        self.assertEqual(summary["bias_kg"], 5)
        self.assertEqual(summary["p90_absolute_error_kg"], 20)
        self.assertEqual(summary["within_25_kg_pct"], 100)
        self.assertEqual(summary["mae_reduction_vs_baseline_pct"], 50)
        self.assertIsNone(error_summary(diagnostic.assign(baseline_error=0))["mae_reduction_vs_baseline_pct"])
        for invalid in [diagnostic.iloc[:0], diagnostic.assign(error=np.nan)]:
            with self.assertRaises(ValueError):
                error_summary(invalid)
        train = frame.assign(target_date="2022-12-31")
        progression = training_progression(train)
        self.assertEqual(sum(row["rows"] for row in progression), len(train))
        self.assertEqual(progression[0]["mean_delta_kg"], train.loc[0, "delta_total"])
        with self.assertRaises(ValueError):
            training_progression(train.assign(target_date="2023-01-01"))

    def test_permutations_preserve_categories_and_use_fixed_delta_targets(self):
        frame, _ = make_data()
        original = frame.copy(deep=True)
        model = Mock(spec=["pandas_categorical", "predict"])
        model.pandas_categorical = [["F"], ["Raw"], ["A", "B"]]

        def predict(features, **kwargs):
            self.assertEqual(features.columns.tolist(), FEATURE_COLUMNS)
            self.assertEqual(features["federation"].cat.categories.tolist(), ["A", "B"])
            return (features["previous_total"] / 10 + np.where(features["federation"].eq("A"), 5, -5)).to_numpy()

        model.predict.side_effect = predict
        with patch("builtins.print"):
            report = permutation_importance(model, frame, frame["delta_total"], sample_size=12, repeats=2)
            repeated = permutation_importance(model, frame, frame["delta_total"], sample_size=12, repeats=2)
        self.assertEqual(report, repeated)
        self.assertEqual(report["sample_rows"], 12)
        self.assertEqual(report["sample_original_mae_kg"], 0)
        rows = {row["feature"]: row for row in report["features"]}
        self.assertEqual(rows["estimated_age"]["mean_mae_increase_kg"], 0)
        self.assertGreater(rows["federation"]["mean_mae_increase_kg"], 0)
        rng = np.random.default_rng(42)
        sample = frame.iloc[np.sort(rng.choice(len(frame), size=12, replace=False))]
        expected = [float(np.abs(sample["previous_total"].to_numpy()
                                 - sample["previous_total"].to_numpy()[rng.permutation(12)]).mean() / 10)
                    for _ in range(2)]
        np.testing.assert_allclose(rows["previous_total"]["repeat_increases_kg"], expected)
        np.testing.assert_allclose(rows["previous_total"]["std_mae_increase_kg"], np.std(expected))
        pd.testing.assert_frame_equal(frame, original)
        with self.assertRaisesRegex(ValueError, "reproduce"):
            permutation_importance(model, frame, frame["delta_total"] + 1)


if __name__ == "__main__":
    unittest.main()
