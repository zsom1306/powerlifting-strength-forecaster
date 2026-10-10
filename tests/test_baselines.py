"""Checks for training-only drift estimation and chronological baseline evaluation."""

import unittest

import pandas as pd

from powerlifting_forecaster.baselines import DriftBaselines, evaluate_baselines, experience_groups


def make_frame(counts, changes, date):
    frame = pd.DataFrame({"prior_meets": counts, "delta_total": changes})
    frame["target_date"], frame["previous_total"] = date, 500.0
    frame["target_total"] = frame["previous_total"] + frame["delta_total"]
    return frame


class BaselineTests(unittest.TestCase):
    def test_fitted_means_prediction_alignment_and_unseen_group_fallback(self):
        train = make_frame([1, 1, 3, 4], [10, 30, -10, 10], "2022-01-01")
        model = DriftBaselines.fit(train)
        self.assertEqual(model.global_drift, 10)
        self.assertEqual(model.experience_drift, {"1": 20, "3-4": 0})
        self.assertEqual(model.experience_counts, {"1": 2, "3-4": 2})
        features = pd.DataFrame({"previous_total": [500, 600, 700], "prior_meets": [1, 4, 100]},
                                index=[9, 2, 7])
        predictions = model.predict(features)
        self.assertEqual(predictions.index.tolist(), [9, 2, 7])
        self.assertEqual(predictions["persistence"].tolist(), [500, 600, 700])
        self.assertEqual(predictions["global_drift"].tolist(), [510, 610, 710])
        self.assertEqual(predictions["experience_drift"].tolist(), [520, 600, 710])

    def test_experience_boundaries_and_invalid_counts(self):
        counts = pd.Series([1, 2, 3, 4, 5, 8, 9, 16, 17, 100])
        self.assertEqual(experience_groups(counts).tolist(),
                         ["1", "2", "3-4", "3-4", "5-8", "5-8", "9-16", "9-16", "17+", "17+"])
        for value in [0, -1, 1.5, float("inf"), float("nan")]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    experience_groups(pd.Series([value]))

    def test_validation_outcomes_cannot_change_fitted_parameters_or_predictions(self):
        train = make_frame([1, 1, 3, 4], [10, 30, -10, 10], "2022-01-01")
        validation = make_frame([1, 3], [20, 5], "2023-01-01")
        original = train.copy(deep=True)
        first = evaluate_baselines(train, validation)
        changed = validation.copy(deep=True)
        changed["target_total"] += 1000
        changed["delta_total"] += 1000
        second = evaluate_baselines(train, changed)
        self.assertEqual(first["fitted_parameters"], second["fitted_parameters"])
        model = DriftBaselines.fit(train)
        pd.testing.assert_frame_equal(model.predict(validation), model.predict(changed))
        pd.testing.assert_frame_equal(train, original)
        scores = first["validation_metrics"]
        self.assertEqual(scores["persistence"]["mae_kg"], 12.5)
        self.assertEqual(scores["global_drift"]["mae_kg"], 7.5)
        self.assertEqual(scores["experience_drift"]["mae_kg"], 2.5)
        self.assertAlmostEqual(scores["experience_drift"]["mae_reduction_vs_persistence_pct"], 80)
        self.assertEqual(first["best_validation_baseline"], "experience_drift")

    def test_future_periods_and_inconsistent_targets_are_rejected(self):
        train = make_frame([1, 2], [10, 20], "2022-01-01")
        validation = make_frame([1, 2], [20, 10], "2023-01-01")
        for bad_train, bad_validation in [
            (train.assign(target_date="2023-01-01"), validation),
            (train, validation.assign(target_date="2025-01-01")),
            (train, validation.assign(target_date=None)),
            (train, validation.assign(delta_total=999)),
        ]:
            with self.subTest(train_dates=bad_train["target_date"].tolist(),
                              validation_dates=bad_validation["target_date"].tolist()):
                with self.assertRaises(ValueError):
                    evaluate_baselines(bad_train, bad_validation)

    def test_perfect_persistence_has_undefined_relative_improvement(self):
        train = make_frame([1, 2], [0, 0], "2022-01-01")
        validation = make_frame([1, 2], [0, 0], "2023-01-01")
        report = evaluate_baselines(train, validation)
        for metrics in report["validation_metrics"].values():
            self.assertEqual(metrics["mae_kg"], 0)
            self.assertIsNone(metrics["mae_reduction_vs_persistence_pct"])

    def test_custom_periods_preserve_chronology_and_final_test_boundary(self):
        train = make_frame([1, 2], [10, 20], "2018-01-01")
        validation = make_frame([1, 2], [20, 10], "2019-01-01")
        report = evaluate_baselines(train, validation, "2019-01-01", "2020-01-01")
        self.assertEqual(report["validation_start"], "2019-01-01")
        self.assertEqual(report["test_start"], "2020-01-01")
        for start, end in [("2020-01-01", "2019-01-01"), ("2018-01-01", "2020-01-01"),
                           ("2019-01-01", "2019-01-01"), ("2019-01-01", "2026-01-01")]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                evaluate_baselines(train, validation, start, end)


if __name__ == "__main__":
    unittest.main()
