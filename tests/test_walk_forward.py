"""Checks for annual chronology, nested selection, refitting, and pooled metrics."""

import unittest
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd

from powerlifting_forecaster.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from powerlifting_forecaster.evaluation import MODEL_NAMES, score_predictions
from powerlifting_forecaster.train import MODEL_PARAMETERS
from powerlifting_forecaster.walk_forward import (
    evaluate_walk_forward, evaluate_year, split_year,
)


def make_frame():
    frames = []
    for year in range(2018, 2023):
        frame = pd.DataFrame(1.0, index=range(year * 100, year * 100 + 16), columns=FEATURE_COLUMNS)
        frame["sex"], frame["equipment"], frame["federation"] = "F", "Raw", f"FED_{year}"
        frame["target_date"] = f"{year}-06-01"
        frame["previous_total"] = np.tile([400.0, 600.0], 8)
        frame["delta_total"] = np.tile([40.0, -20.0], 8) + year - 2018
        frame["target_total"] = frame["previous_total"] + frame["delta_total"]
        frames.append(frame)
    return pd.concat(frames)


SMALL_PARAMETERS = {"n_estimators": 12, "learning_rate": 0.2, "min_child_samples": 4, "n_jobs": 1}


class WalkForwardTests(unittest.TestCase):
    def test_calendar_boundaries_and_future_rows_are_excluded(self):
        dates = ["2018-12-31", "2019-01-01", "2019-12-31", "2020-01-01", "2020-12-31", "2021-01-01"]
        frame = pd.DataFrame({"target_date": dates, "previous_total": range(6)})
        original = frame.copy(deep=True)
        parts = split_year(frame, 2020)
        self.assertEqual(parts["train"].index.tolist(), [0])
        self.assertEqual(parts["validation"].index.tolist(), [1, 2])
        self.assertEqual(parts["evaluation"].index.tolist(), [3, 4])
        pd.testing.assert_frame_equal(parts["evaluation"], frame.loc[[3, 4]])
        pd.testing.assert_frame_equal(frame, original)

    def test_invalid_periods_dates_and_indices_are_rejected(self):
        frame = make_frame()
        for invalid, year in [(frame, 2023), (frame, 2019), (frame.assign(target_date=None), 2020),
                              (frame.assign(target_date="2025-01-01"), 2020),
                              (pd.concat([frame, frame]), 2020)]:
            with self.subTest(year=year), self.assertRaises(ValueError):
                split_year(invalid, year)
        for years in [(), (2020, 2020), (2021, 2020)]:
            with self.subTest(years=years), self.assertRaises(ValueError):
                evaluate_walk_forward(frame, years)

    def test_every_fit_uses_only_earlier_rows_and_refit_has_no_evaluation_set(self):
        frame = make_frame()
        parts = split_year(frame, 2020)
        refit = pd.concat([parts["train"], parts["validation"]])
        original_fit = lgb.LGBMRegressor.fit
        with patch.dict(MODEL_PARAMETERS, SMALL_PARAMETERS), patch("builtins.print"):
            with patch.object(lgb.LGBMRegressor, "fit", autospec=True, side_effect=original_fit) as fit:
                predictions, report = evaluate_year(frame, 2020)
        self.assertEqual(fit.call_count, 5)
        for call in fit.call_args_list[:4]:
            self.assertEqual(call.args[1].index.tolist(), parts["train"].index.tolist())
            pd.testing.assert_series_equal(call.args[2], parts["train"]["delta_total"])
            self.assertEqual(call.kwargs["eval_X"].index.tolist(), parts["validation"].index.tolist())
            pd.testing.assert_series_equal(call.kwargs["eval_y"], parts["validation"]["delta_total"])
        last = fit.call_args_list[-1]
        self.assertEqual(last.args[1].index.tolist(), refit.index.tolist())
        pd.testing.assert_series_equal(last.args[2], refit["delta_total"])
        self.assertEqual(last.kwargs, {"categorical_feature": CATEGORICAL_FEATURES})
        self.assertEqual(last.args[1]["federation"].cat.categories.tolist(), ["FED_2018", "FED_2019"])
        for call in fit.call_args_list:
            self.assertEqual(call.args[1].columns.tolist(), FEATURE_COLUMNS)
        self.assertEqual(predictions["source_row"].tolist(), parts["evaluation"].index.tolist())
        self.assertEqual(report["refit_parameters"]["n_estimators"], report["selection"]["best_iteration"])
        self.assertEqual(report["baseline_parameters"]["global_drift"], refit["delta_total"].mean())
        self.assertEqual(report["baseline_parameters"]["experience_counts"], {"1": len(refit)})

    def test_evaluation_outcomes_and_future_rows_cannot_change_predictions(self):
        frame = make_frame()
        changed = frame.copy(deep=True)
        evaluation = changed["target_date"].str.startswith("2020")
        changed.loc[evaluation, ["target_total", "delta_total"]] += 1000
        future = changed["target_date"].ge("2021-01-01")
        changed.loc[future, ["target_total", "delta_total", "previous_total", "estimated_age"]] = 99999
        changed.loc[future, "federation"] = "FUTURE_ONLY"
        with patch.dict(MODEL_PARAMETERS, SMALL_PARAMETERS), patch("builtins.print"):
            first_predictions, first = evaluate_year(frame, 2020)
            second_predictions, second = evaluate_year(changed, 2020)
        pd.testing.assert_frame_equal(first_predictions[MODEL_NAMES], second_predictions[MODEL_NAMES])
        for key in ["selection", "refit_parameters", "baseline_parameters"]:
            self.assertEqual(first[key], second[key])
        self.assertNotEqual(first["evaluation_metrics"], second["evaluation_metrics"])

    def test_pooled_metrics_weight_observations_and_score_each_year_once(self):
        def fold_result(frame, year):
            error = 1 if year == 2020 else 3
            rows = 1 if year == 2020 else 3
            predictions = pd.DataFrame({name: [500 + error] * rows for name in MODEL_NAMES})
            predictions["actual_total"] = 500.0
            predictions["evaluation_year"] = year
            return predictions, {"evaluation_year": year, "evaluation_metrics": score_predictions(predictions)}

        with patch("powerlifting_forecaster.walk_forward.evaluate_year", side_effect=fold_result) as evaluate:
            with patch("builtins.print"):
                predictions, report = evaluate_walk_forward(make_frame(), (2020, 2021))
        self.assertEqual([call.args[1] for call in evaluate.call_args_list], [2020, 2021])
        self.assertEqual(len(predictions), report["evaluation_rows"])
        metrics = report["pooled_metrics"]["lightgbm"]
        self.assertEqual(metrics["mae_kg"], 2.5)
        self.assertAlmostEqual(metrics["rmse_kg"], 7 ** 0.5)
        self.assertIsNone(metrics["r2_total"])
        self.assertEqual(report["years_improved_vs_experience_drift"], 0)


if __name__ == "__main__":
    unittest.main()
