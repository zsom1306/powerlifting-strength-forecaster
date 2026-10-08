"""Checks for predictor isolation, category handling, and delta-model evaluation."""

import unittest
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd

from powerlifting_forecaster.evaluation import regression_metrics
from powerlifting_forecaster.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from powerlifting_forecaster.train import MODEL_PARAMETERS, prepare_features, train_regressor


def make_frame(rows, date):
    frame = pd.DataFrame(1.0, index=range(rows), columns=FEATURE_COLUMNS)
    for column, value in zip(CATEGORICAL_FEATURES, ["F", "Raw", "IPF"], strict=True):
        frame[column] = value
    frame["previous_total"] = np.where(np.arange(rows) % 2, 600.0, 400.0)
    frame["delta_total"] = np.where(np.arange(rows) % 2, -40.0, 40.0)
    frame["target_total"] = frame["previous_total"] + frame["delta_total"]
    frame["target_date"] = date
    frame["lifter_name"] = "Excluded identifier"
    return frame


class TrainingTests(unittest.TestCase):
    def test_predictor_isolation_and_training_only_category_levels(self):
        train = make_frame(4, "2022-01-01")
        train.loc[1, "federation"] = "USAPL"
        original = train.copy(deep=True)
        features = prepare_features(train)
        self.assertEqual(features.columns.tolist(), FEATURE_COLUMNS)
        self.assertTrue({"target_total", "delta_total", "target_date", "lifter_name"}.isdisjoint(features))
        changed = train.assign(target_total=-999, delta_total=999, lifter_name="Another name")
        pd.testing.assert_frame_equal(features, prepare_features(changed))
        categories = {column: features[column].cat.categories.tolist() for column in CATEGORICAL_FEATURES}
        validation = make_frame(3, "2023-01-01")
        validation["federation"] = ["USAPL", "UNSEEN", None]
        validation.loc[0, "estimated_age"] = np.nan
        encoded = prepare_features(validation, categories)
        self.assertEqual(encoded["federation"].cat.codes.tolist(), [1, -1, -1])
        self.assertTrue(pd.isna(encoded.loc[0, "estimated_age"]))
        self.assertEqual(encoded["federation"].cat.categories.tolist(), ["IPF", "USAPL"])
        pd.testing.assert_frame_equal(train, original)

    def test_infinite_predictor_is_rejected(self):
        frame = make_frame(2, "2022-01-01")
        for value in [float("inf"), -float("inf")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                prepare_features(frame.assign(estimated_age=value))

    def test_delta_training_metrics_and_serialized_model(self):
        train, validation = make_frame(240, "2022-01-01"), make_frame(40, "2023-01-01")
        validation.index += 500
        validation.loc[500, "federation"] = "UNSEEN"
        validation.loc[501, "estimated_age"] = np.nan
        parameters = {"n_estimators": 60, "learning_rate": 0.2, "num_leaves": 4,
                      "min_child_samples": 5, "n_jobs": 1}
        with patch.dict(MODEL_PARAMETERS, parameters):
            model, report = train_regressor(train, validation)
        restored = lgb.Booster(model_str=model.booster_.model_to_string(num_iteration=model.best_iteration_))
        categories = dict(zip(CATEGORICAL_FEATURES, restored.pandas_categorical, strict=True))
        features = prepare_features(validation, categories)
        predicted = restored.predict(features, num_threads=1, validate_features=True)
        np.testing.assert_allclose(predicted, model.predict(features), rtol=0, atol=1e-10)
        self.assertEqual(restored.feature_name(), FEATURE_COLUMNS)
        self.assertEqual(restored.current_iteration(), report["best_iteration"])
        totals = validation["previous_total"] + pd.Series(predicted, index=validation.index)
        scores = regression_metrics(validation["target_total"], totals)
        for name, value in scores.items():
            self.assertAlmostEqual(report["validation_metrics"]["lightgbm"][name], value)
        self.assertLess(scores["mae_kg"], 1)
        self.assertAlmostEqual(scores["mae_kg"], min(model.evals_result_["validation"]["l1"]), places=5)
        self.assertEqual(report["validation_unseen_category_rows"]["federation"], 1)
        self.assertGreater(report["validation_metrics"]["lightgbm"]["mae_reduction_vs_experience_drift_pct"], 95)

    def test_test_period_and_inconsistent_targets_fail_before_fitting(self):
        train, validation = make_frame(4, "2022-01-01"), make_frame(2, "2023-01-01")
        for invalid in [validation.assign(target_date="2025-01-01"), validation.assign(delta_total=999)]:
            with self.subTest(dates=invalid["target_date"].tolist()):
                with patch("powerlifting_forecaster.train.lgb.LGBMRegressor.fit") as fit:
                    with self.assertRaises(ValueError):
                        train_regressor(train, invalid)
                    fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
