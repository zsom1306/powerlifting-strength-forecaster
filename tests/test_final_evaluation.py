"""Checks for frozen selection, final-test isolation, and reproducible saved predictions."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd

from powerlifting_forecaster.evaluation import MODEL_NAMES, score_predictions
from powerlifting_forecaster.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from powerlifting_forecaster.final_evaluation import (
    evaluate_test, file_sha256, fit_final_model, frozen_parameters, main, validate_rows,
)
from powerlifting_forecaster.train import MODEL_PARAMETERS, prepare_features


def make_frame(date):
    frame = pd.DataFrame(1.0, index=range(40), columns=FEATURE_COLUMNS)
    frame["sex"], frame["equipment"], frame["federation"] = "F", "Raw", "IPF"
    frame["target_date"] = date
    frame["previous_total"] = np.tile([400.0, 600.0], 20)
    frame["delta_total"] = np.tile([40.0, -20.0], 20)
    frame["target_total"] = frame["previous_total"] + frame["delta_total"]
    return frame


def make_selection():
    return {
        "feature_columns": FEATURE_COLUMNS.copy(), "categorical_features": CATEGORICAL_FEATURES.copy(),
        "validation_start": "2023-01-01", "test_start": "2025-01-01",
        "selected_candidate": "fixture", "best_iteration": 8,
        "model_parameters": MODEL_PARAMETERS | {"n_estimators": 12, "min_child_samples": 4, "n_jobs": 1},
    }


class FinalEvaluationTests(unittest.TestCase):
    def test_refit_uses_only_past_rows_and_fixed_trees_without_an_evaluation_set(self):
        refit = pd.concat([make_frame("2022-12-31"), make_frame("2024-12-31")], ignore_index=True)
        refit.loc[40:, "federation"] = "VALIDATION_ONLY"
        refit.loc[40:, ["target_total", "delta_total"]] += 10
        selection = make_selection()
        original = copy.deepcopy(selection)
        with patch.object(lgb.LGBMRegressor, "fit", autospec=True, side_effect=lgb.LGBMRegressor.fit) as fit:
            model, baselines, report = fit_final_model(refit, selection)
        fit.assert_called_once()
        self.assertEqual(fit.call_args.args[1].columns.tolist(), FEATURE_COLUMNS)
        pd.testing.assert_series_equal(fit.call_args.args[2], refit["delta_total"])
        self.assertEqual(fit.call_args.kwargs, {"categorical_feature": CATEGORICAL_FEATURES})
        self.assertEqual(fit.call_args.args[1]["federation"].cat.categories.tolist(), ["IPF", "VALIDATION_ONLY"])
        self.assertEqual(model.n_estimators, 8)
        self.assertEqual(report["refit_tree_count"], 8)
        self.assertEqual(report["refit_rows"], 80)
        self.assertEqual(baselines.global_drift, refit["delta_total"].mean())
        self.assertEqual(baselines.experience_counts, {"1": 80})
        self.assertEqual(selection, original)

    def test_test_outcomes_cannot_change_predictions_or_category_levels(self):
        model, baselines, _ = fit_final_model(make_frame("2024-12-31"), make_selection())
        test = make_frame("2025-01-01")
        test.index += 100
        test.loc[100, "federation"] = "TEST_ONLY"
        test.loc[101, "federation"] = None
        changed = test.copy(deep=True)
        changed[["target_total", "delta_total"]] += 1000
        with patch.object(lgb.LGBMRegressor, "fit") as fit:
            first, report = evaluate_test(model, baselines, test)
            second, changed_report = evaluate_test(model, baselines, changed)
        fit.assert_not_called()
        pd.testing.assert_frame_equal(first[MODEL_NAMES], second[MODEL_NAMES])
        self.assertEqual(first["source_row"].tolist(), test.index.tolist())
        self.assertEqual(report["test_unseen_category_rows"]["federation"], 1)
        self.assertEqual(model.booster_.pandas_categorical[-1], ["IPF"])
        self.assertNotEqual(report["test_metrics"], changed_report["test_metrics"])
        np.testing.assert_allclose(first["lightgbm"], test["previous_total"] + first["predicted_delta"])

    def test_invalid_periods_targets_and_selection_fail_before_fitting(self):
        frame = make_frame("2024-12-31")
        invalid_frames = [frame.iloc[:0], pd.concat([frame, frame]), frame.assign(target_date=None),
                          frame.assign(target_date="2025-01-01"), frame.assign(delta_total=999),
                          frame.assign(previous_total=-1), frame.assign(target_total=np.inf)]
        with patch.object(lgb.LGBMRegressor, "fit") as fit:
            for invalid in invalid_frames:
                with self.subTest(rows=len(invalid)), self.assertRaises(ValueError):
                    fit_final_model(invalid, make_selection())
            fit.assert_not_called()
        with self.assertRaises(ValueError):
            validate_rows(frame, test=True)
        for changes in [{"best_iteration": 0}, {"best_iteration": 13}, {"best_iteration": True},
                        {"feature_columns": ["target_total"]}, {"categorical_features": []},
                        {"test_start": "2024-01-01"}, {"validation_start": "2022-01-01"}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                frozen_parameters(make_selection() | changes)

    def test_cli_fits_before_reading_test_and_saved_model_reproduces_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, date in [("train", "2022-12-31"), ("validation", "2024-12-31"), ("test", "2025-01-01")]:
                make_frame(date).to_csv(root / f"{name}.csv", index=False)
            selection = make_selection() | {"inputs": {
                name: {"file": f"{name}.csv", "sha256": file_sha256(root / f"{name}.csv")}
                for name in ["train", "validation"]
            }}
            selection_path = root / "selection.json"
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            model_path, prediction_path, report_path = root / "model.txt", root / "predictions.csv", root / "report.json"
            argv = ["final_evaluation", "--data-dir", str(root), "--selection", str(selection_path),
                    "--model", str(model_path), "--predictions", str(prediction_path), "--report", str(report_path)]
            events = []
            original_fit, original_read = lgb.LGBMRegressor.fit, pd.read_csv

            def fit_spy(*args, **kwargs):
                result = original_fit(*args, **kwargs)
                events.append("fit_complete")
                return result

            def read_spy(path, **kwargs):
                events.append(Path(path).name)
                return original_read(path, **kwargs)

            with patch("sys.argv", argv), patch("builtins.print"):
                with patch.object(lgb.LGBMRegressor, "fit", autospec=True, side_effect=fit_spy):
                    with patch("pandas.read_csv", side_effect=read_spy):
                        main()
                self.assertEqual(events, ["train.csv", "validation.csv", "fit_complete", "test.csv"])
                with self.assertRaises(FileExistsError):
                    main()
            report = json.loads(report_path.read_text(encoding="utf-8"))
            predictions = pd.read_csv(prediction_path)
            self.assertEqual(report["selection_sha256"], file_sha256(selection_path))
            self.assertEqual(report["model_sha256"], file_sha256(model_path))
            self.assertEqual(report["prediction_sha256"], file_sha256(prediction_path))
            self.assertEqual(report["inputs"]["test"]["sha256"], file_sha256(root / "test.csv"))
            restored = lgb.Booster(model_file=str(model_path))
            categories = dict(zip(CATEGORICAL_FEATURES, restored.pandas_categorical, strict=True))
            delta = restored.predict(prepare_features(make_frame("2025-01-01"), categories), num_threads=1)
            np.testing.assert_allclose(delta, predictions["predicted_delta"], rtol=0, atol=1e-10)
            for name, scores in score_predictions(predictions).items():
                for metric, value in scores.items():
                    self.assertAlmostEqual(value, report["test_metrics"][name][metric])
            argv[-1] = str(root / "another_report.json")
            argv[argv.index("--model") + 1] = str(root / "another_model.txt")
            argv[argv.index("--predictions") + 1] = str(root / "another_predictions.csv")
            selection["inputs"]["train"]["sha256"] = "mismatched"
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            with patch("sys.argv", argv), patch.object(lgb.LGBMRegressor, "fit") as fit:
                with self.assertRaisesRegex(ValueError, "inputs used for model selection"):
                    main()
                fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
