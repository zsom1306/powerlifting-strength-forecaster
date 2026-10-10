"""Checks for validation-based candidate selection and fixed experiment scope."""

import unittest
from unittest.mock import patch

import pandas as pd

from powerlifting_forecaster.train import MODEL_PARAMETERS
from powerlifting_forecaster.splits import TEST_START, VALIDATION_START
from powerlifting_forecaster.tune import CANDIDATES, tune_regressor


class TuningTests(unittest.TestCase):
    def run_candidates(self, errors, iterations, **boundaries):
        train, validation = pd.DataFrame({"period": ["train"]}), pd.DataFrame({"period": ["validation"]})
        models = [object() for _ in CANDIDATES]
        results = []
        for index, overrides in enumerate(CANDIDATES.values()):
            results.append((models[index], {
                "model_parameters": MODEL_PARAMETERS | overrides,
                "best_iteration": iterations[index], "evaluated_iterations": iterations[index] + 50,
                "validation_metrics": {"lightgbm": {
                    "mae_kg": errors[index], "rmse_kg": 40 + index, "r2_total": 0.95 - index * 0.01,
                }},
            }))
        with patch("powerlifting_forecaster.tune.train_regressor", side_effect=results) as fit:
            with patch("builtins.print"):
                model, report = tune_regressor(train, validation, **boundaries)
        self.assertEqual(fit.call_count, 4)
        for call, overrides in zip(fit.call_args_list, CANDIDATES.values(), strict=True):
            self.assertIs(call.args[0], train)
            self.assertIs(call.args[1], validation)
            self.assertEqual(call.kwargs, {
                "parameters": overrides, "validation_start": VALIDATION_START,
                "test_start": TEST_START, **boundaries,
            })
        return models.index(model), report

    def test_lowest_mae_wins_even_when_other_metrics_are_worse(self):
        selected, report = self.run_candidates([25, 24, 22, 23], [100, 200, 300, 400])
        self.assertEqual(selected, 2)
        self.assertEqual(report["selected_candidate"], "leaves_127")
        self.assertAlmostEqual(report["mae_reduction_vs_initial_model_pct"], 12)
        self.assertEqual(len(report["tuning_trials"]), 4)
        self.assertEqual([trial["validation_metrics"]["mae_kg"] for trial in report["tuning_trials"]],
                         [25, 24, 22, 23])

    def test_initial_model_is_retained_when_candidates_do_not_improve(self):
        selected, report = self.run_candidates([23, 24, 25, 26], [100, 100, 100, 100])
        self.assertEqual(selected, 0)
        self.assertEqual(report["mae_reduction_vs_initial_model_pct"], 0)

    def test_exact_ties_prefer_fewer_leaves_then_fewer_trees(self):
        selected, _ = self.run_candidates([23, 23, 24, 24], [200, 100, 100, 100])
        self.assertEqual(selected, 0)
        selected, _ = self.run_candidates([25, 24, 23, 23], [100, 100, 200, 100])
        self.assertEqual(selected, 3)

    def test_perfect_initial_model_has_undefined_relative_reduction(self):
        selected, report = self.run_candidates([0, 1, 2, 3], [100, 100, 100, 100])
        self.assertEqual(selected, 0)
        self.assertIsNone(report["mae_reduction_vs_initial_model_pct"])

    def test_historical_boundaries_reach_every_candidate(self):
        self.run_candidates([25, 24, 23, 22], [100, 100, 100, 100],
                            validation_start="2019-01-01", test_start="2020-01-01")


if __name__ == "__main__":
    unittest.main()
