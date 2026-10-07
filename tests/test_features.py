"""Checks for prior-only strength summaries, age estimates, and registered meet context."""

import unittest

import pandas as pd

from powerlifting_forecaster.features import FEATURE_COLUMNS, build_features
from powerlifting_forecaster.histories import INPUT_COLUMNS as HISTORY_INPUT_COLUMNS


def make_frame():
    dates = ["2020-01-01", "2020-01-11", "2020-02-10", "2020-02-20", "2020-03-01", "2020-03-21"]
    totals = [600, 400, 500, 700, 450, 900]
    rows = [["Lifter", date, total, 200, 100, total - 300] for date, total in zip(dates, totals)]
    rows += [["Lifter #1", "2020-01-01", 300, 100, 75, 125],
             ["Lifter #1", "2020-01-31", 330, 110, 80, 140]]
    frame = pd.DataFrame(rows, columns=HISTORY_INPUT_COLUMNS)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Age"] = [20, None, 21, None, 0, 22, None, 30]
    frame["Sex"], frame["Equipment"], frame["Federation"], frame["Tested"] = "M", "Raw", "A", "Yes"
    frame.loc[3, ["Equipment", "Federation", "Tested"]] = ["Wraps", "B", None]
    frame.loc[frame["Name"].eq("Lifter #1"), "Sex"] = "F"
    return frame


class FeatureTests(unittest.TestCase):
    def test_historical_summaries_age_and_registered_context(self):
        frame = make_frame()
        original = frame.copy(deep=True)
        results, audit = build_features(frame)
        shuffled, _ = build_features(frame.sample(frac=1, random_state=17))
        pd.testing.assert_frame_equal(results, shuffled)
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(results["historical_best_total"].tolist(), [600, 600, 600, 700, 700, 300])
        for actual, expected in zip(results["recent_mean_total_3"], [600, 500, 500, 1600 / 3, 550, 300]):
            self.assertAlmostEqual(actual, expected)
        self.assertTrue(results.loc[[0, 5], "mean_prior_gap_days"].isna().all())
        for actual, expected in zip(results["mean_prior_gap_days"].dropna(), [10, 20, 50 / 3, 15]):
            self.assertAlmostEqual(actual, expected)
        ages = [20 + 10 / 365.25, 20 + 40 / 365.25, 21 + 10 / 365.25,
                21 + 20 / 365.25, 21 + 40 / 365.25]
        for actual, expected in zip(results["estimated_age"].iloc[:5], ages):
            self.assertAlmostEqual(actual, expected)
        self.assertTrue(pd.isna(results.loc[5, "estimated_age"]))
        self.assertEqual(results.loc[2, ["equipment", "federation", "tested_category"]].tolist(), ["Wraps", "B", 0])
        self.assertEqual(results.loc[5, "sex"], "F")
        self.assertEqual(audit["output_rows"], 6)
        self.assertEqual(len(FEATURE_COLUMNS), 15)
        self.assertTrue({"target_total", "delta_total", "lifter_name", "Age", "BodyweightKg"}.isdisjoint(FEATURE_COLUMNS))

    def test_current_and_future_results_cannot_change_predictors(self):
        frame = make_frame()
        expected, _ = build_features(frame)
        outcome_columns = [*HISTORY_INPUT_COLUMNS[2:], "Age"]
        for date in expected["target_date"].unique():
            with self.subTest(date=date):
                changed = frame.copy(deep=True)
                mask = changed["Date"].ge(date)
                changed.loc[mask, outcome_columns] = changed.loc[mask, outcome_columns].fillna(0) + 1000
                changed["BodyweightKg"], changed["Place"], changed["Dots"] = 999, "DQ", 9999
                future = changed["Date"].gt(date)
                changed.loc[future, ["Sex", "Equipment", "Federation", "Tested"]] = ["Mx", "Single-ply", "Z", None]
                actual, _ = build_features(changed)
                observed = expected["target_date"].le(date)
                pd.testing.assert_frame_equal(actual.loc[observed, FEATURE_COLUMNS], expected.loc[observed, FEATURE_COLUMNS])

    def test_truncated_histories_match_full_features(self):
        frame = make_frame()
        complete, _ = build_features(frame)
        for date in complete["target_date"].unique():
            with self.subTest(date=date):
                prefix, _ = build_features(frame.loc[frame["Date"].le(date)])
                expected = complete.loc[complete["target_date"].le(date)].reset_index(drop=True)
                pd.testing.assert_frame_equal(prefix, expected)

    def test_unusable_prior_ages_stay_missing(self):
        for age in [0, -1, float("inf"), None]:
            with self.subTest(age=age):
                frame = make_frame().iloc[:2].copy()
                frame["Age"] = [age, 25]
                results, _ = build_features(frame)
                self.assertTrue(results["estimated_age"].isna().all())

    def test_invalid_tested_values_are_rejected(self):
        with self.assertRaises(ValueError):
            build_features(make_frame().assign(Tested="unexpected"))

    def test_first_meets_and_empty_inputs_have_no_prediction_rows(self):
        frame = make_frame().drop_duplicates("Name")
        for subset in [frame, frame.iloc[:0]]:
            with self.subTest(rows=len(subset)):
                results, _ = build_features(subset)
                self.assertTrue(results.empty)
                self.assertTrue(set(FEATURE_COLUMNS).issubset(results.columns))


if __name__ == "__main__":
    unittest.main()
