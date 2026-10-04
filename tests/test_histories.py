"""Checks for chronological alignment, lifter isolation, and result leakage."""

import unittest

import pandas as pd

from powerlifting_forecaster.histories import FEATURE_COLUMNS, INPUT_COLUMNS, build_histories


def make_frame():
    frame = pd.DataFrame([
        ["Lifter", "2020-01-01", 500, 200, 100, 200],
        ["Lifter", "2020-01-11", 520, 210, None, 210],
        ["Lifter", "2020-02-10", 510, 200, 110, 200],
        ["Lifter #1", "2020-01-02", 300, 100, 75, 125],
        ["Lifter #1", "2020-01-22", 330, 110, 80, 140],
        ["Single", "2020-01-01", 400, 150, 100, 150],
    ], columns=INPUT_COLUMNS)
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame


class HistoryTests(unittest.TestCase):
    def test_alignment_missing_lifts_and_lifter_isolation(self):
        frame = make_frame()
        original = frame.copy(deep=True)
        results, audit = build_histories(frame)
        shuffled, _ = build_histories(frame.sample(frac=1, random_state=17))

        pd.testing.assert_frame_equal(results, shuffled)
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(results["lifter_name"].tolist(), ["Lifter", "Lifter", "Lifter #1"])
        self.assertEqual(results["previous_total"].tolist(), [500, 520, 300])
        self.assertEqual(results["previous_squat"].tolist(), [200, 210, 100])
        self.assertEqual(results["previous_deadlift"].tolist(), [200, 210, 125])
        self.assertEqual(results["previous_bench"].dropna().tolist(), [100, 75])
        self.assertTrue(pd.isna(results.loc[1, "previous_bench"]))
        self.assertEqual(results["prior_meets"].tolist(), [1, 2, 1])
        self.assertEqual(results["days_since_previous_meet"].tolist(), [10, 30, 20])
        self.assertEqual(results["days_since_first_meet"].tolist(), [10, 40, 20])
        self.assertEqual(results["delta_total"].tolist(), [20, -10, 30])
        self.assertTrue(results["previous_date"].lt(results["target_date"]).all())
        self.assertTrue((results["previous_total"] + results["delta_total"]).eq(results["target_total"]).all())
        self.assertEqual(audit["input_rows"], 6)
        self.assertEqual(audit["excluded_first_meets"], 3)
        self.assertEqual(audit["output_rows"], 3)
        self.assertEqual(audit["output_lifters"], 2)
        self.assertTrue({"target_total", "delta_total", "lifter_name", "target_date"}.isdisjoint(FEATURE_COLUMNS))

    def test_current_and_future_results_cannot_change_predictors(self):
        frame = make_frame()
        expected, _ = build_histories(frame)
        for date in expected["target_date"].unique():
            with self.subTest(date=date):
                changed = frame.copy(deep=True)
                mask = changed["Date"].ge(date)
                changed.loc[mask, INPUT_COLUMNS[2:]] = changed.loc[mask, INPUT_COLUMNS[2:]].fillna(0) + 1000
                actual, _ = build_histories(changed)
                observed = expected["target_date"].le(date)
                columns = ["previous_date", *FEATURE_COLUMNS]
                pd.testing.assert_frame_equal(actual.loc[observed, columns], expected.loc[observed, columns])
                target = expected["target_date"].eq(date)
                self.assertTrue(actual.loc[target, "delta_total"].ne(expected.loc[target, "delta_total"]).all())

    def test_truncated_history_matches_full_history(self):
        frame = make_frame()
        complete, _ = build_histories(frame)
        for date in complete["target_date"].unique():
            with self.subTest(date=date):
                prefix, _ = build_histories(frame.loc[frame["Date"].le(date)])
                expected = complete.loc[complete["target_date"].le(date)].reset_index(drop=True)
                pd.testing.assert_frame_equal(prefix, expected)

    def test_invalid_history_inputs_are_rejected(self):
        frame = make_frame()
        with self.assertRaises(ValueError):
            build_histories(pd.concat([frame, frame.iloc[[0]]]))
        for column, value in [("Name", " "), ("Name", None), ("Date", pd.NaT),
                              ("TotalKg", 0), ("TotalKg", float("inf")), ("TotalKg", None)]:
            with self.subTest(column=column, value=value):
                invalid = frame.copy(deep=True)
                invalid[column] = invalid[column].astype("object")
                invalid.loc[0, column] = value
                with self.assertRaises(ValueError):
                    build_histories(invalid)

    def test_first_meets_and_empty_inputs_produce_no_targets(self):
        frame = make_frame().drop_duplicates("Name")
        for subset in [frame, frame.iloc[:0]]:
            with self.subTest(rows=len(subset)):
                results, audit = build_histories(subset)
                self.assertTrue(results.empty)
                self.assertEqual(audit["excluded_first_meets"], len(subset))


if __name__ == "__main__":
    unittest.main()
