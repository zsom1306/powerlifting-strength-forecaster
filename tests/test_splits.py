"""Checks for temporal boundaries, complete assignment, and preserved prior history."""

import unittest

import pandas as pd

from powerlifting_forecaster.histories import INPUT_COLUMNS, build_histories
from powerlifting_forecaster.splits import chronological_split


def make_histories():
    dates = ["2022-06-01", "2022-12-31", "2023-01-01", "2024-12-31", "2025-01-01", "2025-02-01"]
    frame = pd.DataFrame([
        ["Lifter", date, 500 + 10 * i, 200 + 5 * i, 100, 200 + 5 * i]
        for i, date in enumerate(dates)
    ], columns=INPUT_COLUMNS)
    return build_histories(frame)[0]


class ChronologicalSplitTests(unittest.TestCase):
    def test_boundaries_coverage_and_prior_history(self):
        frame = make_histories()
        original = frame.copy(deep=True)
        splits = chronological_split(frame)

        self.assertEqual([len(part) for part in splits.values()], [1, 2, 2])
        self.assertEqual(splits["train"]["target_date"].max(), pd.Timestamp("2022-12-31"))
        self.assertEqual(splits["validation"]["target_date"].min(), pd.Timestamp("2023-01-01"))
        self.assertEqual(splits["validation"]["target_date"].max(), pd.Timestamp("2024-12-31"))
        self.assertEqual(splits["test"]["target_date"].min(), pd.Timestamp("2025-01-01"))
        self.assertEqual(splits["validation"]["previous_total"].tolist(), [510, 520])
        self.assertEqual(splits["test"]["previous_total"].tolist(), [530, 540])
        self.assertEqual(splits["test"]["prior_meets"].tolist(), [4, 5])
        pd.testing.assert_frame_equal(pd.concat(splits.values(), ignore_index=True), original)
        pd.testing.assert_frame_equal(frame, original)
        custom = chronological_split(frame, "2024-01-01", "2025-02-01")
        self.assertEqual([len(part) for part in custom.values()], [2, 2, 1])

    def test_assignment_is_independent_of_outcomes_and_input_order(self):
        frame = make_histories()
        expected = chronological_split(frame)
        changed = frame.sample(frac=1, random_state=17).copy()
        changed["target_total"] = -1000
        changed["delta_total"] = pd.NA
        actual = chronological_split(changed)
        keys = ["lifter_name", "target_date"]
        for name in expected:
            with self.subTest(split=name):
                pd.testing.assert_frame_equal(
                    actual[name][keys].sort_values(keys).reset_index(drop=True),
                    expected[name][keys].sort_values(keys).reset_index(drop=True),
                )

    def test_invalid_boundaries_are_rejected(self):
        for validation, test in [
            ("2025-01-01", "2025-01-01"), ("2026-01-01", "2025-01-01"),
            ("invalid", "2025-01-01"), ("2023-01-01", "NaT"),
        ]:
            with self.subTest(validation=validation, test=test):
                with self.assertRaises(ValueError):
                    chronological_split(make_histories(), validation, test)

    def test_invalid_history_keys_and_dates_are_rejected(self):
        frame = make_histories()
        with self.assertRaises(ValueError):
            chronological_split(pd.concat([frame, frame.iloc[[0]]]))
        for column, value in [
            ("lifter_name", " "), ("target_date", pd.NaT), ("previous_date", pd.NaT),
            ("previous_date", pd.Timestamp("2022-12-31")),
            ("previous_date", pd.Timestamp("2023-01-01")),
        ]:
            with self.subTest(column=column, value=value):
                invalid = frame.copy(deep=True)
                invalid.loc[0, column] = value
                with self.assertRaises(ValueError):
                    chronological_split(invalid)

    def test_empty_periods_are_rejected(self):
        splits = chronological_split(make_histories())
        for missing in splits:
            with self.subTest(missing=missing):
                frame = pd.concat([part for name, part in splits.items() if name != missing])
                with self.assertRaises(ValueError):
                    chronological_split(frame)


if __name__ == "__main__":
    unittest.main()
