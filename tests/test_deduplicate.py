"""Checks for unambiguous, reproducible lifter-date resolution."""

import unittest

import pandas as pd

from powerlifting_forecaster.clean_data import RESULT_COLUMNS
from powerlifting_forecaster.deduplicate import HISTORY_KEYS, deduplicate_results


def make_frame(records):
    defaults = {
        "Name": "Lifter", "Date": "2024-02-01", "Sex": "M", "Event": "SBD",
        "Equipment": "Raw", "MeetCountry": "USA", "TotalKg": 500.0,
        "Best3SquatKg": 200.0, "Best3BenchKg": 100.0, "Best3DeadliftKg": 200.0,
    }
    frame = pd.DataFrame([defaults | row for row in records], columns=RESULT_COLUMNS)
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame


class DeduplicateResultsTests(unittest.TestCase):
    def test_whole_rows_order_and_ties(self):
        frame = make_frame([
            {}, {"Age": 25.0}, {"Date": "2023-01-01"}, {"Name": "Lifter #1"},
            {"Name": "Tie", "Federation": "B"}, {"Name": "Tie", "Federation": "A"},
            {"Name": "Incomplete", "Age": 30.0},
            {"Name": "Incomplete", "BodyweightKg": 80.0},
        ])
        original = frame.copy(deep=True)
        results, audit = deduplicate_results(frame)
        shuffled, _ = deduplicate_results(frame.sample(frac=1, random_state=17))

        pd.testing.assert_frame_equal(results, shuffled)
        pd.testing.assert_frame_equal(frame, original)
        self.assertFalse(results.duplicated(HISTORY_KEYS).any())
        self.assertEqual(audit["collapsed_duplicate_rows"], 3)
        self.assertEqual(audit["excluded_conflicting_rows"], 0)
        expected = frame.loc[[6, 2, 1, 3, 5]].reset_index(drop=True)
        pd.testing.assert_frame_equal(results, expected)
        gaps = results.groupby("Name")["Date"].diff().dropna()
        self.assertTrue(gaps.gt(pd.Timedelta(0)).all())

    def test_conflicting_days_are_excluded(self):
        for column, value in {
            "TotalKg": 510.0, "Sex": "F", "Equipment": "Wraps",
            "MeetCountry": "Canada", "Best3SquatKg": 190.0,
            "Best3BenchKg": 110.0, "Best3DeadliftKg": 210.0,
        }.items():
            with self.subTest(column=column):
                frame = make_frame([{}, {column: value}, {"Name": "Other"}])
                results, audit = deduplicate_results(frame)
                self.assertEqual(results["Name"].tolist(), ["Other"])
                self.assertEqual(audit["conflicting_groups"], 1)
                self.assertEqual(audit["excluded_conflicting_rows"], 2)


if __name__ == "__main__":
    unittest.main()
