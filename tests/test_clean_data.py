"""Checks for full-power eligibility and preservation of incomplete histories."""

import unittest

import pandas as pd

from powerlifting_forecaster.clean_data import RESULT_COLUMNS, clean_results


class CleanResultsTests(unittest.TestCase):
    def test_eligibility_missing_lifts_and_duplicate_preservation(self):
        frame = pd.DataFrame(pd.NA, index=range(7), columns=RESULT_COLUMNS, dtype="string")
        frame["Name"], frame["Date"], frame["Event"] = "NA", "2024-01-01", "SBD"
        frame["TotalKg"] = pd.Series(["500", "100", pd.NA, "0", "-5", "inf", "510"], dtype="string")
        frame.loc[1, "Event"] = "B"
        frame.loc[0, ["Best3SquatKg", "Best3BenchKg"]] = ["-100", "0"]
        original = frame.copy(deep=True)

        results, audit = clean_results(frame)

        self.assertEqual(results["TotalKg"].tolist(), [500.0, 510.0])
        self.assertEqual(results["Name"].tolist(), ["NA", "NA"])
        self.assertTrue(results[["Age", "BodyweightKg", "Best3SquatKg", "Best3BenchKg"]].isna().all().all())
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(results["Date"]))
        self.assertEqual(audit, {
            "input_rows": 7, "excluded_non_sbd": 1, "excluded_invalid_total": 4,
            "nonpositive_best_lifts_to_missing": 2, "output_rows": 2,
        })
        pd.testing.assert_frame_equal(frame, original)

    def test_missing_history_keys_are_rejected(self):
        frame = pd.DataFrame(pd.NA, index=[0], columns=RESULT_COLUMNS, dtype="string")
        frame["Event"], frame["TotalKg"] = "SBD", "500"
        for name, date in [(pd.NA, "2024-01-01"), ("Lifter", pd.NA), ("Lifter", "invalid")]:
            with self.subTest(name=name, date=date):
                frame["Name"], frame["Date"] = name, date
                with self.assertRaises(ValueError):
                    clean_results(frame)


if __name__ == "__main__":
    unittest.main()
