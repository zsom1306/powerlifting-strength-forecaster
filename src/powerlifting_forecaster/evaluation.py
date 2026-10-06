"""Regression metrics for reconstructed next-meet totals."""

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float | None]:
    """Score aligned totals in kilograms; report undefined R-squared as null."""
    if not actual.index.equals(predicted.index):
        raise ValueError("Actual and predicted totals must have identical row indices.")
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(root_mean_squared_error(actual, predicted))
    r2 = float(r2_score(actual, predicted)) if len(actual) > 1 and actual.nunique() > 1 else None
    return {"mae_kg": mae, "rmse_kg": rmse, "r2_total": r2}
