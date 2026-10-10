"""Compare a fixed, small set of LightGBM configurations on chronological validation."""

import argparse
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from powerlifting_forecaster.splits import TEST_START, VALIDATION_START
from powerlifting_forecaster.train import (
    MODEL_PARAMETERS, load_training_splits, save_training_run, train_regressor,
)


CANDIDATES = {
    "initial": {},
    "leaves_63": {"num_leaves": 63},
    "leaves_127": {"num_leaves": 127},
    "regularized_127": {"num_leaves": 127, "min_child_samples": 300, "reg_lambda": 10.0},
}


def tune_regressor(
    train: pd.DataFrame, validation: pd.DataFrame,
    *, validation_start: str = VALIDATION_START, test_start: str = TEST_START,
) -> tuple[lgb.LGBMRegressor, dict[str, object]]:
    """Select by validation MAE; break exact ties by fewer leaves, then fewer trees."""
    trials = []
    best_model, best_report, best_key, best_name = None, None, None, None
    for name, overrides in CANDIDATES.items():
        print(f"Training candidate: {name}", flush=True)
        model, report = train_regressor(
            train, validation, parameters=overrides,
            validation_start=validation_start, test_start=test_start,
        )
        metrics = report["validation_metrics"]["lightgbm"]
        trials.append({
            "candidate": name, "parameter_overrides": overrides.copy(),
            "best_iteration": report["best_iteration"],
            "evaluated_iterations": report["evaluated_iterations"],
            "validation_metrics": metrics,
        })
        key = (metrics["mae_kg"], report["model_parameters"]["num_leaves"], report["best_iteration"])
        if best_key is None or key < best_key:
            best_model, best_report, best_key, best_name = model, report, key, name
        print(f"{name}: MAE {metrics['mae_kg']:.4f} kg; {report['best_iteration']} trees", flush=True)
    initial_mae = trials[0]["validation_metrics"]["mae_kg"]
    return best_model, {
        **best_report,
        "selection_policy": "Lowest validation MAE across four prespecified configurations; exact ties prefer fewer leaves, then fewer trees. Final test is not loaded.",
        "base_parameters": MODEL_PARAMETERS.copy(),
        "selected_candidate": best_name, "tuning_trials": trials,
        "mae_reduction_vs_initial_model_pct": (
            100 * (1 - best_key[0] / initial_mae) if initial_mae > 0 else None
        ),
    }


def main() -> None:
    """Save the selected model and all candidate scores separately from the initial run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/splits"))
    parser.add_argument("--model", type=Path, default=Path("models/lightgbm_tuned.txt"))
    parser.add_argument("--report", type=Path, default=Path("reports/lightgbm_tuning.json"))
    args = parser.parse_args()
    frames, inputs = load_training_splits(args.data_dir)
    model, report = tune_regressor(frames["train"], frames["validation"])
    save_training_run(model, {"inputs": inputs, **report}, args.model, args.report)


if __name__ == "__main__":
    main()
