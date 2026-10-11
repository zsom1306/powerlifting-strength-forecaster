"""Render four publication-ready summaries of the frozen forecasting analysis."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TEAL, SLATE = "#087f8c", "#8394a5"
MODEL_LABELS = ["Persistence", "Global drift", "Experience-adjusted drift", "LightGBM"]


def save_figure(fig, path: Path, title: str, subtitle: str, note: str) -> None:
    """Apply consistent labeling and save a static figure without opening a GUI."""
    fig.suptitle(title, x=0.07, ha="left", fontsize=18, fontweight="bold", color="#183247")
    fig.text(0.07, 0.90, subtitle, fontsize=10, color="#4d6272")
    fig.text(0.07, 0.025, note, fontsize=9, color="#4d6272", va="bottom")
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.10, 1, 0.87))
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def create_figures(report: dict[str, object], output_dir: Path) -> None:
    """Plot model MAE, training progression, grouped test errors, and permutation reliance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.labelcolor": "#183247", "axes.edgecolor": "#bac5cd"})
    test_label = (f"{report['test_rows']:,} held-out lifter-meet observations | "
                  f"{report['test_first_target_date']} to {report['test_last_target_date']}")
    names = ["persistence", "global_drift", "experience_drift", "lightgbm"]
    values = [report["test_metrics"][name]["mae_kg"] for name in names]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(MODEL_LABELS, values, color=[SLATE, SLATE, SLATE, TEAL], height=0.6)
    ax.bar_label(bars, fmt="%.2f", padding=6)
    ax.invert_yaxis()
    ax.set(xlabel="Mean absolute error (kg) · lower is better", xlim=(0, max(values) * 1.18))
    ax.grid(axis="x", alpha=0.15)
    improvement = report["test_metrics"]["lightgbm"]["mae_reduction_vs_experience_drift_pct"]
    save_figure(fig, output_dir / "test_mae.png", "Next-meet total forecast accuracy", test_label,
                f"LightGBM reduces MAE by {improvement:.1f}% versus experience-adjusted drift. All models score the same meets.")

    progression = report["training_progression"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ticks = [f"{row['group']}\nn={row['rows']:,}" for row in progression]
    bars = ax.bar(ticks, [row["mean_delta_kg"] for row in progression], color=TEAL, width=0.6)
    ax.bar_label(bars, fmt="%.1f", padding=5)
    ax.axhline(0, color=SLATE, linewidth=0.8)
    ax.set(xlabel="Number of prior retained meets", ylabel="Mean next-meet change (kg)")
    ax.margins(y=0.18)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="y", alpha=0.15)
    save_figure(fig, output_dir / "training_progression.png", "Progression by competition experience",
                f"Original training period only: pre-2023 | {sum(row['rows'] for row in progression):,} observations",
                "Band averages describe retained repeat competitors, not an individual athlete's trajectory.\nExperience counts include only prior retained valid full-power meets.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharey=True)
    max_error = max(row[metric] for key in ["errors_by_experience", "errors_by_gap"]
                    for row in report[key] for metric in ["baseline_mae_kg", "lightgbm_mae_kg"])
    for ax, key, title, xlabel in zip(
        axes, ["errors_by_experience", "errors_by_gap"], ["Competition experience", "Time since previous meet"],
        ["Number of prior retained meets", "Gap in days"], strict=True,
    ):
        rows = report[key]
        x = np.arange(len(rows))
        ax.bar(x - 0.18, [row["baseline_mae_kg"] for row in rows], width=0.36, color=SLATE, label="Experience drift")
        ax.bar(x + 0.18, [row["lightgbm_mae_kg"] for row in rows], width=0.36, color=TEAL, label="LightGBM")
        ax.set_xticks(x, [f"{row['group']}\nn={row['rows']:,}" for row in rows], fontsize=8)
        ax.set(title=title, xlabel=xlabel, ylim=(0, max_error * 1.10))
        ax.grid(axis="y", alpha=0.15)
    axes[0].set_ylabel("Mean absolute error (kg)")
    axes[0].legend(frameon=False, fontsize=9)
    save_figure(fig, output_dir / "test_error_groups.png", "Where forecast errors change", test_label,
                "Fixed bands; every test meet appears once in each panel. Counts are meets, not independent athletes.\nComparisons are descriptive associations and do not isolate causal effects.")

    importance = report["permutation_importance"]
    rows = sorted(importance["features"], key=lambda row: row["mean_mae_increase_kg"])
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.barh([row["feature"].replace("_", " ") for row in rows],
            [row["mean_mae_increase_kg"] for row in rows], color=TEAL, height=0.65,
            xerr=[row["std_mae_increase_kg"] for row in rows], error_kw={"capsize": 3, "ecolor": "#183247"})
    ax.axvline(0, color=SLATE, linewidth=0.8)
    ax.set_xlabel("Increase in delta MAE after shuffling (kg)")
    ax.grid(axis="x", alpha=0.15)
    save_figure(fig, output_dir / "permutation_importance.png", "Information used by the progression model",
                f"Fixed sample of {importance['sample_rows']:,} test observations | {importance['repeats']} shuffles per feature",
                "Error bars show repeat standard deviations, not confidence intervals. Importance is not causal or additive.\nCorrelated features and implausible shuffled combinations affect interpretation; the reconstruction offset stays fixed.")


def main() -> None:
    """Render figures directly from the saved analysis report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("reports/analysis.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()
    create_figures(json.loads(args.report.read_text(encoding="utf-8")), args.output_dir)
    print(f"Saved four figures: {args.output_dir.resolve()} (Matplotlib {matplotlib.__version__})")


if __name__ == "__main__":
    main()
