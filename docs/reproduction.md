# Reproducing the analysis

Run commands from the repository root. Use a fresh checkout for a new dataset snapshot. The commands below rebuild local `data/` files and write new reports under the ignored `artifacts/reproduction/` directory, preserving the published aggregate reports in `reports/`.

## Environment

The published run used Python 3.13.14, LightGBM 4.7.0, pandas 3.0.6, NumPy 2.5.3, scikit-learn 1.9.1, and Matplotlib 3.11.2. LightGBM uses four CPU threads with a fixed seed and deterministic settings. Downloading and initial cleaning stream the archive, but deduplication, feature construction, and fitting load tables into memory. Allow several gigabytes of disk space and memory; runtime depends on hardware.

```shell
git clone https://github.com/zsom1306/powerlifting-strength-forecaster.git
cd powerlifting-strength-forecaster
```

Create and activate an environment using one of these options.

Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux with Python 3.13 installed:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
```

Then install the project and run the tests:

```shell
python -m pip install -e .
python -m unittest discover -s tests -v
```

All following `python` commands refer to this environment. Activation can also be skipped by replacing `python` with `.\.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on macOS/Linux. The published run and command checks were performed on Windows.

## Dataset snapshot

Download the official full dataset:

```shell
python -m powerlifting_forecaster.download
```

This writes `data/raw/openpowerlifting-latest.zip` and a JSON sidecar containing the download time, source URL, byte count, and SHA-256. The downloader verifies ZIP integrity and refuses to replace an existing snapshot. If the intended archive is already present, skip the download.

The published results use the **September 26, 2026** snapshot:

| Property | Value |
| --- | --- |
| Archive size | 169,848,348 bytes |
| CSV member | `openpowerlifting-2026-09-26/openpowerlifting-2026-09-26-f231b4f6.csv` |
| Records / columns | 4,036,909 / 42 |
| Latest recorded meet | September 20, 2026 |

Archive SHA-256:

```text
32a90763bff87e3e88174cae0dab0c5ee0b178b0db0cbee6027accdd5f350f98
```

Verify a local archive in PowerShell:

```powershell
Get-FileHash data/raw/openpowerlifting-latest.zip -Algorithm SHA256
```

The [official download](https://openpowerlifting.gitlab.io/opl-csv/) is a changing latest snapshot. This repository does not distribute the original archive or guarantee its continued availability upstream. Exact reproduction requires a retained copy matching the checksum above. A later download can be processed with the same pipeline, but counts, selected settings, and scores may change; it is a new experiment rather than reproduction of the published numbers. The dated member name alone does not identify the archive bytes.

## Build prediction rows

```shell
python -m powerlifting_forecaster.inspect_data --output artifacts/reproduction/data_inspection.json
python -m powerlifting_forecaster.clean_data --report artifacts/reproduction/initial_cleaning.json
python -m powerlifting_forecaster.deduplicate --report artifacts/reproduction/deduplication.json
python -m powerlifting_forecaster.features --report artifacts/reproduction/features.json
python -m powerlifting_forecaster.splits --report artifacts/reproduction/splits.json
```

The feature command constructs histories internally; a separate invocation of `histories` is unnecessary for this sequence. The split command writes `data/processed/splits/{train,validation,test}.csv` using target dates before 2023, during 2023–2024, and from 2025 onward. Earlier retained results remain available across these boundaries.

## Fit, select, and evaluate

```shell
python -m powerlifting_forecaster.baselines --report artifacts/reproduction/baseline_validation.json
python -m powerlifting_forecaster.tune --model models/lightgbm_reproduced.txt --report artifacts/reproduction/lightgbm_tuning.json
python -m powerlifting_forecaster.walk_forward --report artifacts/reproduction/walk_forward.json --predictions artifacts/reproduction/walk_forward_predictions.csv
python -m powerlifting_forecaster.final_evaluation --selection artifacts/reproduction/lightgbm_tuning.json --model models/lightgbm_final_reproduced.txt --predictions artifacts/reproduction/test_predictions.csv --report artifacts/reproduction/final_test.json
```

`tune` includes the initial configuration among its four candidates, so a separate `train` invocation is unnecessary. The walk-forward command repeats selection within the past of each historical fold and does not read the main validation or final-test files. It is the most computationally expensive evaluation step.

The final evaluator verifies the train/validation file hashes against the selection report, fixes the chosen tree count, refits on all pre-2025 data, and only then loads the test file. It refuses to overwrite any existing final model, predictions, or report. For another run, use new output paths and pass those same paths to the analysis command. Preserve the test boundary: decisions for a new feature or model experiment must not be selected using the published test outcomes.

## Analyze and render

```shell
python -m powerlifting_forecaster.analyze --evaluation artifacts/reproduction/final_test.json --model models/lightgbm_final_reproduced.txt --predictions artifacts/reproduction/test_predictions.csv --report artifacts/reproduction/analysis.json
python -m powerlifting_forecaster.figures --report artifacts/reproduction/analysis.json --output-dir artifacts/reproduction/figures
```

The analysis verifies source hashes and row alignment, checks that predictions reproduce the reported scores, summarizes all test observations, and computes permutation importance on a seeded 20,000-row sample. It performs no model fitting. Figures can also be rendered directly from the committed analysis report without downloading data or fitting a model:

```shell
python -m powerlifting_forecaster.figures --report reports/analysis.json --output-dir artifacts/figure_preview
```

## Outputs and verification

| Location | Contents |
| --- | --- |
| `data/raw/` | Local source archive and download metadata |
| `data/interim/`, `data/processed/` | Cleaned results, features, and chronological splits |
| `models/` | Native LightGBM model files |
| `artifacts/reproduction/` | New aggregate reports, row-level predictions, and figures |
| `reports/` | Published aggregate reports and four figures |

The published prediction table has 1,780,795 rows, partitioned into 1,213,356 training, 296,583 validation, and 270,856 test observations. The final refit uses 1,509,939 observations. Detailed model parameters, Python/package versions, and input/output checksums are recorded in [the final report](../reports/final_test.json); full-precision results should be compared before rounding.

File hashes identify exact artifact bytes. Formatting, line endings, package versions, or platforms can change hashes; cross-platform numerical results need not be bit-identical. The original verification reloaded the saved final model, reproduced all 270,856 predictions, and independently recalculated the metrics. The automated suite includes leakage perturbation tests, chronological boundaries, categorical handling, serialization checks, and diagnostic/permutation checks.
