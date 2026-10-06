# Machine Learning for PVT Property Prediction

A reproducible framework for predicting **bubble-point pressure (Pb)** and **oil formation volume factor at bubble point (Bob)** from reservoir-fluid properties. The project compares regression, ensemble, and neural-network models using a shared nested cross-validation protocol.

## Overview

The framework provides:

- A configurable experiment workflow for both prediction targets and multiple datasets.
- Training-fold hyperparameter selection and preprocessing.
- Consistent metrics and figures generated from saved held-out predictions.
- Experiment records containing split membership, parameters, predictions, and dependency versions.

Dataset evaluation is performed locally. The automated tests use synthetic inputs; existing PDF reports document earlier experiments.

## Inputs and targets

| Column | Description | Role |
| --- | --- | --- |
| `Tf` | Temperature | Predictor |
| `Rs` | Solution gas–oil ratio | Predictor |
| `gg` | Gas specific gravity | Predictor |
| `api` | Oil API gravity | Predictor |
| `Pb` | Bubble-point pressure | Target |
| `Bob` | Oil formation volume factor at bubble point | Target |

Inputs may be **CSV, XLS, or XLSX** files. Column names are matched case-insensitively after trimming surrounding spaces. Each experiment uses the four listed predictors and one target; other columns are excluded from modelling.

Values remain in their supplied units. The workflow preserves row order and observations, including repeated rows. Missing, nonfinite, nonnumeric, or ambiguous selected inputs raise validation errors. Data preparation and unit consistency are the responsibility of the dataset provider.

## Installation

Use Python 3.10–3.13 with the neural-network dependencies. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[boosting,neural,notebook,test]'
```

For the core scikit-learn models only:

```bash
pip install -e .
```

## Run an experiment

Place the input file in a local `Dataset/` directory, or supply its absolute path:

```bash
python -m pvt_ml \
  --data Dataset/PVT_1225.xls \
  --target Pb \
  --models linear ridge extra_trees \
  --outer-folds 5 \
  --inner-folds 5 \
  --seed 42 \
  --output results/PVT_1225_Pb_seed42
```

Use `--target Bob` for the formation-volume-factor experiment. `--phase 1`, `--phase 2`, and `--phase all` select model presets; an explicit `--models` list takes precedence. Without either option, all model families are evaluated.

Each run requires an empty output directory. Raw datasets and generated results are excluded from version control by default. Fits execute serially for controlled random-state handling. Full nested searches can be computationally expensive; begin with a small model list to verify the local setup.

## Models

| Preset | Model families |
| --- | --- |
| Phase 1 | Linear regression, Ridge, Lasso, decision tree, Random Forest, KNN, SVR |
| Phase 2 | XGBoost, CatBoost, neural network, stacking ensemble, Extra Trees, deep neural network |
| Optional baseline | Mean predictor (`dummy`) |

The stacking ensemble combines linear regression, Ridge, Lasso, a decision tree, KNN, SVR, and XGBoost. A CatBoost final estimator learns from their out-of-fold predictions and the original features. Scaling-sensitive base learners use independent preprocessing pipelines.

The neural-network search includes hidden-layer configurations of 64/32, 128/64, and 128/64/32 units. The deep neural network uses 256/128/64 units. Both use ReLU hidden layers, a linear output, Adam optimization, and mean squared error loss.

## Evaluation

The default protocol uses five outer folds and five inner folds:

1. Reserve an outer fold for evaluation.
2. Select hyperparameters using inner cross-validation on the remaining data.
3. Refit the selected configuration on the outer training data.
4. Generate predictions for the reserved outer fold.
5. Repeat until every observation has a held-out prediction for each model.

Preprocessing is fitted within the relevant training folds. Neural stopping epochs are selected using a training-only validation subset, followed by a freshly initialized fit on all supplied training data. Metrics and plots use the same saved outer-fold predictions.

The reported metrics are MSE, RMSE, MAE, R², and Pearson correlation. Both fold summaries and pooled out-of-fold metrics are available. See [Evaluation protocol](EVALUATION.md) for search spaces, reproducibility settings, and interpretation.

## Notebooks and project structure

| Path | Purpose |
| --- | --- |
| `notebooks/experiment.ipynb` | Configure and execute experiments |
| `notebooks/results_analysis.ipynb` | Inspect saved predictions, metrics, parameters, and figures |
| `src/pvt_ml/data.py` | Load and validate inputs |
| `src/pvt_ml/models.py` | Define model families and search spaces |
| `src/pvt_ml/neural.py` | Train neural models with internal epoch selection |
| `src/pvt_ml/tuning.py` | Select configurations using inner folds |
| `src/pvt_ml/experiment.py` | Coordinate nested evaluation |
| `src/pvt_ml/evaluation.py` | Calculate metrics from predictions |
| `src/pvt_ml/plotting.py` | Generate prediction, residual, and comparison figures |
| `src/pvt_ml/results.py` | Save experiment records |
| `tests/` | Verify evaluation isolation, reproducibility, and input handling |

Install the package in the notebook's Python environment. Example notebook paths assume the working directory is `notebooks/`; adjust them for the local setup.

## Saved outputs

| Output | Contents |
| --- | --- |
| `metadata.json` | Configuration, run status, data fingerprint, environment versions, and Git revision |
| `splits.json` | Outer and inner split memberships |
| `row_identifiers.csv` | Row positions and source index labels |
| `*_search.csv` | Candidate parameters and inner-fold scores |
| `*_fit_audit.json` | Fit memberships, stacking splits, and neural stopping records |
| `selected_parameters.json` | Selected and resolved estimator parameters |
| `predictions.csv` | Held-out observations and predictions by model and fold |
| `fold_metrics.csv` | Metrics for each outer fold |
| `summary.csv` | Fold means and sample standard deviations |
| `pooled_metrics.csv` | Metrics across pooled held-out predictions |
| `*_oof.png`, `comparison.png` | Prediction, residual, and model-comparison figures |

Incomplete runs are marked in the manifest and retain completed-fold checkpoints. Undefined R² or correlation values are stored as missing.

## Validation and scope

```bash
pytest -q
```

Automated tests cover fold isolation, preprocessing, deterministic execution, input formats, model architectures, and consistency between predictions and metrics. See [Validation notes](VALIDATION.md).

Shuffled cross-validation evaluates new samples from the same dataset mixture. Generalization to unseen wells requires a separate well-held-out study. Model-family rankings and fold standard deviations should be interpreted within this evaluation scope.

Earlier experiment notebooks remain available in Git history. PDFs under `Phase 1/` and `Phase 2/` are archived reports and are separate from results produced by the current workflow.
