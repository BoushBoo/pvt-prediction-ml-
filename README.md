# PVT prediction with reproducible nested evaluation

Predict bubble-point pressure (Pb) and oil formation volume factor (Bob) from Tf, Rs, gg and api. One shared Python workflow replaces 16 duplicated notebooks. All evaluation tables and plots use the same outer-fold predictions from the training-selected models.

## Install and run

Python 3.10–3.13 is recommended for the TensorFlow extra. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[boosting,neural,notebook,test]'
pytest -q
python -m pvt_ml --data Dataset/PVT_1225.xls --target Pb \
  --models linear ridge extra_trees --output results/PVT_1225_Pb_seed42
```

For all model families use `--phase all` (the default); `--phase 1` and `--phase 2` select presets. An explicit `--models` list takes precedence. Full nested searches, particularly neural networks and CatBoost, can take hours. Start with a small model list. The output directory must be empty, so every run preserves previous results. Fits run serially to control neural random state.

Data files stay local. CSV, XLS and XLSX are supported. Headers are matched case-insensitively after stripping surrounding spaces; the target accepts Pb or Bob. The selected predictors are exactly Tf, Rs, gg and api; the other target and unrelated columns are excluded. Missing/nonfinite/nonnumeric values and ambiguous duplicate column names raise errors. The workflow does not remove observations, deduplicate, impute, clip, or otherwise clean data. Source order is preserved and row positions identify every observation.

## Notebooks

- `notebooks/experiment.ipynb`: configure file, target, models, seeds, fold counts and optional search-space overrides, then run the shared workflow.
- `notebooks/results_analysis.ipynb`: reload saved predictions, reconstruct metrics and regenerate plots without fitting models.

Install the package in the notebook kernel first. Data paths in the examples assume the notebook working directory is `notebooks/`; adjust them or use absolute paths.

## Model families

Phase 1 preset: linear regression, Ridge, Lasso, decision tree, KNN, SVR, plus Random Forest from the ZIP baseline. Phase 2: XGBoost, CatBoost, NN, the original stacking ensemble, Extra Trees and DNN. A dummy mean predictor is available explicitly.

The stack preserves the repository's seven base learners (linear, Ridge, Lasso, depth-10 decision tree, 5-neighbor KNN, SVR, and XGBoost), CatBoost final estimator, and `passthrough=True`. It is not the ZIP's Extra Trees/Ridge stack. Its base configurations are fixed, with five-fold OOF meta-training inside every fit. Scaling-sensitive base learners have independent fold-fitted pipelines.

NN retains the original three candidate ReLU architectures and paired epoch/batch settings; DNN retains 256/128/64 hidden units and the original epoch/batch grid. Both use Adam/MSE and a linear output. Training-only early stopping chooses an epoch count, then a freshly seeded model refits all supplied training data for that count. Neither outer test data nor inner scoring data controls stopping or preprocessing. Target values remain in original units.

## Results

Each run writes:

- `metadata.json`: run status, configuration/search grids, data fingerprint, source file checksum when supplied, Python/platform, all installed package versions and Git revision/dirty status.
- `splits.json` and `row_identifiers.csv`: positional outer/inner split membership and original index labels.
- `*_search.csv`: candidate parameters and inner-fold validation RMSE.
- `*_fit_audit.json`: training memberships, stacking subfolds, neural stopping memberships/scaler means/loss histories and selected epochs.
- `selected_parameters.json`: chosen parameters, resolved estimator settings and inner selection score per model/fold.
- `predictions.csv`: one held-out prediction per row/model, with fold and row position.
- `fold_metrics.csv`, `summary.csv`, `pooled_metrics.csv`: MSE, RMSE, MAE, R² and Pearson correlation reconstructed from those predictions.
- `*_oof.png` and `comparison.png`: prediction/residual plots and fold summaries from that same table.

A failed run is marked `failed`, with completed-fold prediction/parameter checkpoints retained. Do not use partial outputs as complete results. Undefined constant-target R² or constant-vector correlation is represented as missing. Fold summaries omit undefined values.

See [EVALUATION.md](EVALUATION.md) for the ZIP review, protocol and interpretation. The PDFs under Phase 1/2 are historical reports; they were not regenerated and their old metrics must not be treated as corrected results. The former notebooks remain in Git history at commit `ed145f43ca3a94a851fcc24179f63f864a04c5d8`:

```bash
git show 'ed145f43ca3a94a851fcc24179f63f864a04c5d8:Phase 2/Pb/Pb (1225).ipynb' > original.ipynb
```
