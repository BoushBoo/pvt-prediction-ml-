# Validation performed

Validation uses synthetic input, not the private reservoir data. All production neural and stacking architectures are exercised with short, explicitly overridden training budgets; full production hyperparameter grids have not been run.

- `pytest -q`: 13 tests passed, no skips. Tests cover actual estimator fit memberships, scaler fit isolation, held-out-label perturbation, seeded repeatability, original stacking/NN/DNN architecture preservation, all-family nested-CV smoke execution, CSV/XLS/XLSX, header normalization, retained duplicate observations, rejected missing values, output preservation, and metrics reconstructed from saved predictions.
- CLI versus configurable experiment notebook: matching predictions, fold/pooled metrics, summaries and split JSON for the same synthetic dataset and configuration.
- Both notebook code paths executed, including results analysis and regenerated plots, with local synthetic paths/model/fold configuration substituted for the examples.
- Notebook JSON/code compilation, Python compilation, `git diff --check`, and Ruff E4/E7/E9/F/I checks passed.

Test environment: Python 3.12.14; NumPy 2.3.5; pandas 2.2.3; scikit-learn 1.8.0; matplotlib 3.10.8; XGBoost 3.4.1; CatBoost 1.2.10; TensorFlow CPU 2.20.0; xlrd 2.0.1; openpyxl 3.1.5. Each actual run saves its own complete dependency manifest.

CI repeats the tests on pushes and pull requests. Reservoir evaluation must be run separately with the local input files; existing PDF figures are historical and do not become corrected results through this refactor.
