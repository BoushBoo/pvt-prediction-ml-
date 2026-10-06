# Software Validation

## Automated checks

The recorded local validation run passed **13 tests without skips**. The test suite covers:

- Isolation of outer evaluation observations from model fitting and parameter selection.
- Scaler fitting on the corresponding training rows.
- Unchanged fold-specific selection and predictions when only that fold's held-out labels are perturbed.
- Seeded repeatability, including neural training.
- Stacking and neural-network architecture definitions.
- Nested evaluation of every model family with reduced training budgets.
- CSV, XLS, and XLSX loading, column-name normalization, and observation preservation.
- Rejection of invalid selected inputs and protection of existing output directories.
- Agreement between saved predictions and reported metrics.

The command-line workflow and configurable experiment notebook produced identical predictions, metric tables, and split records for the same synthetic input and configuration. Both notebook code paths were executed, including results analysis and plot generation.

Python and notebook code compilation, Ruff E4/E7/E9/F/I checks, and `git diff --check` also passed. GitHub Actions runs the test suite for pushes and pull requests.

## Recorded environment

| Dependency | Version |
| --- | --- |
| Python | 3.12.14 |
| NumPy | 2.3.5 |
| pandas | 2.2.3 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.10.8 |
| XGBoost | 3.4.1 |
| CatBoost | 1.2.10 |
| TensorFlow CPU | 2.20.0 |
| xlrd | 2.0.1 |
| openpyxl | 3.1.5 |

Each experiment saves its own complete dependency manifest.

## Validation scope

Software validation uses synthetic inputs and explicitly reduced search or training budgets. It exercises the production model architectures but does not establish predictive performance on reservoir datasets or complete the full default hyperparameter searches.

Dataset-level evaluation must be performed using the configured experiment workflow. Archived PDF reports describe earlier experiments and are not outputs of this validation run.
