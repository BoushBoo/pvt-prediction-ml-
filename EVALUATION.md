# Evaluation Protocol

## Design

All datasets, prediction targets, and model families use the same evaluation procedure. Outer folds measure held-out predictive performance; inner folds select hyperparameters using only the corresponding outer training data.

This separation prevents evaluation observations from influencing parameter selection. The workflow saves the predictions used for scoring so that tables and figures can be reconstructed without retraining.

## Nested cross-validation

1. Validate the selected input columns without changing values or removing observations.
2. Generate seeded, shuffled outer KFold splits shared across models.
3. Evaluate each parameter-grid candidate on seeded inner folds drawn from the outer training data.
4. Select the candidate with the lowest mean inner-fold RMSE. Ties select the first candidate in the grid.
5. Refit the selected configuration on the outer training fold and predict its held-out observations.
6. Aggregate the saved predictions into fold and pooled evaluation metrics.

Default fold counts are five at each level. An invalid fold size or unsupported neighbor count raises an error rather than changing the requested evaluation automatically.

## Preprocessing

Ridge, Lasso, KNN, and SVR use standardization within estimator pipelines. Each pipeline fits its scaler only on the data supplied to that training fit. The other target and unrelated input columns are excluded from the predictor set.

Data validation does not impute, deduplicate, clip, remove outliers, or otherwise alter observations. Targets remain in their supplied units.

## Stacking ensemble

The seven base learners are linear regression, Ridge, Lasso, a depth-10 decision tree, five-neighbor KNN, SVR, and XGBoost with 100 trees, a learning rate of 0.1, and depth 4.

Five-fold out-of-fold base predictions train a CatBoost final estimator. Original input features are also passed to the final estimator (`passthrough=True`). Scaling-sensitive base learners own separate pipelines, including during the stacking subfolds.

The default stacking configuration is fixed, so its inner search contains one candidate. All base fitting and meta-training occur within the training data supplied to the current inner or outer fit. Subfold memberships are saved in the fit records.

## Neural-network training

The neural-network candidates use hidden layers of 64/32, 128/64, or 128/64/32 units. The deep network uses 256/128/64 units. Both retain ReLU hidden activations, a linear output, Adam, and MSE loss.

Each fit creates a seeded 20% validation subset from its supplied training data. A scaler fitted only on the remaining training subset supports epoch selection. Training stops after ten epochs without an improvement in validation loss, up to the candidate's epoch budget.

The best validation epoch determines the training duration for a freshly initialized model fitted on all supplied training observations. The scaler for this final fit uses those training observations. Inner scoring folds and outer evaluation folds do not affect either stopping or scaling.

Initialization, batch ordering, and validation splitting use controlled seeds. Validation losses, selected epoch counts, stopping memberships, and scaler means are saved.

## Search spaces

| Model | Default candidate settings |
| --- | --- |
| Linear regression | Fixed configuration |
| Ridge / Lasso | 10 logarithmically spaced alpha values from 0.001 to 10 |
| Decision tree | Depths 1–20 |
| Random Forest | 200 trees; depths unrestricted or 10; minimum leaf sizes 1 or 3 |
| KNN | Neighbors 1–20 |
| SVR | 10 logarithmically spaced C values from 0.001 to 100; 5 epsilon values from 0.01 to 1 |
| XGBoost | 100/200 trees; learning rates 0.01/0.05/0.1; depths 4/6/8 |
| CatBoost | 500/1,000 iterations; learning rates 0.01/0.05/0.1; depths 4/6; L2 penalties 3/5/7 |
| Neural network | 64/32 and 128/64: 100 epochs, batch 16; 128/64/32: 150 epochs, batch 32 |
| Deep neural network | 256/128/64 layers; 100/150 epochs; batch sizes 16/32 |
| Extra Trees | Fixed 200-tree configuration |
| Stacking | Fixed configuration described above |
| Mean predictor | Fixed baseline |

Explicit Python `search_spaces` overrides support alternative experiments and short validation runs. The manifest records the actual search spaces used.

## Metrics and reproducibility

MSE, RMSE, MAE, R², and Pearson correlation are calculated from the held-out prediction table. Undefined constant-target R² or constant-vector correlations are represented as missing. Fold summaries omit undefined values.

`summary.csv` contains fold means and sample standard deviations; `pooled_metrics.csv` scores all held-out observations together. These statistics can differ because pooling weights observations and RMSE/R² are nonlinear. Fold standard deviation is descriptive, not a confidence interval.

Each run records split identifiers, candidate scores, selected parameters, predictions, a data fingerprint, dependency versions, Python/platform details, and Git revision information. CLI and notebook runs also record source filenames and file checksums. Fits execute serially and TensorFlow deterministic operations are enabled. Reproduction should use comparable dependency versions and hardware; bitwise agreement across different environments is not guaranteed.

Adjusted R² is not reported: the number of input columns does not describe the effective complexity of the nonlinear model families being compared.

## Interpretation and archived experiments

Shuffled folds estimate performance on new samples from the same dataset mixture. They do not establish transfer to unseen wells. When pooled data includes its constituent datasets, those experiments are not independent replications. A well-held-out study requires verified well identifiers and an explicitly specified grouping protocol.

Each family is evaluated independently. Selecting an overall winner after inspecting outer scores introduces an additional selection step; the winner's score is not an independent evaluation of that choice.

Earlier notebooks and PDF reports, available in Git history, document a previous evaluation procedure. The current workflow separates tuning from held-out scoring, standardizes fold-local preprocessing, and generates all figures from saved evaluation predictions. Results from these procedures should be reported separately. Earlier notebooks can be recovered from Git history before the workflow refactor.
