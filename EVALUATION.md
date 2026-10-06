# Evaluation protocol and migration review

## Review before implementation

The ZIP's 16 original notebooks were checked byte-for-byte against the repository at `ed145f43ca3a94a851fcc24179f63f864a04c5d8`; all matched. The ZIP adds a useful initial nested-CV workflow, but keeps the duplicated notebooks, omits NN/DNN, substitutes an Extra Trees + Ridge stack with a Ridge meta-model, and combines model definitions, tuning, plotting and saving in one module. It saves outer splits but not inner or neural stopping splits. Those gaps are addressed here rather than copying the ZIP wholesale.

The old notebooks tune repeatedly against a 30% test split, then cross-validate configurations already chosen using that dataset. Some plots use the last loop predictions or refitted default configurations rather than the selected model. Those numbers are not unbiased held-out estimates. Historical PDFs remain for context, not as current results.

## Nested protocol

1. Validate headers and selected numeric columns without modifying values or deleting rows.
2. Generate seeded shuffled outer KFold splits once, shared across models.
3. For each model and outer fold, run a parameter grid on seeded shuffled inner splits of only the outer training data. Minimize the mean inner-fold RMSE; ties choose the first candidate.
4. Refit the selected configuration on the entire outer training fold. For neural models, select stopping epochs using a separate seeded 20% validation subset inside each fit; preprocessing at this stage sees only that subset's training portion. Reinitialize and refit all supplied training rows for the selected number of epochs.
5. Predict the outer test fold once and save it. Compute every evaluation metric and plot from these predictions. No post-selection refit supplies evaluation plots.

For each stacking fit, five-fold OOF predictions of the original seven base learners train CatBoost. Each scaling-sensitive learner owns a pipeline, so scalers are refitted during the stack's subfolds. Meta-training includes original raw features (`passthrough=True`). The stack's configurations remain fixed as in the repository, so its inner search has one candidate. Both inner fits and outer refits construct the stack entirely within their supplied training data.

## Search spaces and changes

Original Ridge/Lasso alpha ranges, decision-tree depths, KNN neighbor counts, SVR C/epsilon ranges, XGBoost grid, CatBoost grid (including l2 regularization), NN candidates and DNN epoch/batch grid are migrated from the notebooks. Extra Trees retains the original fixed 200 trees. Random Forest and the dummy predictor are additional ZIP baseline families, explicitly identified as such; neither replaces an original learner. Random Forest retains the ZIP's 200 trees with None/10-depth and 1/3-minimum-leaf grid.

Scaled Ridge/Lasso/KNN and the stacking base pipelines are methodological improvements; their new results need not match the historical raw-feature fits. Manual deterministic neural batching implements training-only patience-10 epoch selection, followed by an all-training refit instead of retaining only the stopping-training subset. The architecture, optimizer, loss and units are preserved. Adjusted R² is omitted because counting input columns does not estimate effective model complexity for these nonlinear learners; MSE/RMSE/MAE/R² and correlation remain available.

Explicit Python `search_spaces` overrides support smaller smoke runs or intentionally expanded studies. They are saved in the manifest. Invalid neighbor/fold sizes raise rather than silently shrinking the search or deleting observations. Optional dependencies are resolved before outputs are created. Training runs serially; fixed seeds control KFold, tree/boosting models, neural initialization and batch order. TensorFlow deterministic operations are enabled. Reproduction should use the saved dependency versions and comparable CPU/hardware; deterministic settings do not promise bitwise equality across library/hardware versions.

## Interpretation

Outer folds estimate performance on new samples from the same dataset mixture. Shuffled folds do not estimate performance on unseen wells. The combined dataset overlaps its constituent well datasets; results across those files are not independent replications. If generalization across wells is the goal, a separately specified group-held-out protocol is needed.

Fold mean/sample standard deviation and pooled OOF metrics are both reported and distinctly named. They differ because pooling weights observations and RMSE/R² are nonlinear. Fold SD is not a confidence interval. Every family is evaluated independently; choosing the best family after comparing outer scores makes the winner's reported score optimistic for that additional choice. There is no claimed unbiased evaluation of an overall model-family winner.

The workflow is verified with synthetic data, including all original model architectures and both neural networks. This PR does not claim replacement reservoir results: the ZIP/repository contains no raw input datasets. Run the configured notebook or CLI on the local data to generate corrected results; do not reuse the old PDFs.
