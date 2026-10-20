# Experiment 5b - burned-only controlled regression comparison

Evaluation-only experiment. Experiment 5 (`exp5_baselines/`, `06_experiments.ipynb` cells 26-27) is not modified. No neural model was retrained.

**Single intended change vs Experiment 5:** the Random Forest regressors are fitted on burned pixels of the original training split and evaluated on burned pixels of the original test split (burned = `y_burned.sum(axis=1) > 0`, the `MaskedMultiTaskLoss` definition). Estimator, hyperparameters (200 trees, max_depth=None, random_state=42, n_jobs=-1, sklearn defaults otherwise), features (mean+std over time, 104), PCA (0.95, random_state=42, fitted on all training pixels -> 14 components), targets, split and seed are those of Experiment 5.

## Burned-only test population (n = 44,303) - Pearson r

| Model | Train regression population | Test regression population | Severity r | Persistence r | Recovery-gap r |
|---|---|---|---|---|---|
| L-TAE | train ∩ burned (206,502), regression loss masked | test ∩ burned (44,303) | 0.8487 | 0.9624 | 0.9828 |
| BiGRU | train ∩ burned (206,502), regression loss masked | test ∩ burned (44,303) | 0.9020 | 0.9752 | 0.9911 |
| TempCNN | train ∩ burned (206,502), regression loss masked | test ∩ burned (44,303) | 0.8901 | 0.9653 | 0.9855 |
| RF raw statistics | train ∩ burned (206,502) | test ∩ burned (44,303) | 0.8201 | 0.9366 | 0.8927 |
| RF PCA | train ∩ burned (206,502) | test ∩ burned (44,303) | 0.7513 | 0.8928 | 0.8477 |

## Original all-pixel values (Table 14 population, n = 91,339)

| Model | Severity r | Persistence r | Recovery-gap r |
|---|---|---|---|
| L-TAE | 0.7740 | 0.5229 | 0.6921 |
| BiGRU | 0.7747 | 0.5150 | 0.6842 |
| TempCNN | 0.7452 | 0.5024 | 0.6634 |
| RF raw statistics | 0.8628 | 0.8992 | 0.8780 |
| RF PCA | 0.8136 | 0.8632 | 0.8385 |

RF all-pixel values are the published Experiment 5 numbers; neural all-pixel values are recomputed from the saved predictions/checkpoints and match the saved results to <= 3e-5.

## Reproduction check

RF raw severity refitted in the Experiment 5 setup (all train -> all test): r = 0.862870 vs published 0.862820 (|diff| = 5.1e-05). The pipeline here reproduces Experiment 5 to within numerical/library variation (Experiment 5 did not record its library versions; PCA explained variance 0.953997 here vs 0.953991 published).

## Caveats

- One random split, no confidence intervals or significance tests; small differences between neural models (e.g. BiGRU vs TempCNN, 0.003-0.012) are not tested.
- Correlations on the burned and all-pixel populations are not directly comparable: the populations and target variances differ.
- RF predictions are stored as float32 (Experiment 5 scored float64; effect on r ~1e-7).

## Files

- `config.json` - full configuration, input hashes, counts, environment, reproduction check
- `indices.npz` - train/test indices, all and burned
- `rf_results.json` - RF metrics (Pearson, Spearman, MAE, RMSE)
- `rf_burned_only_predictions.npz` - RF predictions on test ∩ burned + `test_indices_burned` + targets
- `rf_parts/` - one JSON + prediction array per forest, incl. `rf_raw_severity_all` (reproduction check)
- `neural_results.json` - L-TAE/BiGRU/TempCNN all-test and burned-test correlations, checkpoint hashes

Run: see the docstring of `notebooks/24_exp5b_burned_only_baselines.py` (stages extract, neural, rf_one x 7, rf_collect).
