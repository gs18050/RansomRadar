# RansomRadar Training

This directory contains new training code for sample-level grouped cross-validation.
It does not modify the existing inference code in `code/` or the shipped models in `models/`.

## Command

First create the filtered LSTM feature directory. This keeps benign LSTM files
unchanged, but for ransomware LSTM files keeps only rows from the resolved
ransomware process.

```bash
python code/step1_5_filter_lstm_ransomware_process.py
```

The output is written to `features/lstm_process_filtered`.

```bash
python training/ransomradar_training.py
```

By default this runs 5-fold cross-validation over samples that exist in both
`features/1s` and `features/lstm_process_filtered`, saves outputs under `training_runs/<timestamp>/`,
and trains:

- KNN encryption detector on `features/1s` with MinMax scaling, `k=6`, uniform
  weighting, and paper-style SMOTE balancing.
- LSTM temporal-correlation detector on `features/lstm_process_filtered` with 10 features per timestep.
- Step4-style final evaluator using `final_predict = enc_predict AND tc_predict`.

Use `--lstm-dir-name lstm` to train from the original unfiltered LSTM feature
directory instead.

## Tuned Defaults

The paper does not specify LSTM optimizer/training hyperparameters or the
probability threshold used to turn LSTM scores into class predictions. The script
therefore tunes only those unspecified values:

- `--class-weight-mode balanced`
- `--lstm-thresholds 0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9`
- `--threshold-recall-floor 0.95`
- `--learning-rate 3e-4`
- `--batch-size 64`
- `--epochs 80`

For each fold, the selected LSTM threshold is chosen on the validation subset
from the training fold using final Step4 process-level F1, preferring thresholds
with recall at least `0.95`.
If all thresholds have the same final F1, checkpoint selection falls back to
validation score separation between positive and negative rows, then validation
log loss, so it no longer gets stuck on epoch 1 just because every threshold
initially predicts all negatives.
For threshold selection, the validation KNN predictions come from a temporary
KNN trained only on the LSTM-fit subset, not on the validation subset.

Recommended next run:

```bash
python training/ransomradar_training.py
```

## Cross Evaluation

To evaluate the shipped `models/` and a new 5-fold training run on both the
legacy dataset and the current dataset:

```bash
python training/cross_evaluate_models.py --legacy-run-name <legacy_run> --new-run-name 20260530_173056
```

Expected data layout:

- Current data: `features/1s/<label>/*.csv` and `features/lstm_process_filtered/<label>/*.csv`
- Legacy data: `features/1s/<label>/legacy/*.csv` and `features/lstm/<label>/legacy/*.csv`

If `--legacy-run-name` is provided, both legacy and new models are loaded from
`training_runs/<run>/fold_*` and use each fold's saved `metrics.json` selected
LSTM threshold. If `--legacy-run-name` is omitted, the shipped `models/`
artifact is used for the legacy model with LSTM threshold `0.5`.

Use `--cross-only` to run only `legacy_model on new_data` and
`new_model on legacy_data`.

To count the full-dataset process-label rows that would be used for training:

```bash
python training/count_dataset_training_rows.py --output training_runs/dataset_row_counts.json
```

## Notes

- The fold group is the paired CSV path, so rows from the same time-series file never cross folds.
- Labels are process-specific: benign files are all `0`; ransomware files are `1` only for
  rows where the process matches `code/sample_process.py`, and other processes are `0`.
- Ransomware process resolution checks `code/sample_process.py` first. Only samples not listed
  there use a fallback rule when either the CSV `Sample` value or source filename starts with
  `My10_` or `My_`: remove that prefix and append `.exe` to get the malicious process name.
- LSTM early model selection uses a validation split carved only from each training fold;
  the held-out fold is not used for training decisions.
- Final metrics group by `(Sample, Process)`. Ransomware rows use `code/sample_process.py`
  to filter to the known malicious process, matching `step4_final_result.py`.
- If a KNN training subset has only one class, training stops and prints which ransomware
  feature files had no positive rows, including the resolved malicious process name and
  available process names from that feature file.
- The LSTM intentionally uses `input_size=10` to match the current `step3_temporal_correlation_detection.py`
  feature list.
