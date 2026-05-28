# RansomRadar Training

This directory contains new training code for sample-level grouped cross-validation.
It does not modify the existing inference code in `code/` or the shipped models in `models/`.

## Command

```bash
python training/ransomradar_training.py
```

By default this runs 5-fold cross-validation over samples that exist in both
`features/1s` and `features/lstm`, saves outputs under `training_runs/<timestamp>/`,
and trains:

- KNN encryption detector on `features/1s` with MinMax scaling, `k=6`, uniform
  weighting, and paper-style SMOTE balancing.
- LSTM temporal-correlation detector on `features/lstm` with 10 features per timestep.
- Step4-style final evaluator using `final_predict = enc_predict AND tc_predict`.

## Tuned Defaults

The paper does not specify LSTM optimizer/training hyperparameters or the
probability threshold used to turn LSTM scores into class predictions. The script
therefore tunes only those unspecified values:

- `--class-weight-mode sqrt_balanced`
- `--lstm-thresholds 0.5,0.6,0.7,0.8,0.9`
- `--threshold-recall-floor 0.95`
- `--learning-rate 3e-4`
- `--batch-size 64`
- `--epochs 80`

For each fold, the selected LSTM threshold is chosen on the validation subset
from the training fold using final Step4 process-level F1, preferring thresholds
with recall at least `0.95`.
For threshold selection, the validation KNN predictions come from a temporary
KNN trained only on the LSTM-fit subset, not on the validation subset.

Recommended next run:

```bash
python training/ransomradar_training.py
```

## Notes

- The fold group is the paired CSV path, so rows from the same time-series file never cross folds.
- Labels are process-specific: benign files are all `0`; ransomware files are `1` only for
  rows where the process matches `code/sample_process.py`, and other processes are `0`.
- Ransomware samples not listed in `code/sample_process.py` use a fallback rule when either
  the CSV `Sample` value or source filename starts with `My10_` or `My_`: remove that prefix
  and append `.exe` to get the malicious process name.
- LSTM early model selection uses a validation split carved only from each training fold;
  the held-out fold is not used for training decisions.
- Final metrics group by `(Sample, Process)`. Ransomware rows use `code/sample_process.py`
  to filter to the known malicious process, matching `step4_final_result.py`.
- The LSTM intentionally uses `input_size=10` to match the current `step3_temporal_correlation_detection.py`
  feature list.
