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

- KNN encryption detector on `features/1s` with MinMax scaling, `k=6`, and SMOTE.
- LSTM temporal-correlation detector on `features/lstm` with 10 features per timestep.
- Step4-style final evaluator using `final_predict = enc_predict AND tc_predict`.

## Notes

- The fold group is the paired CSV path, so rows from the same time-series file never cross folds.
- LSTM early model selection uses a validation split carved only from each training fold;
  the held-out fold is not used for training decisions.
- Final metrics group by `(Sample, Process)`. Ransomware rows use `code/sample_process.py`
  to filter to the known malicious process, matching `step4_final_result.py`.
- The LSTM intentionally uses `input_size=10` to match the current `step3_temporal_correlation_detection.py`
  feature list.
