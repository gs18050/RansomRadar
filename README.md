# RansomRadar

This repository contains the implementation for **RansomRadar**, based on the paper
*Ransomware Detection Through Temporal Correlation between Encryption and I/O Behavior*.

RansomRadar detects ransomware by combining two models:

- a KNN-based encryption detector using hardware performance counter (HPC) features, and
- an LSTM-based temporal-correlation detector using file I/O behavior and HPC features.

The final decision follows the Step 4 pipeline: a process is detected as ransomware only when both the KNN and LSTM detectors flag the same process/time window.

Additional experiment results and supporting material are provided under `docs/`.

## Repository Layout

```text
├── code/                 # Data preprocessing, feature extraction, and Step 2-4 inference code
├── training/             # Cross-validation training, diagnostics, and cross-dataset evaluation
├── dataset/              # Expected raw-data location; full datasets are not included in this repository
├── features/             # Expected feature location after extraction
├── models/               # Shipped KNN and LSTM model artifacts
├── result/               # Inference outputs
├── helper/               # ETL parser and WPA profile files
├── monitor/              # Monitoring-related code/tools
└── docs/                 # Experimental results and additional documentation
```

Expected feature directories:

```text
features/
├── 100ms/                         # 100 ms HPC features
├── 1s/                            # 1 second KNN features
├── lstm/                          # Base LSTM features
├── lstm_process_filtered/         # Base LSTM features filtered to the ransomware process
├── lstm_entropy/                  # LSTM features with optional write-entropy features
└── lstm_entropy_process_filtered/ # Entropy LSTM features filtered to the ransomware process
```

## Dataset Availability

The full raw dataset is not included in this repository. The code expects raw ETL and IRP logs to be placed under `dataset/`, and generated features under `features/`.

For cross-dataset experiments, this repository assumes the following layout:

```text
# Current dataset
features/1s/<label>/*.csv
features/lstm_entropy_process_filtered/<label>/*.csv

# Legacy/provided dataset
features/1s/<label>/legacy/*.csv
features/lstm/<label>/legacy/*.csv
```

where `<label>` is either `benign` or `ransomware`.

## Environment

Recommended environment:

- Windows 10/11
- Python 3.7+
- Windows ADK with WPR available
- Minifilter driver environment for IRP logging
- PyTorch with CUDA support if GPU training is desired
- .NET Framework 4.8 Developer Pack if rebuilding the ETL parser under `helper/PARSER/`

Python dependencies are listed in `requirements.txt`.

Install them with:

```bat
pip install -r requirements.txt
```

## Data Collection

RansomRadar uses two kinds of runtime data.

### Hardware Performance Counters

HPC data is collected with Windows Performance Recorder (`wpr`). Example:

```bat
wpr -start path\to\record.wprp -filemode
```

The resulting ETL files are parsed into CSV files by `code/step0_preprocess.py`.

### IRP Logs

File I/O behavior is collected with a minifilter driver. After the driver is built and installed, start it with:

```bat
sc start irpcollection
```

The IRP logs are used to extract read/write/rename/delete/query-information behavior and write entropy features.

## Timestamp Alignment Note

HPC ETL timestamps and IRP timestamps must overlap. If they do not overlap, IRP-derived features such as `read`, `write`, and write entropy may become zero or invalid.

If you use the older parser artifact that applies a hard-coded timezone offset, run Step 0 with:

```bat
python code\step0_preprocess.py --starttime-offset-seconds -3600
```

If you rebuild the parser with the UTC FILETIME fix, do **not** use this offset.

You can verify timestamp overlap with:

```bat
python code\inspect_irp_hpc_time_overlap.py --label all --output irp_hpc_overlap_all.csv
```

All valid samples should report `ok`.

## Feature Extraction

Set the project path in `code/config.py` before running the pipeline.

### Step 0: Parse ETL Files

```bat
python code\step0_preprocess.py
```

If using the old parser artifact, apply the timestamp correction:

```bat
python code\step0_preprocess.py --starttime-offset-seconds -3600
```

### Step 1: Extract Base Features

```bat
python code\step1_extract_feature.py
```

This creates the base KNN and LSTM feature directories, including:

```text
features/100ms/
features/1s/
features/lstm/
```

### Optional: Extract Write-Entropy Features

```bat
python code\step1_extract_feature.py --use-write-entropy-features
```

This writes LSTM features with two additional write entropy features to:

```text
features/lstm_entropy/
```

The added features are:

```text
write_entropy_byte_weighted_avg
write_entropy_byte_max
```

For large HPC CSV files, reduce the chunk size:

```bat
python code\step1_extract_feature.py --use-write-entropy-features --hpc-chunksize 100000
```

## LSTM Process Filtering

Ransomware samples contain many background processes. For process-level training and evaluation, only the known ransomware process should be labeled as positive.

Create the base filtered LSTM directory:

```bat
python code\step1_5_filter_lstm_ransomware_process.py
```

Create the entropy filtered LSTM directory:

```bat
python code\step1_5_filter_lstm_ransomware_process.py --input-dir-name lstm_entropy --output-dir-name lstm_entropy_process_filtered
```

Ransomware process names are resolved from `code/sample_process.py`. If a sample is not listed there and its name starts with `My_` or `My10_`, the fallback rule removes that prefix and appends `.exe`.

## Features

### KNN Features

The KNN encryption detector uses 1-second HPC statistics:

```text
avg_branchinstructionrate
std_branchinstructionrate
avg_branchmispredictsrate
std_branchmispredictsrate
avg_llcrefrate
std_llcrefrate
avg_llcmissrate
std_llcmissrate
```

### LSTM Base Features

The base LSTM detector uses 10 time steps per sample, with the following features per step:

```text
read
write
rename
delete
filesize
instructions
branchinstructions
branchmispredicts
llcrefs
llcmisses
```

### Optional LSTM Entropy Features

The entropy experiment adds two write-entropy features per step:

```text
write_entropy_byte_weighted_avg
write_entropy_byte_max
```

With entropy enabled, the LSTM input size becomes 12 features per time step.

## Shipped Inference Pipeline

The original inference pipeline uses the model artifacts under `models/`.

### Step 2: Encryption Detection

```bat
python code\step2_encryption_detection.py
```

### Step 3: Temporal Correlation Detection

```bat
python code\step3_temporal_correlation_detection.py
```

### Step 4: Final Decision

```bat
python code\step4_final_result.py
```

Step 4 combines the Step 2 and Step 3 outputs. A final positive detection requires both detectors to flag the same process/time window.

To run Step 2-4 on legacy feature subdirectories, use the legacy options provided by the scripts, for example:

```bat
python code\step2_encryption_detection.py --use-legacy-features
python code\step3_temporal_correlation_detection.py --use-legacy-features
python code\step4_final_result.py --use-legacy-features
```

## Training

The training code is under `training/`. It does not overwrite the shipped artifacts in `models/`.

### Paper-Style Base Training

```bat
python training\ransomradar_training.py
```

By default, this runs grouped 5-fold cross-validation over samples that exist in both:

```text
features/1s/<label>/
features/lstm_entropy_process_filtered/<label>/
```

If `--use-write-entropy-features` is not provided, entropy columns are ignored and only the base 10 LSTM features are used.

### Entropy Feature Training

```bat
python training\ransomradar_training.py --use-write-entropy-features
```

### Change Fold Count

```bat
python training\ransomradar_training.py --folds 3
```

### Training Method

The training script uses sample-level grouped folds. Rows from the same feature file never cross train/test folds.

KNN training:

- uses 1-second HPC features from `features/1s/`,
- applies `MinMaxScaler`,
- uses KNN with `k=6`, and
- applies SMOTE by default unless `--no-smote` is used.

LSTM training:

- uses 10 time steps,
- uses 10 base features per step, or 12 features when write entropy is enabled,
- selects an LSTM threshold on a validation split from the training fold, and
- evaluates with the same Step 4-style process-level final decision.

Labels are process-specific:

- benign rows are labeled `0`,
- ransomware rows are labeled `1` only when the process matches the resolved ransomware process,
- other processes inside ransomware samples are labeled `0`.

### Hyperparameter Tuning Experiments

The default command follows the paper-style architecture and basic training setup. Additional hyperparameter tuning was performed separately for the new dataset and entropy-feature experiments.

Example tuned entropy run:

```bat
python training\ransomradar_training.py --folds 5 --use-write-entropy-features --class-weight-mode none --threshold-recall-floor 0.75 --lstm-thresholds 0.7,0.8,0.85,0.9,0.93,0.95,0.97,0.99 --validation-fraction 0.25 --epochs 120 --learning-rate 0.0001 --batch-size 32
```

## Diagnostics

### Verify LSTM Feature Usage

```bat
python training\inspect_lstm_feature_usage.py --run-dir training_runs\<run_name>
```

This reports whether `read`, `write`, and entropy columns exist and whether they contain nonzero values.

### Count Training Rows and Processes

```bat
python training\count_dataset_training_rows.py --output training_runs\dataset_row_counts.json
```

### Inspect Ransomware Process Labels

```bat
python training\inspect_ransomware_process_labels.py
```

This helps verify whether each ransomware feature file contains the resolved ransomware process.

## Cross-Dataset Evaluation

Cross evaluation compares models trained on different datasets.

### Evaluate Legacy and New Training Runs

```bat
python training\cross_evaluate_models.py --legacy-run-name <legacy_run> --new-run-name <new_run> --save-predictions
```

### Cross Only

```bat
python training\cross_evaluate_models.py --legacy-run-name <legacy_run> --new-run-name <new_run> --cross-only --save-predictions
```

This runs only:

- `legacy_model` on `new_data`, and
- `new_model` on `legacy_data`.

### Evaluate Shipped Models on New Data

```bat
python training\cross_evaluate_models.py --models-on-new-data-only --new-lstm-dir-name lstm_entropy_process_filtered --save-predictions
```

The shipped `models/` LSTM artifact is evaluated in a Step 3-compatible mode so that it follows the behavior of `code/step3_temporal_correlation_detection.py`.

## Notes

- The full raw dataset is not distributed in this repository.
- Large feature extraction may require a smaller `--hpc-chunksize`.
- Timestamp alignment must be verified before using IRP-derived features.
- `docs/` contains additional experiment results and supporting material.
