import argparse
import importlib.util
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

try:
    from imblearn.over_sampling import SMOTE
except ImportError:  # pragma: no cover - handled at runtime with a clearer message.
    SMOTE = None


KNN_FEATURES = [
    "avg_branchinstructionrate",
    "std_branchinstructionrate",
    "avg_branchmispredictsrate",
    "std_branchmispredictsrate",
    "avg_llcrefrate",
    "std_llcrefrate",
    "avg_llcmissrate",
    "std_llcmissrate",
]

LSTM_STEP_FEATURES = [
    "read",
    "write",
    "rename",
    "delete",
    "filesize",
    "instructions",
    "branchinstructions",
    "branchmispredicts",
    "llcrefs",
    "llcmisses",
]
LSTM_STEPS = 10
LSTM_FEATURES = [f"{name}_{i}" for i in range(LSTM_STEPS) for name in LSTM_STEP_FEATURES]


@dataclass(frozen=True)
class SampleRecord:
    label_name: str
    label: int
    rel_path: str
    one_s_path: Path
    lstm_path: Path


class LSTMDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_size: int = 10, hidden_size: int = 50, num_layers: int = 1, num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sample_process_map(root: Path) -> Dict[str, str]:
    sample_process_path = root / "code" / "sample_process.py"
    spec = importlib.util.spec_from_file_location("ransomradar_sample_process", sample_process_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {sample_process_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sample_process


def read_feature_csv(path: Path, required_columns: Sequence[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed") or str(c) == ""]
    if unnamed:
        df = df.drop(columns=unnamed)
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def numeric_matrix(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    data = df.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return data.to_numpy(dtype=np.float32)


def discover_paired_samples(features_root: Path) -> List[SampleRecord]:
    records: List[SampleRecord] = []
    for label_name, label in [("benign", 0), ("ransomware", 1)]:
        one_s_dir = features_root / "1s" / label_name
        lstm_dir = features_root / "lstm" / label_name
        one_s = {p.name: p for p in one_s_dir.glob("*.csv")}
        lstm = {p.name: p for p in lstm_dir.glob("*.csv")}
        for filename in sorted(set(one_s) & set(lstm)):
            rel_path = f"{label_name}/{filename}"
            records.append(
                SampleRecord(
                    label_name=label_name,
                    label=label,
                    rel_path=rel_path,
                    one_s_path=one_s[filename],
                    lstm_path=lstm[filename],
                )
            )
    return records


def split_records(records: Sequence[SampleRecord], n_splits: int, seed: int):
    labels = np.array([r.label for r in records])
    groups = np.array([r.rel_path for r in records])
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy_x = np.zeros((len(records), 1), dtype=np.float32)
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(dummy_x, labels, groups), start=1):
        yield fold_idx, [records[i] for i in train_idx], [records[i] for i in test_idx]


def validation_split(
    records: Sequence[SampleRecord],
    validation_fraction: float,
    seed: int,
) -> Tuple[List[SampleRecord], List[SampleRecord]]:
    labels = np.array([r.label for r in records])
    indices = np.arange(len(records))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    return [records[i] for i in train_idx], [records[i] for i in val_idx]


def load_knn_frame(records: Sequence[SampleRecord]) -> pd.DataFrame:
    frames = []
    for record in records:
        df = read_feature_csv(record.one_s_path, KNN_FEATURES + ["Sample", "Process", "Second"])
        df = df.copy()
        df["label"] = record.label
        df["label_name"] = record.label_name
        df["source_path"] = record.rel_path
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_lstm_frame(records: Sequence[SampleRecord]) -> pd.DataFrame:
    frames = []
    required = LSTM_FEATURES + ["sample", "process", "starttime"]
    for record in records:
        df = read_feature_csv(record.lstm_path, required)
        df = df.copy()
        df["label"] = record.label
        df["label_name"] = record.label_name
        df["source_path"] = record.rel_path
        df["Second"] = pd.to_numeric(df["starttime"], errors="coerce").fillna(0).astype(np.int64) // 10000000
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def max_valid_smote_k(y: np.ndarray, requested_k: int = 5) -> int:
    _, counts = np.unique(y, return_counts=True)
    minority = int(counts.min())
    return max(1, min(requested_k, minority - 1))


def train_knn(train_df: pd.DataFrame, use_smote: bool = True) -> Tuple[KNeighborsClassifier, MinMaxScaler]:
    x_train = numeric_matrix(train_df, KNN_FEATURES)
    y_train = train_df["label"].to_numpy(dtype=np.int64)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)

    if use_smote:
        if SMOTE is None:
            raise RuntimeError(
                "SMOTE requires imbalanced-learn. Run `pip install -r requirements.txt` "
                "after the new dependency is added."
            )
        k_neighbors = max_valid_smote_k(y_train)
        x_train, y_train = SMOTE(k_neighbors=k_neighbors, random_state=42).fit_resample(x_train, y_train)

    clf = KNeighborsClassifier(n_neighbors=6, weights="uniform", metric="minkowski")
    clf.fit(x_train, y_train)
    return clf, scaler


def predict_knn(df: pd.DataFrame, clf: KNeighborsClassifier, scaler: MinMaxScaler) -> pd.DataFrame:
    out = df[["source_path", "label_name", "label", "Sample", "Process", "Second"]].copy()
    x = scaler.transform(numeric_matrix(df, KNN_FEATURES))
    out["enc_predict"] = clf.predict(x).astype(int)
    return out


def fit_lstm_scaler(train_df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(numeric_matrix(train_df, LSTM_FEATURES))
    return scaler


def prepare_lstm_arrays(df: pd.DataFrame, scaler: StandardScaler) -> Tuple[np.ndarray, np.ndarray]:
    x = scaler.transform(numeric_matrix(df, LSTM_FEATURES))
    x = x.reshape(-1, LSTM_STEPS, len(LSTM_STEP_FEATURES))
    y = df["label"].to_numpy(dtype=np.int64)
    return x.astype(np.float32), y


def class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y, minlength=2).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_lstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> Tuple[LSTMModel, StandardScaler, Dict[str, float]]:
    set_seed(seed)
    scaler = fit_lstm_scaler(train_df)
    x_train, y_train = prepare_lstm_arrays(train_df, scaler)
    x_val, y_val = prepare_lstm_arrays(val_df, scaler)

    model = LSTMModel(input_size=len(LSTM_STEP_FEATURES), hidden_size=50, num_layers=1, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y_train, device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        LSTMDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    best_state = None
    best_f1 = -1.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        val_pred = predict_lstm_arrays(model, x_val, device)
        val_f1 = f1_score(y_val, val_pred, zero_division=0)
        if val_f1 > best_f1:
            best_f1 = float(val_f1)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler, {"best_val_f1": best_f1, "best_epoch": best_epoch}


def predict_lstm_arrays(model: LSTMModel, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    preds: List[np.ndarray] = []
    loader = DataLoader(LSTMDataset(x, np.zeros(len(x), dtype=np.int64)), batch_size=1024, shuffle=False)
    with torch.no_grad():
        for batch_x, _ in loader:
            logits = model(batch_x.to(device))
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=np.int64)


def predict_lstm(df: pd.DataFrame, model: LSTMModel, scaler: StandardScaler, device: torch.device) -> pd.DataFrame:
    out = df[["source_path", "label_name", "label", "sample", "process", "Second"]].copy()
    out = out.rename(columns={"sample": "Sample", "process": "Process"})
    x, _ = prepare_lstm_arrays(df, scaler)
    out["tc_predict"] = predict_lstm_arrays(model, x, device).astype(int)
    return out


def binary_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, object]:
    y_true = np.array(list(y_true), dtype=np.int64)
    y_pred = np.array(list(y_pred), dtype=np.int64)
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "count": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0,
        "confusion_matrix": cm.tolist(),
    }


def group_any_metrics(pred_df: pd.DataFrame, pred_col: str) -> Dict[str, object]:
    grouped = (
        pred_df.groupby(["source_path", "label"], as_index=False)[pred_col]
        .any()
        .rename(columns={pred_col: "pred"})
    )
    return binary_metrics(grouped["label"], grouped["pred"].astype(int))


def process_group_any_metrics(pred_df: pd.DataFrame, pred_col: str) -> Dict[str, object]:
    grouped = (
        pred_df.groupby(["source_path", "Sample", "Process", "label"], as_index=False)[pred_col]
        .any()
        .rename(columns={pred_col: "pred"})
    )
    return binary_metrics(grouped["label"], grouped["pred"].astype(int))


def final_step4_predictions(
    knn_pred: pd.DataFrame,
    lstm_pred: pd.DataFrame,
    sample_process: Dict[str, str],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    merged = pd.merge(
        knn_pred,
        lstm_pred,
        on=["source_path", "label_name", "label", "Sample", "Process", "Second"],
        how="inner",
    )
    merged["final_predict"] = (merged["enc_predict"].astype(bool) & merged["tc_predict"].astype(bool)).astype(int)

    benign = merged[merged["label"] == 0].copy()
    ransomware = merged[merged["label"] == 1].copy()
    mapped = ransomware["Sample"].isin(sample_process)
    mapped_ransomware = ransomware[mapped].copy()
    mapped_ransomware = mapped_ransomware[
        mapped_ransomware.apply(lambda row: row["Process"] == sample_process.get(row["Sample"], ""), axis=1)
    ]
    final_df = pd.concat([benign, mapped_ransomware], ignore_index=True)

    return final_df, {
        "merged_rows": int(len(merged)),
        "benign_rows": int(len(benign)),
        "ransomware_rows": int(len(ransomware)),
        "unmapped_ransomware_rows": int((~mapped).sum()),
        "mapped_ransomware_rows_after_process_filter": int(len(mapped_ransomware)),
    }


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def save_records(path: Path, records: Sequence[SampleRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(r) for r in records]).assign(
        one_s_path=lambda df: df["one_s_path"].astype(str),
        lstm_path=lambda df: df["lstm_path"].astype(str),
    ).to_csv(path, index=False)


def aggregate_metric_dicts(metric_dicts: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    numeric_keys = ["accuracy", "precision", "recall", "f1"]
    return {
        key: {
            "mean": float(np.mean([float(m[key]) for m in metric_dicts])) if metric_dicts else 0.0,
            "std": float(np.std([float(m[key]) for m in metric_dicts])) if metric_dicts else 0.0,
        }
        for key in numeric_keys
    }


def format_metrics(name: str, metrics: Dict[str, object]) -> str:
    return (
        f"{name}: "
        f"acc={float(metrics['accuracy']):.4f} "
        f"precision={float(metrics['precision']):.4f} "
        f"recall={float(metrics['recall']):.4f} "
        f"f1={float(metrics['f1']):.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RansomRadar KNN and LSTM models with sample-level 5-fold CV.")
    parser.add_argument("--features-root", default=str(repo_root() / "features"))
    parser.add_argument("--output-dir", default=str(repo_root() / "training_runs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--no-smote", action="store_true", help="Disable paper-style SMOTE for KNN.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    root = repo_root()
    features_root = Path(args.features_root).resolve()
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir).resolve() / run_name
    device = choose_device(args.device)

    records = discover_paired_samples(features_root)
    if not records:
        raise RuntimeError(f"no paired feature samples found under {features_root}")

    labels = [r.label for r in records]
    label_counts = {str(label): int(labels.count(label)) for label in sorted(set(labels))}
    print(f"paired samples: {len(records)} label_counts={label_counts}")
    print(f"output: {run_dir}")
    print(f"device: {device}")

    sample_process = load_sample_process_map(root)
    write_json(
        run_dir / "config.json",
        {
            "features_root": str(features_root),
            "folds": args.folds,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "validation_fraction": args.validation_fraction,
            "use_smote": not args.no_smote,
            "device": str(device),
            "knn_features": KNN_FEATURES,
            "lstm_step_features": LSTM_STEP_FEATURES,
            "lstm_input_size": len(LSTM_STEP_FEATURES),
            "paired_sample_count": len(records),
            "paired_label_counts": label_counts,
        },
    )
    save_records(run_dir / "paired_samples.csv", records)

    fold_metrics: List[Dict[str, object]] = []

    for fold_idx, train_records, test_records in split_records(records, args.folds, args.seed):
        fold_dir = run_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        lstm_fit_records, lstm_val_records = validation_split(
            train_records,
            validation_fraction=args.validation_fraction,
            seed=args.seed + fold_idx,
        )
        print(
            f"fold {fold_idx}: train_samples={len(train_records)} "
            f"lstm_fit_samples={len(lstm_fit_records)} "
            f"lstm_val_samples={len(lstm_val_records)} "
            f"test_samples={len(test_records)}"
        )
        save_records(fold_dir / "train_samples.csv", train_records)
        save_records(fold_dir / "lstm_fit_samples.csv", lstm_fit_records)
        save_records(fold_dir / "lstm_validation_samples.csv", lstm_val_records)
        save_records(fold_dir / "test_samples.csv", test_records)

        knn_train_df = load_knn_frame(train_records)
        knn_test_df = load_knn_frame(test_records)
        lstm_train_df = load_lstm_frame(lstm_fit_records)
        lstm_val_df = load_lstm_frame(lstm_val_records)
        lstm_test_df = load_lstm_frame(test_records)

        knn_clf, knn_scaler = train_knn(knn_train_df, use_smote=not args.no_smote)
        knn_pred = predict_knn(knn_test_df, knn_clf, knn_scaler)
        joblib.dump(knn_clf, fold_dir / "encryption_detection_clf.joblib")
        joblib.dump(knn_scaler, fold_dir / "encryption_detection_scaler.joblib")
        knn_pred.to_csv(fold_dir / "knn_predictions.csv", index=False)

        lstm_model, lstm_scaler, lstm_train_info = train_lstm(
            lstm_train_df,
            lstm_val_df,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed + fold_idx,
            device=device,
        )
        lstm_pred = predict_lstm(lstm_test_df, lstm_model, lstm_scaler, device)
        torch.save(lstm_model.state_dict(), fold_dir / "tc_detection_clf.pth")
        joblib.dump(lstm_scaler, fold_dir / "tc_detection_scaler.joblib")
        lstm_pred.to_csv(fold_dir / "lstm_predictions.csv", index=False)

        final_pred, final_counts = final_step4_predictions(knn_pred, lstm_pred, sample_process)
        final_pred.to_csv(fold_dir / "final_predictions.csv", index=False)

        metrics = {
            "fold": fold_idx,
            "train_sample_count": len(train_records),
            "test_sample_count": len(test_records),
            "knn_window_metrics": binary_metrics(knn_pred["label"], knn_pred["enc_predict"]),
            "knn_sample_metrics": group_any_metrics(knn_pred, "enc_predict"),
            "lstm_window_metrics": binary_metrics(lstm_pred["label"], lstm_pred["tc_predict"]),
            "lstm_sample_metrics": group_any_metrics(lstm_pred, "tc_predict"),
            "final_step4_counts": final_counts,
            "final_step4_process_metrics": process_group_any_metrics(final_pred, "final_predict"),
            "lstm_training": lstm_train_info,
        }
        write_json(fold_dir / "metrics.json", metrics)
        fold_metrics.append(metrics)
        print(f"fold {fold_idx} metrics:")
        print(f"  {format_metrics('knn_sample', metrics['knn_sample_metrics'])}")
        print(f"  {format_metrics('lstm_sample', metrics['lstm_sample_metrics'])}")
        print(f"  {format_metrics('final_step4_process', metrics['final_step4_process_metrics'])}")

    summary = {
        "folds": fold_metrics,
        "aggregate": {
            "knn_sample_metrics": aggregate_metric_dicts([m["knn_sample_metrics"] for m in fold_metrics]),
            "lstm_sample_metrics": aggregate_metric_dicts([m["lstm_sample_metrics"] for m in fold_metrics]),
            "final_step4_process_metrics": aggregate_metric_dicts(
                [m["final_step4_process_metrics"] for m in fold_metrics]
            ),
        },
    }
    write_json(run_dir / "summary_metrics.json", summary)
    print(f"summary written: {run_dir / 'summary_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
