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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


def my_prefixed_process_name(value: str) -> str:
    value = Path(str(value)).stem
    for prefix in ("My10_", "My_"):
        if value.startswith(prefix):
            process = value[len(prefix):]
            if not process.lower().endswith(".exe"):
                process = f"{process}.exe"
            return process
    return ""


def resolve_malicious_process(sample: str, sample_process: Dict[str, str], source_path: str = "") -> str:
    sample = str(sample)
    if sample in sample_process:
        return sample_process[sample]

    process = my_prefixed_process_name(sample)
    if process:
        return process

    process = my_prefixed_process_name(source_path)
    if process:
        return process

    return ""


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


def assign_process_labels(
    df: pd.DataFrame,
    record: SampleRecord,
    sample_process: Dict[str, str],
    sample_col: str,
    process_col: str,
) -> pd.Series:
    if record.label_name == "benign":
        return pd.Series(np.zeros(len(df), dtype=np.int64), index=df.index)
    return (
        df.apply(
            lambda row: row[process_col]
            == resolve_malicious_process(row[sample_col], sample_process, record.rel_path),
            axis=1,
        )
        .astype(np.int64)
    )


def load_knn_frame(records: Sequence[SampleRecord], sample_process: Dict[str, str]) -> pd.DataFrame:
    frames = []
    for record in records:
        df = read_feature_csv(record.one_s_path, KNN_FEATURES + ["Sample", "Process", "Second"])
        df = df.copy()
        df["label"] = assign_process_labels(df, record, sample_process, "Sample", "Process")
        df["label_name"] = record.label_name
        df["source_path"] = record.rel_path
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_lstm_frame(records: Sequence[SampleRecord], sample_process: Dict[str, str]) -> pd.DataFrame:
    frames = []
    required = LSTM_FEATURES + ["sample", "process", "starttime"]
    for record in records:
        df = read_feature_csv(record.lstm_path, required)
        df = df.copy()
        df["label"] = assign_process_labels(df, record, sample_process, "sample", "process")
        df["label_name"] = record.label_name
        df["source_path"] = record.rel_path
        df["Second"] = pd.to_numeric(df["starttime"], errors="coerce").fillna(0).astype(np.int64) // 10000000
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def label_counts_dict(df: pd.DataFrame) -> Dict[str, int]:
    return {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()}


def allocate_middle_crop_counts(group_sizes: pd.Series, target_total: int) -> pd.Series:
    total = int(group_sizes.sum())
    if target_total >= total:
        return group_sizes.astype(int)
    if target_total <= 0:
        return pd.Series(0, index=group_sizes.index, dtype=np.int64)

    raw = group_sizes.astype(float) * (float(target_total) / float(total))
    if target_total >= len(group_sizes):
        counts = np.floor(raw).astype(int).clip(lower=1, upper=group_sizes.astype(int))
    else:
        counts = np.floor(raw).astype(int).clip(lower=0, upper=group_sizes.astype(int))

    counts = pd.Series(counts, index=group_sizes.index, dtype=np.int64)
    diff = int(target_total - counts.sum())
    if diff > 0:
        fractions = raw - np.floor(raw)
        candidates = pd.DataFrame(
            {
                "fraction": fractions,
                "size": group_sizes,
                "capacity": group_sizes.astype(int) - counts,
            }
        )
        candidates = candidates[candidates["capacity"] > 0]
        order = candidates.sort_values(["fraction", "size"], ascending=[False, False]).index
        for key in order:
            if diff <= 0:
                break
            add = min(diff, int(candidates.loc[key, "capacity"]))
            counts.loc[key] += add
            diff -= add
    elif diff < 0:
        removable_floor = 1 if target_total >= len(group_sizes) else 0
        fractions = raw - np.floor(raw)
        candidates = pd.DataFrame(
            {
                "fraction": fractions,
                "size": group_sizes,
                "removable": counts - removable_floor,
            }
        )
        candidates = candidates[candidates["removable"] > 0]
        order = candidates.sort_values(["fraction", "size"], ascending=[True, True]).index
        for key in order:
            if diff >= 0:
                break
            remove = min(-diff, int(candidates.loc[key, "removable"]))
            counts.loc[key] -= remove
            diff += remove

    if int(counts.sum()) != target_total:
        raise RuntimeError(
            f"could not allocate middle crop counts: target={target_total}, allocated={int(counts.sum())}"
        )
    return counts.astype(int)


def middle_crop_lstm_negative_rows(
    df: pd.DataFrame,
    target_negative_positive_ratio: Optional[float],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    before = label_counts_dict(df)
    info: Dict[str, object] = {
        "enabled": target_negative_positive_ratio is not None,
        "target_negative_positive_ratio": (
            float(target_negative_positive_ratio) if target_negative_positive_ratio is not None else None
        ),
        "before": before,
        "after": before,
        "target_negative_count": None,
        "kept_negative_count": int((df["label"] == 0).sum()),
        "cropped_negative_count": 0,
        "reason": "disabled",
    }
    if target_negative_positive_ratio is None:
        return df, info

    positive_count = int((df["label"] == 1).sum())
    negative_count = int((df["label"] == 0).sum())
    target_negative_count = int(math.floor(positive_count * target_negative_positive_ratio))
    info["target_negative_count"] = target_negative_count

    if positive_count <= 0:
        info["reason"] = "no_positive_rows"
        return df, info
    if negative_count <= target_negative_count:
        info["reason"] = "already_within_target_ratio"
        return df, info

    positive_df = df[df["label"] == 1]
    negative_df = df[df["label"] == 0]
    group_cols = ["source_path", "sample", "process"]
    group_sizes = negative_df.groupby(group_cols, sort=False).size()
    keep_counts = allocate_middle_crop_counts(group_sizes, target_negative_count)

    kept_negative_indices: List[int] = []
    for key, group in negative_df.groupby(group_cols, sort=False):
        keep_count = int(keep_counts.loc[key])
        if keep_count <= 0:
            continue
        ordered = group.sort_values("Second", kind="mergesort")
        start = (len(ordered) - keep_count) // 2
        kept_negative_indices.extend(ordered.index[start : start + keep_count].tolist())

    kept_indices = sorted(positive_df.index.tolist() + kept_negative_indices)
    cropped = df.loc[kept_indices].reset_index(drop=True)
    after = label_counts_dict(cropped)
    info.update(
        {
            "after": after,
            "kept_negative_count": int(after.get("0", 0)),
            "cropped_negative_count": int(negative_count - after.get("0", 0)),
            "reason": "cropped",
        }
    )
    return cropped, info


def max_valid_smote_k(y: np.ndarray, requested_k: int = 5) -> int:
    _, counts = np.unique(y, return_counts=True)
    minority = int(counts.min())
    return max(1, min(requested_k, minority - 1))


def print_knn_one_class_diagnostics(
    train_df: pd.DataFrame,
    sample_process: Dict[str, str],
    context: str,
) -> None:
    label_counts = {str(k): int(v) for k, v in train_df["label"].value_counts().sort_index().items()}
    print(f"[KNN one-class diagnostic] context={context}")
    print(f"  label_counts={label_counts}")

    ransomware_df = train_df[train_df["label_name"] == "ransomware"]
    if ransomware_df.empty:
        print("  no ransomware feature files are present in this KNN training subset")
        return

    print("  ransomware feature files with no positive rows:")
    for source_path, source_df in ransomware_df.groupby("source_path", sort=True):
        positive_rows = int(source_df["label"].sum())
        if positive_rows > 0:
            continue

        samples = sorted(str(sample) for sample in source_df["Sample"].dropna().unique())
        target_candidates = sorted(
            {
                resolve_malicious_process(sample, sample_process, source_path)
                for sample in samples
            }
        )
        processes = sorted(str(process) for process in source_df["Process"].dropna().unique())
        print(f"    source_path={source_path}")
        print(f"      samples={samples[:5]}")
        print(f"      resolved_malicious_process={target_candidates}")
        print(f"      available_processes_first_30={processes[:30]}")


def train_knn(
    train_df: pd.DataFrame,
    use_smote: bool = True,
    n_neighbors: int = 6,
    sample_process: Optional[Dict[str, str]] = None,
    context: str = "KNN",
) -> Tuple[KNeighborsClassifier, MinMaxScaler]:
    x_train = numeric_matrix(train_df, KNN_FEATURES)
    y_train = train_df["label"].to_numpy(dtype=np.int64)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)

    classes = np.unique(y_train)
    if use_smote and len(classes) <= 1:
        if sample_process is not None:
            print_knn_one_class_diagnostics(train_df, sample_process, context)
        raise ValueError(
            f"{context}: KNN training data has only one class: {classes.tolist()}. "
            "SMOTE requires both benign and ransomware positive rows."
        )

    if use_smote:
        if SMOTE is None:
            raise RuntimeError(
                "SMOTE requires imbalanced-learn. Run `pip install -r requirements.txt` "
                "after the new dependency is added."
            )
        k_neighbors = max_valid_smote_k(y_train)
        x_train, y_train = SMOTE(k_neighbors=k_neighbors, random_state=42).fit_resample(x_train, y_train)

    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights="uniform", metric="minkowski")
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


def class_weights(y: np.ndarray, device: torch.device, mode: str):
    if mode == "none":
        return None

    counts = np.bincount(y, minlength=2).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    if mode == "sqrt_balanced":
        weights = np.sqrt(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_lstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    val_knn_pred: pd.DataFrame,
    sample_process: Dict[str, str],
    threshold_candidates: Sequence[float],
    recall_floor: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    class_weight_mode: str,
    seed: int,
    device: torch.device,
) -> Tuple[LSTMModel, StandardScaler, Dict[str, object]]:
    set_seed(seed)
    scaler = fit_lstm_scaler(train_df)
    x_train, y_train = prepare_lstm_arrays(train_df, scaler)
    x_val, y_val = prepare_lstm_arrays(val_df, scaler)

    model = LSTMModel(input_size=len(LSTM_STEP_FEATURES), hidden_size=50, num_layers=1, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y_train, device, class_weight_mode))
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
    best_key = (False, -1.0, -1.0, -1.0, -float("inf"), -float("inf"))
    best_selection = None
    best_score_stats = None
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

        val_scores = predict_lstm_scores_arrays(model, x_val, device)
        selection = select_lstm_threshold(
            val_df,
            val_scores,
            val_knn_pred,
            sample_process,
            threshold_candidates,
            recall_floor,
        )
        metric = selection["final_step4_process_metrics"]
        score_stats = lstm_score_stats(y_val, val_scores)
        key = (
            bool(selection["meets_recall_floor"]),
            float(metric["f1"]),
            float(metric["recall"]),
            float(metric["precision"]),
            float(score_stats["score_separation"]),
            -float(score_stats["log_loss"]),
        )
        if key > best_key:
            best_key = key
            best_selection = selection
            best_score_stats = score_stats
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler, {
        "best_epoch": best_epoch,
        "class_weight_mode": class_weight_mode,
        "selected_threshold": float(best_selection["threshold"]) if best_selection else 0.5,
        "selected_threshold_meets_recall_floor": bool(best_selection["meets_recall_floor"]) if best_selection else False,
        "selected_threshold_validation_metrics": best_selection["final_step4_process_metrics"] if best_selection else {},
        "selected_epoch_validation_score_stats": best_score_stats if best_score_stats else {},
        "threshold_candidates": [float(t) for t in threshold_candidates],
        "recall_floor": float(recall_floor),
    }


def lstm_score_stats(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    eps = 1e-7
    y_true = y_true.astype(np.int64)
    scores = np.clip(scores.astype(np.float64), eps, 1.0 - eps)
    pos_scores = scores[y_true == 1]
    neg_scores = scores[y_true == 0]
    pos_mean = float(pos_scores.mean()) if len(pos_scores) else 0.0
    neg_mean = float(neg_scores.mean()) if len(neg_scores) else 0.0
    log_loss = -float(np.mean(y_true * np.log(scores) + (1 - y_true) * np.log(1 - scores))) if len(scores) else 0.0
    return {
        "positive_count": int(len(pos_scores)),
        "negative_count": int(len(neg_scores)),
        "positive_score_mean": pos_mean,
        "positive_score_max": float(pos_scores.max()) if len(pos_scores) else 0.0,
        "positive_score_p95": float(np.quantile(pos_scores, 0.95)) if len(pos_scores) else 0.0,
        "negative_score_mean": neg_mean,
        "negative_score_max": float(neg_scores.max()) if len(neg_scores) else 0.0,
        "score_separation": pos_mean - neg_mean,
        "log_loss": log_loss,
    }


def predict_lstm_scores_arrays(model: LSTMModel, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    scores: List[np.ndarray] = []
    loader = DataLoader(LSTMDataset(x, np.zeros(len(x), dtype=np.int64)), batch_size=1024, shuffle=False)
    with torch.no_grad():
        for batch_x, _ in loader:
            logits = model(batch_x.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1]
            scores.append(probs.cpu().numpy())
    return np.concatenate(scores) if scores else np.array([], dtype=np.float32)


def build_lstm_prediction(df: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    out = df[["source_path", "label_name", "label", "sample", "process", "Second"]].copy()
    out = out.rename(columns={"sample": "Sample", "process": "Process"})
    out["tc_score"] = scores
    out["tc_predict"] = (out["tc_score"] >= threshold).astype(int)
    return out


def predict_lstm(
    df: pd.DataFrame,
    model: LSTMModel,
    scaler: StandardScaler,
    device: torch.device,
    threshold: float,
) -> pd.DataFrame:
    x, _ = prepare_lstm_arrays(df, scaler)
    scores = predict_lstm_scores_arrays(model, x, device)
    return build_lstm_prediction(df, scores, threshold)


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

    benign = merged[merged["label_name"] == "benign"].copy()
    ransomware = merged[merged["label_name"] == "ransomware"].copy()
    mapped_processes = ransomware.apply(
        lambda row: resolve_malicious_process(row["Sample"], sample_process, row["source_path"]),
        axis=1,
    )
    mapped = mapped_processes != ""
    mapped_ransomware = ransomware[mapped].copy()
    mapped_ransomware = mapped_ransomware[
        mapped_ransomware.apply(
            lambda row: row["Process"]
            == resolve_malicious_process(row["Sample"], sample_process, row["source_path"]),
            axis=1,
        )
    ]
    final_df = pd.concat([benign, mapped_ransomware], ignore_index=True)

    return final_df, {
        "merged_rows": int(len(merged)),
        "benign_rows": int(len(benign)),
        "ransomware_rows": int(len(ransomware)),
        "unmapped_ransomware_rows": int((~mapped).sum()),
        "mapped_ransomware_rows_after_process_filter": int(len(mapped_ransomware)),
    }


def select_lstm_threshold(
    lstm_df: pd.DataFrame,
    lstm_scores: np.ndarray,
    knn_pred: pd.DataFrame,
    sample_process: Dict[str, str],
    threshold_candidates: Sequence[float],
    recall_floor: float,
) -> Dict[str, object]:
    selections = []
    for threshold in threshold_candidates:
        lstm_pred = build_lstm_prediction(lstm_df, lstm_scores, threshold)
        final_pred, final_counts = final_step4_predictions(knn_pred, lstm_pred, sample_process)
        metrics = process_group_any_metrics(final_pred, "final_predict")
        meets_recall = float(metrics["recall"]) >= recall_floor
        selections.append(
            {
                "threshold": float(threshold),
                "meets_recall_floor": bool(meets_recall),
                "final_step4_process_metrics": metrics,
                "final_step4_counts": final_counts,
            }
        )

    def selection_key(item: Dict[str, object]):
        metrics = item["final_step4_process_metrics"]
        return (
            bool(item["meets_recall_floor"]),
            float(metrics["f1"]),
            float(metrics["recall"]),
            float(metrics["precision"]),
        )

    best = dict(max(selections, key=selection_key))
    best["all_threshold_metrics"] = selections
    return best


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


def parse_float_list(value: str) -> List[float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    try:
        return [float(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid float list: {value}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RansomRadar KNN and LSTM models with sample-level 5-fold CV.")
    parser.add_argument("--features-root", default=str(repo_root() / "features"))
    parser.add_argument("--output-dir", default=str(repo_root() / "training_runs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--class-weight-mode", default="balanced", choices=["none", "sqrt_balanced", "balanced"])
    parser.add_argument(
        "--lstm-thresholds",
        type=parse_float_list,
        default=parse_float_list("0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"),
    )
    parser.add_argument("--threshold-recall-floor", type=float, default=0.95)
    parser.add_argument(
        "--lstm-train-negative-positive-ratio",
        type=float,
        default=None,
        help=(
            "If set, crop only LSTM training negative rows with per-process middle crop so "
            "negative:positive is at most this ratio. Validation/test data and KNN are unchanged."
        ),
    )
    parser.add_argument("--no-smote", action="store_true", help="Disable paper-style SMOTE for KNN.")
    parser.add_argument("--knn-neighbors", type=int, default=6)
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
    if args.lstm_train_negative_positive_ratio is not None and args.lstm_train_negative_positive_ratio <= 0:
        raise ValueError("--lstm-train-negative-positive-ratio must be greater than 0")

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
            "class_weight_mode": args.class_weight_mode,
            "lstm_thresholds": [float(t) for t in args.lstm_thresholds],
            "threshold_recall_floor": args.threshold_recall_floor,
            "lstm_train_negative_positive_ratio": args.lstm_train_negative_positive_ratio,
            "use_smote": not args.no_smote,
            "knn_neighbors": args.knn_neighbors,
            "knn_weights": "uniform",
            "knn_smote_sampling_strategy": "auto",
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

        knn_threshold_train_df = load_knn_frame(lstm_fit_records, sample_process)
        knn_train_df = load_knn_frame(train_records, sample_process)
        knn_val_df = load_knn_frame(lstm_val_records, sample_process)
        knn_test_df = load_knn_frame(test_records, sample_process)
        lstm_train_df = load_lstm_frame(lstm_fit_records, sample_process)
        lstm_val_df = load_lstm_frame(lstm_val_records, sample_process)
        lstm_test_df = load_lstm_frame(test_records, sample_process)
        lstm_train_df, lstm_crop_info = middle_crop_lstm_negative_rows(
            lstm_train_df,
            args.lstm_train_negative_positive_ratio,
        )
        knn_threshold_label_counts = {
            str(k): int(v) for k, v in knn_threshold_train_df["label"].value_counts().sort_index().items()
        }

        knn_threshold_clf, knn_threshold_scaler = train_knn(
            knn_threshold_train_df,
            use_smote=not args.no_smote,
            n_neighbors=args.knn_neighbors,
            sample_process=sample_process,
            context=f"fold {fold_idx} threshold KNN",
        )
        knn_val_pred = predict_knn(knn_val_df, knn_threshold_clf, knn_threshold_scaler)

        knn_clf, knn_scaler = train_knn(
            knn_train_df,
            use_smote=not args.no_smote,
            n_neighbors=args.knn_neighbors,
            sample_process=sample_process,
            context=f"fold {fold_idx} final KNN",
        )
        knn_pred = predict_knn(knn_test_df, knn_clf, knn_scaler)
        joblib.dump(knn_clf, fold_dir / "encryption_detection_clf.joblib")
        joblib.dump(knn_scaler, fold_dir / "encryption_detection_scaler.joblib")
        knn_val_pred.to_csv(fold_dir / "knn_validation_predictions.csv", index=False)
        knn_pred.to_csv(fold_dir / "knn_predictions.csv", index=False)

        lstm_model, lstm_scaler, lstm_train_info = train_lstm(
            lstm_train_df,
            lstm_val_df,
            knn_val_pred,
            sample_process,
            threshold_candidates=args.lstm_thresholds,
            recall_floor=args.threshold_recall_floor,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            class_weight_mode=args.class_weight_mode,
            seed=args.seed + fold_idx,
            device=device,
        )
        selected_threshold = float(lstm_train_info["selected_threshold"])
        lstm_val_pred = predict_lstm(lstm_val_df, lstm_model, lstm_scaler, device, selected_threshold)
        lstm_pred = predict_lstm(lstm_test_df, lstm_model, lstm_scaler, device, selected_threshold)
        torch.save(lstm_model.state_dict(), fold_dir / "tc_detection_clf.pth")
        joblib.dump(lstm_scaler, fold_dir / "tc_detection_scaler.joblib")
        lstm_val_pred.to_csv(fold_dir / "lstm_validation_predictions.csv", index=False)
        lstm_pred.to_csv(fold_dir / "lstm_predictions.csv", index=False)

        final_pred, final_counts = final_step4_predictions(knn_pred, lstm_pred, sample_process)
        final_pred.to_csv(fold_dir / "final_predictions.csv", index=False)

        metrics = {
            "fold": fold_idx,
            "train_sample_count": len(train_records),
            "test_sample_count": len(test_records),
            "knn_train_label_counts": {
                str(k): int(v) for k, v in knn_train_df["label"].value_counts().sort_index().items()
            },
            "lstm_train_label_counts": {
                str(k): int(v) for k, v in lstm_train_df["label"].value_counts().sort_index().items()
            },
            "lstm_train_negative_positive_crop": lstm_crop_info,
            "knn_threshold_train_label_counts": knn_threshold_label_counts,
            "knn_window_metrics": binary_metrics(knn_pred["label"], knn_pred["enc_predict"]),
            "knn_process_metrics": process_group_any_metrics(knn_pred, "enc_predict"),
            "lstm_window_metrics": binary_metrics(lstm_pred["label"], lstm_pred["tc_predict"]),
            "lstm_process_metrics": process_group_any_metrics(lstm_pred, "tc_predict"),
            "final_step4_counts": final_counts,
            "final_step4_process_metrics": process_group_any_metrics(final_pred, "final_predict"),
            "lstm_training": lstm_train_info,
        }
        write_json(fold_dir / "metrics.json", metrics)
        fold_metrics.append(metrics)
        print(f"fold {fold_idx} metrics:")
        print(f"  knn_threshold_train_label_counts={knn_threshold_label_counts}")
        if lstm_crop_info["enabled"]:
            print(
                "  lstm_train_negative_positive_crop="
                f"ratio={lstm_crop_info['target_negative_positive_ratio']} "
                f"before={lstm_crop_info['before']} after={lstm_crop_info['after']} "
                f"reason={lstm_crop_info['reason']}"
            )
        print(f"  selected_lstm_threshold={selected_threshold:.4f}")
        print(f"  {format_metrics('knn_process', metrics['knn_process_metrics'])}")
        print(f"  {format_metrics('lstm_process', metrics['lstm_process_metrics'])}")
        print(f"  {format_metrics('final_step4_process', metrics['final_step4_process_metrics'])}")

    summary = {
        "folds": fold_metrics,
        "aggregate": {
            "knn_process_metrics": aggregate_metric_dicts([m["knn_process_metrics"] for m in fold_metrics]),
            "lstm_process_metrics": aggregate_metric_dicts([m["lstm_process_metrics"] for m in fold_metrics]),
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
