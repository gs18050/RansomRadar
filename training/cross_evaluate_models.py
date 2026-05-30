import argparse
import importlib.util
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


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

LSTM_STEP_FEATURES_10 = [
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

LSTM_STEP_FEATURES_11 = [
    "read",
    "write",
    "rename",
    "delete",
    "query_information",
    "filesize",
    "instructions",
    "branchinstructions",
    "branchmispredicts",
    "llcrefs",
    "llcmisses",
]

LSTM_STEPS = 10


@dataclass(frozen=True)
class SampleRecord:
    label_name: str
    label: int
    rel_path: str
    one_s_path: Path
    lstm_path: Path


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_sample_process_map(root: Path) -> Dict[str, str]:
    sample_process_path = root / "code" / "sample_process.py"
    spec = importlib.util.spec_from_file_location("ransomradar_sample_process", sample_process_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {sample_process_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sample_process


def my_prefixed_process_name(value: str) -> str:
    stem = Path(str(value)).stem
    for prefix in ("My10_", "My_"):
        if stem.startswith(prefix):
            process = stem[len(prefix):]
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


def lstm_features(input_size: int) -> List[str]:
    if input_size == 10:
        steps = LSTM_STEP_FEATURES_10
    elif input_size == 11:
        steps = LSTM_STEP_FEATURES_11
    else:
        raise ValueError(f"unsupported LSTM input_size={input_size}; expected 10 or 11")
    return [f"{name}_{i}" for i in range(LSTM_STEPS) for name in steps]


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


def discover_dataset_records(
    features_root: Path,
    dataset: str,
    lstm_dir_name: str,
) -> List[SampleRecord]:
    if dataset not in {"new", "legacy"}:
        raise ValueError(f"unknown dataset={dataset!r}")

    records: List[SampleRecord] = []
    for label_name, label in [("benign", 0), ("ransomware", 1)]:
        one_s_base = features_root / "1s" / label_name
        lstm_base = features_root / lstm_dir_name / label_name
        if dataset == "legacy":
            one_s_dir = one_s_base / "legacy"
            lstm_dir = lstm_base / "legacy"
            rel_prefix = f"{label_name}/legacy"
        else:
            one_s_dir = one_s_base
            lstm_dir = lstm_base
            rel_prefix = label_name

        if not one_s_dir.exists():
            raise RuntimeError(f"missing 1s directory for {dataset} dataset: {one_s_dir}")
        if not lstm_dir.exists():
            raise RuntimeError(f"missing LSTM directory for {dataset} dataset: {lstm_dir}")

        one_s = {p.name: p for p in one_s_dir.glob("*.csv")}
        lstm = {p.name: p for p in lstm_dir.glob("*.csv")}
        for filename in sorted(set(one_s) & set(lstm)):
            records.append(
                SampleRecord(
                    label_name=label_name,
                    label=label,
                    rel_path=f"{rel_prefix}/{filename}",
                    one_s_path=one_s[filename],
                    lstm_path=lstm[filename],
                )
            )
    return records


def load_knn_frame(records: Sequence[SampleRecord], sample_process: Dict[str, str]) -> pd.DataFrame:
    frames = []
    for record in records:
        df = read_feature_csv(record.one_s_path, KNN_FEATURES + ["Sample", "Process", "Second"])
        df = df.copy()
        df["label"] = assign_process_labels(df, record, sample_process, "Sample", "Process")
        df["label_name"] = record.label_name
        df["source_path"] = record.rel_path
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_lstm_frame(
    records: Sequence[SampleRecord],
    sample_process: Dict[str, str],
    input_size: int,
) -> pd.DataFrame:
    frames = []
    feature_cols = lstm_features(input_size)
    required = feature_cols + ["sample", "process", "starttime"]
    for record in records:
        df = read_feature_csv(record.lstm_path, required)
        df = df.copy()
        df["label"] = assign_process_labels(df, record, sample_process, "sample", "process")
        df["label_name"] = record.label_name
        df["source_path"] = record.rel_path
        df["Second"] = pd.to_numeric(df["starttime"], errors="coerce").fillna(0).astype(np.int64) // 10000000
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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


def detect_lstm_architecture(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    input_size = int(state_dict["lstm.weight_ih_l0"].shape[1])
    hidden_size = int(state_dict["lstm.weight_hh_l0"].shape[1])
    num_classes = int(state_dict["fc.weight"].shape[0])
    num_layers = len(
        {
            key.split("_l", 1)[1].split(".", 1)[0]
            for key in state_dict
            if key.startswith("lstm.weight_ih_l")
        }
    )
    return {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_classes": num_classes,
    }


def load_lstm_model(path: Path) -> Tuple[LSTMModel, Dict[str, int]]:
    state_dict = torch.load(path, map_location="cpu")
    architecture = detect_lstm_architecture(state_dict)
    model = LSTMModel(**architecture)
    model.load_state_dict(state_dict)
    model.eval()
    return model, architecture


def scaler_input_size(scaler) -> int:
    feature_count = getattr(scaler, "n_features_in_", None)
    if feature_count is None and hasattr(scaler, "mean_"):
        feature_count = len(scaler.mean_)
    if feature_count is None:
        raise RuntimeError("could not infer LSTM scaler input feature count")
    feature_count = int(feature_count)
    if feature_count % LSTM_STEPS != 0:
        raise RuntimeError(f"LSTM scaler feature count must be divisible by {LSTM_STEPS}: {feature_count}")
    input_size = feature_count // LSTM_STEPS
    if input_size not in {10, 11}:
        raise RuntimeError(f"unsupported LSTM scaler input size={input_size}; expected 10 or 11")
    return input_size


def prepare_lstm_arrays(
    df: pd.DataFrame,
    scaler,
    scaler_feature_size: int,
    model_input_size: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    x = scaler.transform(numeric_matrix(df, lstm_features(scaler_feature_size)))
    x = x.reshape(-1, LSTM_STEPS, scaler_feature_size).astype(np.float32)
    adapter = {
        "scaler_input_size": int(scaler_feature_size),
        "model_input_size": int(model_input_size),
        "mode": "none",
    }
    if scaler_feature_size < model_input_size:
        pad_width = model_input_size - scaler_feature_size
        x = np.pad(x, ((0, 0), (0, 0), (0, pad_width)), mode="constant")
        adapter["mode"] = "zero_pad"
        adapter["padded_features_per_step"] = int(pad_width)
    elif scaler_feature_size > model_input_size:
        x = x[:, :, :model_input_size]
        adapter["mode"] = "truncate"
        adapter["truncated_features_per_step"] = int(scaler_feature_size - model_input_size)
    return x.astype(np.float32), adapter


def predict_knn(df: pd.DataFrame, clf, scaler) -> pd.DataFrame:
    out = df[["source_path", "label_name", "label", "Sample", "Process", "Second"]].copy()
    x = scaler.transform(numeric_matrix(df, KNN_FEATURES))
    out["enc_predict"] = clf.predict(x).astype(int)
    return out


def predict_lstm_scores(model: LSTMModel, x: np.ndarray) -> np.ndarray:
    scores: List[np.ndarray] = []
    batch_size = 2048
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.tensor(x[start : start + batch_size], dtype=torch.float32)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[:, 1]
            scores.append(probs.cpu().numpy())
    return np.concatenate(scores) if scores else np.array([], dtype=np.float32)


def build_lstm_prediction(df: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    out = df[["source_path", "label_name", "label", "sample", "process", "Second"]].copy()
    out = out.rename(columns={"sample": "Sample", "process": "Process"})
    out["tc_score"] = scores
    out["tc_predict"] = (out["tc_score"] >= threshold).astype(int)
    return out


def load_model_bundle(model_dir: Path, threshold: float) -> Dict[str, object]:
    knn_clf = joblib.load(model_dir / "encryption_detection_clf.joblib")
    knn_scaler = joblib.load(model_dir / "encryption_detection_scaler.joblib")
    lstm_model, lstm_architecture = load_lstm_model(model_dir / "tc_detection_clf.pth")
    lstm_scaler = joblib.load(model_dir / "tc_detection_scaler.joblib")
    return {
        "model_dir": str(model_dir),
        "knn_clf": knn_clf,
        "knn_scaler": knn_scaler,
        "lstm_model": lstm_model,
        "lstm_scaler": lstm_scaler,
        "lstm_architecture": lstm_architecture,
        "threshold": float(threshold),
    }


def new_fold_threshold(fold_dir: Path) -> float:
    metrics_path = fold_dir / "metrics.json"
    if not metrics_path.exists():
        raise RuntimeError(f"missing fold metrics with selected LSTM threshold: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    return float(metrics["lstm_training"]["selected_threshold"])


def evaluate_bundle(
    bundle: Dict[str, object],
    records: Sequence[SampleRecord],
    sample_process: Dict[str, str],
    output_dir: Path,
    save_predictions: bool,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_input_size = int(bundle["lstm_architecture"]["input_size"])
    scaler_feature_size = scaler_input_size(bundle["lstm_scaler"])
    knn_df = load_knn_frame(records, sample_process)
    lstm_df = load_lstm_frame(records, sample_process, scaler_feature_size)

    knn_pred = predict_knn(knn_df, bundle["knn_clf"], bundle["knn_scaler"])
    x_lstm, lstm_input_adapter = prepare_lstm_arrays(
        lstm_df,
        bundle["lstm_scaler"],
        scaler_feature_size,
        model_input_size,
    )
    lstm_scores = predict_lstm_scores(bundle["lstm_model"], x_lstm)
    lstm_pred = build_lstm_prediction(lstm_df, lstm_scores, float(bundle["threshold"]))
    final_pred, final_counts = final_step4_predictions(knn_pred, lstm_pred, sample_process)

    metrics = {
        "model_dir": bundle["model_dir"],
        "lstm_architecture": bundle["lstm_architecture"],
        "lstm_input_adapter": lstm_input_adapter,
        "lstm_threshold": float(bundle["threshold"]),
        "sample_count": len(records),
        "label_counts": {
            str(label): int(sum(1 for record in records if record.label == label))
            for label in sorted({record.label for record in records})
        },
        "knn_row_count": int(len(knn_df)),
        "lstm_row_count": int(len(lstm_df)),
        "knn_process_metrics": process_group_any_metrics(knn_pred, "enc_predict"),
        "lstm_process_metrics": process_group_any_metrics(lstm_pred, "tc_predict"),
        "final_step4_counts": final_counts,
        "final_step4_process_metrics": process_group_any_metrics(final_pred, "final_predict"),
    }

    write_json(output_dir / "metrics.json", metrics)
    if save_predictions:
        knn_pred.to_csv(output_dir / "knn_predictions.csv", index=False)
        lstm_pred.to_csv(output_dir / "lstm_predictions.csv", index=False)
        final_pred.to_csv(output_dir / "final_predictions.csv", index=False)
    return metrics


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def record_to_json(record: SampleRecord) -> Dict[str, object]:
    data = asdict(record)
    data["one_s_path"] = str(data["one_s_path"])
    data["lstm_path"] = str(data["lstm_path"])
    return data


def aggregate_metric_dicts(metric_dicts: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    keys = ["accuracy", "precision", "recall", "f1"]
    return {
        key: {
            "mean": float(np.mean([float(m[key]) for m in metric_dicts])) if metric_dicts else 0.0,
            "std": float(np.std([float(m[key]) for m in metric_dicts])) if metric_dicts else 0.0,
        }
        for key in keys
    }


def aggregate_fold_results(fold_metrics: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "fold_count": len(fold_metrics),
        "knn_process_metrics": aggregate_metric_dicts([m["knn_process_metrics"] for m in fold_metrics]),
        "lstm_process_metrics": aggregate_metric_dicts([m["lstm_process_metrics"] for m in fold_metrics]),
        "final_step4_process_metrics": aggregate_metric_dicts([m["final_step4_process_metrics"] for m in fold_metrics]),
    }


def format_final_metrics(metrics: Dict[str, object]) -> str:
    final_metrics = metrics["final_step4_process_metrics"]
    return (
        f"acc={float(final_metrics['accuracy']):.4f} "
        f"precision={float(final_metrics['precision']):.4f} "
        f"recall={float(final_metrics['recall']):.4f} "
        f"f1={float(final_metrics['f1']):.4f}"
    )


def format_aggregate(aggregate: Dict[str, object]) -> str:
    final_metrics = aggregate["final_step4_process_metrics"]
    return (
        f"acc={final_metrics['accuracy']['mean']:.4f}+/-{final_metrics['accuracy']['std']:.4f} "
        f"precision={final_metrics['precision']['mean']:.4f}+/-{final_metrics['precision']['std']:.4f} "
        f"recall={final_metrics['recall']['mean']:.4f}+/-{final_metrics['recall']['std']:.4f} "
        f"f1={final_metrics['f1']['mean']:.4f}+/-{final_metrics['f1']['std']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-evaluate legacy and newly trained RansomRadar models.")
    parser.add_argument("--features-root", default=str(repo_root() / "features"))
    parser.add_argument("--legacy-model-dir", default=str(repo_root() / "models"))
    parser.add_argument("--training-runs-dir", default=str(repo_root() / "training_runs"))
    parser.add_argument("--new-run-name", required=True)
    parser.add_argument("--new-lstm-dir-name", default="lstm_process_filtered")
    parser.add_argument("--legacy-lstm-dir-name", default="lstm")
    parser.add_argument("--output-dir", default=str(repo_root() / "cross_eval_runs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    features_root = Path(args.features_root).resolve()
    legacy_model_dir = Path(args.legacy_model_dir).resolve()
    new_run_dir = Path(args.training_runs_dir).resolve() / args.new_run_name
    run_name = args.run_name or f"{args.new_run_name}_cross_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_root = Path(args.output_dir).resolve() / run_name

    if not legacy_model_dir.exists():
        raise RuntimeError(f"legacy model directory does not exist: {legacy_model_dir}")
    if not new_run_dir.exists():
        raise RuntimeError(f"new training run directory does not exist: {new_run_dir}")

    sample_process = load_sample_process_map(root)
    datasets = {
        "legacy_data": discover_dataset_records(features_root, "legacy", args.legacy_lstm_dir_name),
        "new_data": discover_dataset_records(features_root, "new", args.new_lstm_dir_name),
    }
    for dataset_name, records in datasets.items():
        if not records:
            raise RuntimeError(f"{dataset_name} has no paired samples")

    write_json(
        output_root / "config.json",
        {
            "features_root": str(features_root),
            "legacy_model_dir": str(legacy_model_dir),
            "new_run_dir": str(new_run_dir),
            "legacy_lstm_dir_name": args.legacy_lstm_dir_name,
            "new_lstm_dir_name": args.new_lstm_dir_name,
            "legacy_model_lstm_threshold": 0.5,
            "new_model_lstm_threshold_source": "fold metrics.json lstm_training.selected_threshold",
            "dataset_sample_counts": {name: len(records) for name, records in datasets.items()},
        },
    )
    for dataset_name, records in datasets.items():
        write_json(output_root / f"{dataset_name}_samples.json", [record_to_json(record) for record in records])

    legacy_bundle = load_model_bundle(legacy_model_dir, threshold=0.5)
    summary: Dict[str, object] = {"legacy_model": {}, "new_model": {}}

    print(f"output: {output_root}")
    print(f"legacy samples: {len(datasets['legacy_data'])}")
    print(f"new samples: {len(datasets['new_data'])}")

    for dataset_name, records in datasets.items():
        metrics = evaluate_bundle(
            legacy_bundle,
            records,
            sample_process,
            output_root / f"legacy_model_on_{dataset_name}",
            save_predictions=args.save_predictions,
        )
        summary["legacy_model"][dataset_name] = metrics
        print(f"legacy_model on {dataset_name}: {format_final_metrics(metrics)}")

    for dataset_name, records in datasets.items():
        fold_metrics = []
        dataset_output_dir = output_root / f"new_model_on_{dataset_name}"
        for fold_dir in sorted(new_run_dir.glob("fold_*")):
            if not fold_dir.is_dir():
                continue
            threshold = new_fold_threshold(fold_dir)
            bundle = load_model_bundle(fold_dir, threshold=threshold)
            metrics = evaluate_bundle(
                bundle,
                records,
                sample_process,
                dataset_output_dir / fold_dir.name,
                save_predictions=args.save_predictions,
            )
            fold_metrics.append(metrics)
            print(f"new_model {fold_dir.name} on {dataset_name}: {format_final_metrics(metrics)}")
        if not fold_metrics:
            raise RuntimeError(f"no fold directories found under {new_run_dir}")
        aggregate = aggregate_fold_results(fold_metrics)
        summary["new_model"][dataset_name] = {
            "folds": fold_metrics,
            "aggregate": aggregate,
        }
        write_json(dataset_output_dir / "aggregate_metrics.json", aggregate)
        print(f"new_model mean on {dataset_name}: {format_aggregate(aggregate)}")

    write_json(output_root / "summary_metrics.json", summary)
    print(f"summary written: {output_root / 'summary_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
