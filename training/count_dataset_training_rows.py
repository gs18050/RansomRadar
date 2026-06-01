import argparse
import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


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
    lookup_candidates = [sample, Path(sample).stem, Path(str(source_path)).stem]
    for candidate in lookup_candidates:
        if candidate in sample_process:
            return sample_process[candidate]

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


def discover_dataset_records(features_root: Path, dataset: str, lstm_dir_name: str) -> List[SampleRecord]:
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


def label_counts_dict(labels: pd.Series) -> Dict[str, int]:
    return {str(k): int(v) for k, v in labels.value_counts().sort_index().items()}


def summarize_counts(total_rows: int, label_counts: Dict[str, int]) -> Dict[str, object]:
    negative = int(label_counts.get("0", 0))
    positive = int(label_counts.get("1", 0))
    return {
        "total_rows": int(total_rows),
        "negative_rows": negative,
        "positive_rows": positive,
        "positive_ratio": float(positive / total_rows) if total_rows else 0.0,
        "negative_positive_ratio": float(negative / positive) if positive else None,
        "label_counts": label_counts,
    }


def process_stats(processes: Sequence[str]) -> Dict[str, object]:
    normalized = [str(process).strip().lower() for process in processes if str(process).strip()]
    unique = sorted(set(normalized))
    return {
        "unique_count": len(unique),
        "unique_processes": unique,
    }


def summarize_per_file_unique_counts(per_file: Sequence[Dict[str, object]], key: str) -> Dict[str, object]:
    counts = [int(item.get(key, 0)) for item in per_file]
    return {
        "min": int(min(counts)) if counts else 0,
        "max": int(max(counts)) if counts else 0,
        "mean": float(np.mean(counts)) if counts else 0.0,
        "median": float(np.median(counts)) if counts else 0.0,
    }


def count_knn_rows(records: Sequence[SampleRecord], sample_process: Dict[str, str]) -> Dict[str, object]:
    labels = []
    benign_processes: List[str] = []
    per_file = []
    for record in records:
        df = read_feature_csv(record.one_s_path, KNN_FEATURES + ["Sample", "Process", "Second"])
        if df.empty:
            file_counts: Dict[str, int] = {}
            file_benign_processes: List[str] = []
        else:
            file_labels = assign_process_labels(df, record, sample_process, "Sample", "Process")
            labels.append(file_labels)
            file_counts = label_counts_dict(file_labels)
            file_benign_processes = df.loc[file_labels == 0, "Process"].astype(str).tolist()
            benign_processes.extend(file_benign_processes)
        per_file.append(
            {
                "source_path": record.rel_path,
                "label_name": record.label_name,
                "row_count": int(len(df)),
                "label_counts": file_counts,
                "unique_benign_process_count": process_stats(file_benign_processes)["unique_count"],
            }
        )

    all_labels = pd.concat(labels, ignore_index=True) if labels else pd.Series(dtype=np.int64)
    summary = summarize_counts(len(all_labels), label_counts_dict(all_labels))
    summary["benign_processes"] = process_stats(benign_processes)
    summary["per_file_unique_benign_process_count"] = summarize_per_file_unique_counts(
        per_file,
        "unique_benign_process_count",
    )
    summary["per_file"] = per_file
    return summary


def count_lstm_rows(records: Sequence[SampleRecord], sample_process: Dict[str, str]) -> Dict[str, object]:
    labels = []
    benign_processes: List[str] = []
    per_file = []
    required = LSTM_FEATURES + ["sample", "process", "starttime"]
    for record in records:
        df = read_feature_csv(record.lstm_path, required)
        if df.empty:
            file_counts = {}
            file_benign_processes = []
        else:
            file_labels = assign_process_labels(df, record, sample_process, "sample", "process")
            labels.append(file_labels)
            file_counts = label_counts_dict(file_labels)
            file_benign_processes = df.loc[file_labels == 0, "process"].astype(str).tolist()
            benign_processes.extend(file_benign_processes)
        per_file.append(
            {
                "source_path": record.rel_path,
                "label_name": record.label_name,
                "row_count": int(len(df)),
                "label_counts": file_counts,
                "unique_benign_process_count": process_stats(file_benign_processes)["unique_count"],
            }
        )

    all_labels = pd.concat(labels, ignore_index=True) if labels else pd.Series(dtype=np.int64)
    summary = summarize_counts(len(all_labels), label_counts_dict(all_labels))
    summary["benign_processes"] = process_stats(benign_processes)
    summary["per_file_unique_benign_process_count"] = summarize_per_file_unique_counts(
        per_file,
        "unique_benign_process_count",
    )
    summary["per_file"] = per_file
    return summary


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count process-label rows that would be used if a full dataset were used for training."
    )
    parser.add_argument("--features-root", default=str(repo_root() / "features"))
    parser.add_argument("--new-lstm-dir-name", default="lstm_process_filtered")
    parser.add_argument("--legacy-lstm-dir-name", default="lstm")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def print_dataset_summary(dataset_name: str, summary: Dict[str, object]) -> None:
    print(f"{dataset_name}: paired_samples={summary['paired_sample_count']} label_counts={summary['sample_label_counts']}")
    for feature_name in ["knn_1s", "lstm"]:
        item = summary[feature_name]
        ratio = item["negative_positive_ratio"]
        ratio_text = "None" if ratio is None else f"{ratio:.4f}:1"
        print(
            f"  {feature_name}: total={item['total_rows']} "
            f"negative={item['negative_rows']} positive={item['positive_rows']} "
            f"positive_ratio={item['positive_ratio']:.6f} "
            f"negative_positive_ratio={ratio_text} "
            f"unique_benign_processes={item['benign_processes']['unique_count']}"
        )
        per_file_stats = item["per_file_unique_benign_process_count"]
        print(
            f"    per_file_unique_benign_process_count: "
            f"min={per_file_stats['min']} max={per_file_stats['max']} "
            f"mean={per_file_stats['mean']:.2f} median={per_file_stats['median']:.2f}"
        )


def main() -> int:
    args = parse_args()
    root = repo_root()
    features_root = Path(args.features_root).resolve()
    sample_process = load_sample_process_map(root)

    output = {}
    for dataset_name, lstm_dir_name in [
        ("legacy", args.legacy_lstm_dir_name),
        ("new", args.new_lstm_dir_name),
    ]:
        records = discover_dataset_records(features_root, dataset_name, lstm_dir_name)
        sample_labels = [record.label for record in records]
        summary = {
            "dataset": dataset_name,
            "lstm_dir_name": lstm_dir_name,
            "paired_sample_count": len(records),
            "sample_label_counts": {
                str(label): int(sample_labels.count(label)) for label in sorted(set(sample_labels))
            },
            "records": [record_to_json(record) for record in records],
            "knn_1s": count_knn_rows(records, sample_process),
            "lstm": count_lstm_rows(records, sample_process),
        }
        output[dataset_name] = summary
        print_dataset_summary(dataset_name, summary)

    if args.output:
        output_path = Path(args.output).resolve()
        write_json(output_path, output)
        print(f"summary written: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
