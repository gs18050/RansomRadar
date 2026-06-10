import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


BASE_LSTM_STEP_FEATURES = [
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
WRITE_ENTROPY_LSTM_STEP_FEATURES = [
    "write_entropy_byte_weighted_avg",
    "write_entropy_byte_max",
]
LSTM_STEPS = 10


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def feature_columns(step_features: Sequence[str]) -> List[str]:
    return [f"{name}_{i}" for i in range(LSTM_STEPS) for name in step_features]


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_paired_files(features_root: Path, lstm_dir_name: str, label: str) -> List[Path]:
    one_s_dir = features_root / "1s" / label
    lstm_dir = features_root / lstm_dir_name / label
    one_s_names = {p.name for p in one_s_dir.glob("*.csv")}
    return sorted(p for p in lstm_dir.glob("*.csv") if p.name in one_s_names)


def summarize_group(df: pd.DataFrame, columns: Sequence[str]) -> Dict[str, object]:
    present = [col for col in columns if col in df.columns]
    if not present:
        return {
            "present_columns": 0,
            "sum": 0.0,
            "max": 0.0,
            "nonzero_cells": 0,
            "nonzero_rows": 0,
        }

    numeric = df.loc[:, present].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return {
        "present_columns": len(present),
        "sum": float(numeric.to_numpy(dtype=np.float64).sum()),
        "max": float(numeric.to_numpy(dtype=np.float64).max()) if len(numeric) else 0.0,
        "nonzero_cells": int((numeric != 0).to_numpy().sum()),
        "nonzero_rows": int((numeric != 0).any(axis=1).sum()),
    }


def update_totals(totals: Dict[str, object], stats: Dict[str, object]) -> None:
    totals["present_columns"] = max(int(totals["present_columns"]), int(stats["present_columns"]))
    totals["sum"] = float(totals["sum"]) + float(stats["sum"])
    totals["max"] = max(float(totals["max"]), float(stats["max"]))
    totals["nonzero_cells"] = int(totals["nonzero_cells"]) + int(stats["nonzero_cells"])
    totals["nonzero_rows"] = int(totals["nonzero_rows"]) + int(stats["nonzero_rows"])


def empty_group_stats() -> Dict[str, object]:
    return {
        "present_columns": 0,
        "sum": 0.0,
        "max": 0.0,
        "nonzero_cells": 0,
        "nonzero_rows": 0,
    }


def inspect_files(files: Iterable[Path], expected_columns: Sequence[str]) -> Dict[str, object]:
    totals: Dict[str, object] = {
        "files": 0,
        "rows": 0,
        "missing_required_files": 0,
        "read": empty_group_stats(),
        "write": empty_group_stats(),
        "write_entropy_byte_weighted_avg": empty_group_stats(),
        "write_entropy_byte_max": empty_group_stats(),
    }
    missing_examples: List[Dict[str, object]] = []

    read_cols = [f"read_{i}" for i in range(LSTM_STEPS)]
    write_cols = [f"write_{i}" for i in range(LSTM_STEPS)]
    entropy_avg_cols = [f"write_entropy_byte_weighted_avg_{i}" for i in range(LSTM_STEPS)]
    entropy_max_cols = [f"write_entropy_byte_max_{i}" for i in range(LSTM_STEPS)]
    needed = sorted(set(expected_columns) | set(read_cols) | set(write_cols) | set(entropy_avg_cols) | set(entropy_max_cols))

    for path in files:
        header = pd.read_csv(path, nrows=0)
        columns = [col for col in needed if col in header.columns]
        missing = [col for col in expected_columns if col not in header.columns]
        if missing:
            totals["missing_required_files"] = int(totals["missing_required_files"]) + 1
            if len(missing_examples) < 10:
                missing_examples.append({"path": str(path), "missing": missing[:20]})

        df = pd.read_csv(path, usecols=columns) if columns else pd.DataFrame()
        totals["files"] = int(totals["files"]) + 1
        totals["rows"] = int(totals["rows"]) + int(len(df))
        update_totals(totals["read"], summarize_group(df, read_cols))
        update_totals(totals["write"], summarize_group(df, write_cols))
        update_totals(totals["write_entropy_byte_weighted_avg"], summarize_group(df, entropy_avg_cols))
        update_totals(totals["write_entropy_byte_max"], summarize_group(df, entropy_max_cols))

    totals["missing_examples"] = missing_examples
    return totals


def print_group(name: str, stats: Dict[str, object]) -> None:
    print(
        f"  {name}: present_columns={stats['present_columns']} "
        f"sum={float(stats['sum']):.4f} max={float(stats['max']):.4f} "
        f"nonzero_cells={stats['nonzero_cells']} nonzero_rows={stats['nonzero_rows']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect whether LSTM feature CSVs contain the columns and nonzero values used by training."
    )
    parser.add_argument("--features-root", default=str(repo_root() / "features"))
    parser.add_argument("--lstm-dir-name", default=None)
    parser.add_argument("--run-dir", default=None, help="Training run directory. If set, config.json is used.")
    parser.add_argument("--use-write-entropy-features", action="store_true")
    parser.add_argument("--max-files-per-label", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config: Dict[str, object] = {}
    if args.run_dir:
        config = read_json(Path(args.run_dir) / "config.json")

    features_root = Path(config.get("features_root") or args.features_root).resolve()
    lstm_dir_name = args.lstm_dir_name or str(config.get("lstm_dir_name") or "lstm_entropy_process_filtered")
    use_entropy = bool(config.get("use_write_entropy_features", args.use_write_entropy_features))
    step_features = list(BASE_LSTM_STEP_FEATURES)
    if use_entropy:
        step_features.extend(WRITE_ENTROPY_LSTM_STEP_FEATURES)
    expected_columns = feature_columns(step_features)

    print(f"features_root: {features_root}")
    print(f"lstm_dir_name: {lstm_dir_name}")
    print(f"use_write_entropy_features: {use_entropy}")
    print(f"lstm_input_size: {len(step_features)}")
    print(f"expected_feature_columns: {len(expected_columns)}")

    for label in ["benign", "ransomware"]:
        files = discover_paired_files(features_root, lstm_dir_name, label)
        if args.max_files_per_label is not None:
            files = files[: args.max_files_per_label]
        stats = inspect_files(files, expected_columns)
        print(f"\n{label}: files={stats['files']} rows={stats['rows']} missing_required_files={stats['missing_required_files']}")
        print_group("read", stats["read"])
        print_group("write", stats["write"])
        print_group("write_entropy_byte_weighted_avg", stats["write_entropy_byte_weighted_avg"])
        print_group("write_entropy_byte_max", stats["write_entropy_byte_max"])
        if stats["missing_examples"]:
            print("  missing examples:")
            for item in stats["missing_examples"]:
                print(f"    {item['path']}: {item['missing']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
