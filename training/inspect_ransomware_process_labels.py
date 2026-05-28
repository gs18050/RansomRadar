import argparse
import csv
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple


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
    if sample in sample_process:
        return sample_process[sample]

    process = my_prefixed_process_name(sample)
    if process:
        return process

    process = my_prefixed_process_name(source_path)
    if process:
        return process

    return ""


def read_csv_rows(path: Path) -> Tuple[List[str], List[dict]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def inspect_file(path: Path, sample_col: str, process_col: str, sample_process: Dict[str, str], source_path: str):
    fieldnames, rows = read_csv_rows(path)
    if sample_col not in fieldnames or process_col not in fieldnames:
        return {
            "source_path": source_path,
            "exists": True,
            "valid_schema": False,
            "row_count": len(rows),
            "error": f"missing columns: sample_col={sample_col in fieldnames}, process_col={process_col in fieldnames}",
        }

    samples = sorted({str(row.get(sample_col, "")).strip() for row in rows if str(row.get(sample_col, "")).strip()})
    processes = sorted({str(row.get(process_col, "")).strip() for row in rows if str(row.get(process_col, "")).strip()})
    target_candidates = sorted({resolve_malicious_process(sample, sample_process, source_path) for sample in samples})
    target_candidates = [target for target in target_candidates if target]

    positive_rows = 0
    positive_processes = set()
    for row in rows:
        sample = str(row.get(sample_col, "")).strip()
        process = str(row.get(process_col, "")).strip()
        target = resolve_malicious_process(sample, sample_process, source_path)
        if target and process == target:
            positive_rows += 1
            positive_processes.add(process)

    return {
        "source_path": source_path,
        "exists": True,
        "valid_schema": True,
        "row_count": len(rows),
        "sample_count": len(samples),
        "samples": samples,
        "resolved_targets": target_candidates,
        "target_exists_in_processes": any(target in processes for target in target_candidates),
        "positive_rows": positive_rows,
        "positive_process_count": len(positive_processes),
        "process_count": len(processes),
        "processes": processes,
    }


def paired_ransomware_files(features_root: Path):
    one_s_dir = features_root / "1s" / "ransomware"
    lstm_dir = features_root / "lstm" / "ransomware"
    one_s = {p.name: p for p in one_s_dir.glob("*.csv")}
    lstm = {p.name: p for p in lstm_dir.glob("*.csv")}
    for filename in sorted(set(one_s) | set(lstm)):
        yield filename, one_s.get(filename), lstm.get(filename)


def format_list(values: List[str], limit: int) -> str:
    if not values:
        return "[]"
    shown = values[:limit]
    suffix = "" if len(values) <= limit else f", ... +{len(values) - limit} more"
    return "[" + ", ".join(shown) + suffix + "]"


def print_file_report(filename: str, one_s_info: dict, lstm_info: dict, process_limit: int) -> None:
    print(f"\n=== {filename} ===")
    for label, info in [("1s", one_s_info), ("lstm", lstm_info)]:
        if not info.get("exists"):
            print(f"[{label}] missing")
            continue
        if not info.get("valid_schema"):
            print(f"[{label}] invalid schema: {info.get('error')}")
            continue

        print(
            f"[{label}] rows={info['row_count']} "
            f"positive_rows={info['positive_rows']} "
            f"positive_processes={info['positive_process_count']} "
            f"target_exists={info['target_exists_in_processes']}"
        )
        print(f"  samples={format_list(info['samples'], 3)}")
        print(f"  resolved_targets={format_list(info['resolved_targets'], 5)}")
        if info["positive_rows"] == 0:
            print(f"  processes={format_list(info['processes'], process_limit)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect ransomware feature files and report whether the resolved malicious process has positive rows."
    )
    parser.add_argument("--features-root", default=str(repo_root() / "features"))
    parser.add_argument("--process-limit", type=int, default=30)
    parser.add_argument(
        "--show",
        choices=["all", "problem"],
        default="problem",
        help="Print all ransomware files or only files with missing/zero-positive labels.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    features_root = Path(args.features_root)
    sample_process = load_sample_process_map(root)

    totals = {
        "files": 0,
        "missing_1s": 0,
        "missing_lstm": 0,
        "zero_positive_1s": 0,
        "zero_positive_lstm": 0,
        "positive_rows_1s": 0,
        "positive_rows_lstm": 0,
        "rows_1s": 0,
        "rows_lstm": 0,
    }

    reports = []
    for filename, one_s_path, lstm_path in paired_ransomware_files(features_root):
        totals["files"] += 1
        source_path = f"ransomware/{filename}"
        if one_s_path is None:
            one_s_info = {"source_path": source_path, "exists": False}
            totals["missing_1s"] += 1
        else:
            one_s_info = inspect_file(one_s_path, "Sample", "Process", sample_process, source_path)
            if one_s_info.get("valid_schema"):
                totals["rows_1s"] += one_s_info["row_count"]
                totals["positive_rows_1s"] += one_s_info["positive_rows"]
                if one_s_info["positive_rows"] == 0:
                    totals["zero_positive_1s"] += 1

        if lstm_path is None:
            lstm_info = {"source_path": source_path, "exists": False}
            totals["missing_lstm"] += 1
        else:
            lstm_info = inspect_file(lstm_path, "sample", "process", sample_process, source_path)
            if lstm_info.get("valid_schema"):
                totals["rows_lstm"] += lstm_info["row_count"]
                totals["positive_rows_lstm"] += lstm_info["positive_rows"]
                if lstm_info["positive_rows"] == 0:
                    totals["zero_positive_lstm"] += 1

        is_problem = (
            not one_s_info.get("exists")
            or not lstm_info.get("exists")
            or one_s_info.get("positive_rows", 0) == 0
            or lstm_info.get("positive_rows", 0) == 0
        )
        if args.show == "all" or is_problem:
            reports.append((filename, one_s_info, lstm_info))

    print("=== Summary ===")
    for key, value in totals.items():
        print(f"{key}: {value}")
    if totals["rows_1s"]:
        print(f"positive_row_ratio_1s: {totals['positive_rows_1s'] / totals['rows_1s']:.6f}")
    if totals["rows_lstm"]:
        print(f"positive_row_ratio_lstm: {totals['positive_rows_lstm'] / totals['rows_lstm']:.6f}")

    for filename, one_s_info, lstm_info in reports:
        print_file_report(filename, one_s_info, lstm_info, args.process_limit)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
