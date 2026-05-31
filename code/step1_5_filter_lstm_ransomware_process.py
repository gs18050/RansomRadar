import argparse
import importlib.util
import shutil
from pathlib import Path
from typing import Dict

import pandas as pd


DEFAULT_INPUT_DIR_NAME = "lstm"
DEFAULT_OUTPUT_DIR_NAME = "lstm_process_filtered"


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


def clean_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed") or str(c) == ""]
    return df.drop(columns=unnamed) if unnamed else df


def filter_ransomware_file(
    input_path: Path,
    output_path: Path,
    sample_process: Dict[str, str],
    write_empty: bool,
) -> Dict[str, object]:
    df = clean_unnamed_columns(pd.read_csv(input_path))
    if "sample" not in df.columns or "process" not in df.columns:
        raise ValueError(f"{input_path} must contain 'sample' and 'process' columns")

    target_processes = {
        resolve_malicious_process(sample, sample_process, input_path.name)
        for sample in df["sample"].dropna().unique()
    }
    target_processes = {process for process in target_processes if process}
    if not target_processes:
        target_processes = {resolve_malicious_process(input_path.stem, sample_process, input_path.name)}
        target_processes = {process for process in target_processes if process}

    filtered = df[df["process"].isin(target_processes)].copy()
    if filtered.empty and not write_empty:
        if output_path.exists():
            output_path.unlink()
        return {
            "filename": input_path.name,
            "input_rows": int(len(df)),
            "output_rows": 0,
            "target_processes": sorted(target_processes),
            "written": False,
            "reason": "no_matching_ransomware_process_rows",
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_path, index=False)
    return {
        "filename": input_path.name,
        "input_rows": int(len(df)),
        "output_rows": int(len(filtered)),
        "target_processes": sorted(target_processes),
        "written": True,
        "reason": "filtered",
    }


def copy_benign_file(input_path: Path, output_path: Path) -> Dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)
    row_count = len(pd.read_csv(input_path, usecols=["sample"]))
    return {
        "filename": input_path.name,
        "input_rows": int(row_count),
        "output_rows": int(row_count),
        "written": True,
        "reason": "copied_benign",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a filtered LSTM feature directory where ransomware files keep only "
            "the resolved ransomware process rows. Benign LSTM files are copied unchanged."
        )
    )
    parser.add_argument("--features-root", default=str(repo_root() / "features"))
    parser.add_argument("--input-dir-name", default=DEFAULT_INPUT_DIR_NAME)
    parser.add_argument("--output-dir-name", default=DEFAULT_OUTPUT_DIR_NAME)
    parser.add_argument(
        "--write-empty-ransomware-files",
        action="store_true",
        help="Write empty ransomware CSVs when the resolved process has no rows. Default skips them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    sample_process = load_sample_process_map(root)
    features_root = Path(args.features_root).resolve()
    input_root = features_root / args.input_dir_name
    output_root = features_root / args.output_dir_name

    if not input_root.exists():
        raise RuntimeError(f"input LSTM feature directory does not exist: {input_root}")

    summaries = []
    for label in ("benign", "ransomware"):
        input_dir = input_root / label
        output_dir = output_root / label
        if not input_dir.exists():
            raise RuntimeError(f"missing input directory: {input_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        for input_path in sorted(input_dir.glob("*.csv")):
            output_path = output_dir / input_path.name
            if label == "benign":
                summaries.append(copy_benign_file(input_path, output_path))
            else:
                summaries.append(
                    filter_ransomware_file(
                        input_path,
                        output_path,
                        sample_process,
                        write_empty=args.write_empty_ransomware_files,
                    )
                )

    benign = [item for item in summaries if item["reason"] == "copied_benign"]
    ransomware = [item for item in summaries if item["reason"] != "copied_benign"]
    skipped = [item for item in ransomware if not item["written"]]
    input_rows = sum(int(item["input_rows"]) for item in summaries)
    output_rows = sum(int(item["output_rows"]) for item in summaries)
    ransomware_input_rows = sum(int(item["input_rows"]) for item in ransomware)
    ransomware_output_rows = sum(int(item["output_rows"]) for item in ransomware)

    print(f"input: {input_root}")
    print(f"output: {output_root}")
    print(f"benign files copied: {len(benign)}")
    print(f"ransomware files processed: {len(ransomware)}")
    print(f"ransomware files skipped with no matching process rows: {len(skipped)}")
    print(f"total rows: input={input_rows} output={output_rows}")
    print(f"ransomware rows: input={ransomware_input_rows} output={ransomware_output_rows}")
    if skipped:
        print("skipped ransomware files:")
        for item in skipped:
            print(f"  {item['filename']} target_processes={item['target_processes']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
