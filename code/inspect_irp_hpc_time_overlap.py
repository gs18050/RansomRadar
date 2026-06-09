import argparse
import os
from pathlib import Path

import pandas as pd

try:
    from config import FEATURE_PATH, IRP_ROOT_PATH
except ImportError:
    FEATURE_PATH = r"features"
    IRP_ROOT_PATH = r"dataset\irplog"


def read_csv_any(path: Path, **kwargs):
    return pd.read_csv(path, on_bad_lines="skip", low_memory=False, **kwargs)


def inspect_pair(irp_path: Path, hpc_path: Path):
    result = {
        "sample": irp_path.stem,
        "irp_path": str(irp_path),
        "hpc_path": str(hpc_path),
        "status": "ok",
        "reason": "",
    }
    if not hpc_path.exists():
        result.update(status="missing_hpc", reason="matching features/100ms csv not found")
        return result

    try:
        irp_df = read_csv_any(irp_path, sep="\t")
    except Exception as exc:
        result.update(status="bad_irp", reason=f"failed to read IRP TSV: {exc}")
        return result
    try:
        hpc_df = read_csv_any(hpc_path)
    except Exception as exc:
        result.update(status="bad_hpc", reason=f"failed to read HPC CSV: {exc}")
        return result

    if "time" not in irp_df.columns:
        result.update(
            status="bad_irp",
            reason=f"missing time column; columns={list(irp_df.columns)[:20]}",
        )
        return result
    if "major_opr" not in irp_df.columns:
        result.update(
            status="bad_irp",
            reason=f"missing major_opr column; columns={list(irp_df.columns)[:20]}",
        )
        return result
    if "fromtime" not in hpc_df.columns or "totime" not in hpc_df.columns:
        result.update(
            status="bad_hpc",
            reason=f"missing fromtime/totime columns; columns={list(hpc_df.columns)[:20]}",
        )
        return result

    irp_time = pd.to_numeric(irp_df["time"], errors="coerce")
    hpc_from = pd.to_numeric(hpc_df["fromtime"], errors="coerce")
    hpc_to = pd.to_numeric(hpc_df["totime"], errors="coerce")
    irp_time = irp_time[irp_time.notna()]
    hpc_from = hpc_from[hpc_from.notna()]
    hpc_to = hpc_to[hpc_to.notna()]
    if irp_time.empty:
        result.update(status="bad_irp", reason="no valid numeric IRP time rows")
        return result
    if hpc_from.empty or hpc_to.empty:
        result.update(status="bad_hpc", reason="no valid numeric HPC time rows")
        return result

    hpc_min = float(hpc_from.min())
    hpc_max = float(hpc_to.max())
    irp_min = float(irp_time.min())
    irp_max = float(irp_time.max())

    write_df = irp_df[irp_df["major_opr"] == "IRP_MJ_WRITE"]
    write_time = pd.to_numeric(write_df["time"], errors="coerce") if len(write_df) else pd.Series(dtype="float64")
    write_time = write_time[write_time.notna()]
    write_overlap = int(((write_time >= hpc_min) & (write_time < hpc_max)).sum()) if len(write_time) else 0
    any_overlap = int(((irp_time >= hpc_min) & (irp_time < hpc_max)).sum())

    result.update(
        irp_rows=int(len(irp_df)),
        hpc_rows=int(len(hpc_df)),
        write_rows=int(len(write_df)),
        any_overlap_rows=any_overlap,
        write_overlap_rows=write_overlap,
        irp_min=irp_min,
        irp_max=irp_max,
        hpc_min=hpc_min,
        hpc_max=hpc_max,
        hpc_start_minus_irp_start=int(hpc_min - irp_min),
        hpc_end_minus_irp_end=int(hpc_max - irp_max),
    )
    if write_overlap == 0:
        result["status"] = "zero_write_overlap"
        result["reason"] = "no IRP_MJ_WRITE rows inside HPC time range"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=["benign", "ransomware", "all"], default="ransomware")
    parser.add_argument("--irp-root", default=IRP_ROOT_PATH)
    parser.add_argument("--features-root", default=FEATURE_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    labels = ["benign", "ransomware"] if args.label == "all" else [args.label]
    rows = []
    for label in labels:
        irp_dir = Path(args.irp_root) / label
        hpc_dir = Path(args.features_root) / "100ms" / label
        paths = sorted(irp_dir.glob("*.txt"))
        if args.limit is not None:
            paths = paths[: args.limit]
        for irp_path in paths:
            row = inspect_pair(irp_path, hpc_dir / f"{irp_path.stem}.csv")
            row["label"] = label
            rows.append(row)

    df = pd.DataFrame(rows)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"wrote {args.output}")

    if df.empty:
        print("no IRP files found")
        return

    print("status counts:")
    print(df["status"].value_counts(dropna=False).to_string())
    print("")

    for status in ["bad_irp", "bad_hpc", "missing_hpc", "zero_write_overlap"]:
        subset = df[df["status"] == status]
        if subset.empty:
            continue
        print(f"{status} examples:")
        cols = [c for c in ["label", "sample", "reason", "write_rows", "write_overlap_rows", "hpc_start_minus_irp_start", "hpc_end_minus_irp_end"] if c in subset.columns]
        print(subset[cols].head(10).to_string(index=False))
        print("")

    ok = df[df["status"] == "ok"]
    if not ok.empty:
        print("ok overlap summary:")
        print(ok[["write_rows", "write_overlap_rows", "any_overlap_rows"]].describe().to_string())


if __name__ == "__main__":
    main()
