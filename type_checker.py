import argparse
import csv
import math
import re
from collections import Counter, defaultdict

INT_RE = re.compile(r"^[+-]?\d+$")

def infer_type(v: str) -> str:
    s = v.strip()
    if s == "":
        return "empty"
    if INT_RE.match(s):
        return "int"
    try:
        x = float(s)
        if math.isfinite(x):
            return "float"
        return "str"
    except ValueError:
        return "str"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="TSV file path (e.g., irplog txt)")
    ap.add_argument("--delimiter", default="\t")
    ap.add_argument("--max-lines", type=int, default=30, help="max mismatch lines per column to print")
    args = ap.parse_args()

    with open(args.path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=args.delimiter)
        try:
            header = next(reader)
        except StopIteration:
            print("Empty file")
            return

        ncols = len(header)
        type_counts = [Counter() for _ in range(ncols)]
        values_by_line = [defaultdict(str) for _ in range(ncols)]
        row_len_mismatch = []

        for line_no, row in enumerate(reader, start=2):
            if len(row) != ncols:
                row_len_mismatch.append((line_no, len(row)))
            if len(row) < ncols:
                row = row + [""] * (ncols - len(row))
            elif len(row) > ncols:
                row = row[:ncols]

            for i, val in enumerate(row):
                t = infer_type(val)
                type_counts[i][t] += 1
                values_by_line[i][line_no] = (t, val)

    print(f"File: {args.path}")
    print(f"Columns: {ncols}")
    if row_len_mismatch:
        print("\n[Row length mismatch]")
        for ln, got in row_len_mismatch[:50]:
            print(f"  line {ln}: field_count={got}")
        if len(row_len_mismatch) > 50:
            print(f"  ... {len(row_len_mismatch)-50} more")

    for i, col in enumerate(header):
        counts = type_counts[i]
        non_empty = Counter({k: v for k, v in counts.items() if k != "empty"})
        dominant = non_empty.most_common(1)[0][0] if non_empty else "empty"

        print(f"\n[{i}] {col}")
        print(f"  counts: {dict(counts)}")
        print(f"  dominant(non-empty): {dominant}")

        mismatches = []
        for ln, (t, v) in values_by_line[i].items():
            if t == "empty":
                continue
            if t != dominant:
                preview = v if len(v) <= 80 else v[:77] + "..."
                mismatches.append((ln, t, preview))

        if not mismatches:
            print("  mismatch lines: none")
        else:
            print(f"  mismatch lines ({len(mismatches)}):")
            for ln, t, pv in mismatches[:args.max_lines]:
                print(f"    line {ln}: type={t}, value={pv!r}")
            if len(mismatches) > args.max_lines:
                print(f"    ... {len(mismatches)-args.max_lines} more")

if __name__ == "__main__":
    main()