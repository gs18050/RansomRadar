import os
import argparse
import pandas as pd
from config import (
    RESULT_PATH,
    BACKGROUND_BASELINE_PATH,
    BACKGROUND_META_PATH,
    BACKGROUND_PROCESS_SET_PATH,
)
from sample_process import sample_process


def normalize_process_name(process_name):
    return str(process_name).strip().lower()


def load_result_pair(encryption_path, tc_path):
    encryption_df = pd.read_csv(encryption_path)
    encryption_df = encryption_df.rename(columns={'predict': 'enc_predict'})
    encryption_df = encryption_df[['Sample', 'Process', 'Second', 'enc_predict']]
    encryption_df['enc_predict'] = encryption_df['enc_predict'].astype(bool)

    tc_df = pd.read_csv(tc_path)
    tc_df = tc_df.rename(columns={'sample': 'Sample', 'process': 'Process', 'predict': 'tc_predict'})
    branchinstructions_cols = [c for c in tc_df.columns if c.startswith('branchinstructions_')]
    branchmispredicts_cols = [c for c in tc_df.columns if c.startswith('branchmispredicts_')]

    if branchinstructions_cols:
        tc_df['branchinstructions_sum'] = tc_df[branchinstructions_cols].sum(axis=1)
    else:
        tc_df['branchinstructions_sum'] = 0.0

    if branchmispredicts_cols:
        tc_df['branchmispredicts_sum'] = tc_df[branchmispredicts_cols].sum(axis=1)
    else:
        tc_df['branchmispredicts_sum'] = 0.0

    tc_df = tc_df[['Sample', 'Process', 'Second', 'tc_predict', 'branchinstructions_sum', 'branchmispredicts_sum']]
    tc_df['tc_predict'] = tc_df['tc_predict'].astype(bool)

    df = pd.merge(encryption_df, tc_df, on=['Sample', 'Process', 'Second'], how='inner')
    df['result'] = df['enc_predict'] & df['tc_predict']
    return df


def discover_baseline_result_pairs():
    pairs = []
    for root, _, files in os.walk(BACKGROUND_BASELINE_PATH):
        file_set = set(files)
        if 'encryption_detection_result_benign.csv' in file_set and 'tc_detection_result_benign.csv' in file_set:
            pairs.append((
                f'{root}\\encryption_detection_result_benign.csv',
                f'{root}\\tc_detection_result_benign.csv',
            ))
    return sorted(pairs)


def build_background_process_union():
    os.makedirs(BACKGROUND_BASELINE_PATH, exist_ok=True)
    os.makedirs(BACKGROUND_META_PATH, exist_ok=True)

    process_union = set()
    valid_pairs = 0

    for encryption_path, tc_path in discover_baseline_result_pairs():
        try:
            df = load_result_pair(encryption_path, tc_path)
        except Exception as e:
            print(f'skip invalid baseline run: {encryption_path}, {tc_path}, reason: {e}')
            continue

        valid_pairs += 1
        for process in df['Process'].dropna().unique():
            process_union.add(normalize_process_name(process))

    with open(BACKGROUND_PROCESS_SET_PATH, 'w', encoding='utf-8') as f:
        for process in sorted(process_union):
            f.write(f'{process}\n')

    return process_union, valid_pairs


def summarize_group_level_detection(df):
    return (
        df.groupby(['Sample', 'Process'], as_index=False)['result']
        .any()
        .rename(columns={'result': 'detected'})
    )


def safe_mean(series):
    if len(series) == 0:
        return 0.0
    return float(series.mean())


def format_rate(detected, total):
    if total == 0:
        return 0.0
    return round(detected / total * 100, 2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--print-ransomware-caught',
        action='store_true',
        help='Print ransomware samples that were caught in step4 evaluation.',
    )
    parser.add_argument(
        '--print-ransomware-not-caught',
        action='store_true',
        help='Print ransomware samples that were not caught in step4 evaluation.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    background_process_union, valid_baseline_runs = build_background_process_union()
    print(f'background baseline runs loaded: {valid_baseline_runs}')
    print(f'background process union size: {len(background_process_union)}')

    for label in ['benign', 'ransomware']:
        df = load_result_pair(
            f'{RESULT_PATH}\\encryption_detection_result_{label}.csv',
            f'{RESULT_PATH}\\tc_detection_result_{label}.csv',
        )

        if label == 'ransomware':
            df = df[df.apply(lambda row: row['Process'] == sample_process.get(row['Sample'], ''), axis=1)]

        grouped_df = summarize_group_level_detection(df)
        total = len(grouped_df)
        cnt_detected = int(grouped_df['detected'].sum())

        if label == 'benign':
            print(f'benign, original false positive rate: {format_rate(cnt_detected, total)}% ({cnt_detected}/{total})')
            print(
                'benign, avg counters (all processes): '
                f"branchinstructions={safe_mean(df['branchinstructions_sum']):.2f}, "
                f"branchmispredicts={safe_mean(df['branchmispredicts_sum']):.2f}"
            )
            benign_fp_df = df[df['result']]
            benign_non_fp_df = df[~df['result']]
            print(
                'benign, avg counters (FP): '
                f"branchinstructions={safe_mean(benign_fp_df['branchinstructions_sum']):.2f}, "
                f"branchmispredicts={safe_mean(benign_fp_df['branchmispredicts_sum']):.2f}"
            )
            print(
                'benign, avg counters (non-FP): '
                f"branchinstructions={safe_mean(benign_non_fp_df['branchinstructions_sum']):.2f}, "
                f"branchmispredicts={safe_mean(benign_non_fp_df['branchmispredicts_sum']):.2f}"
            )

            grouped_df['is_background'] = grouped_df['Process'].apply(
                lambda p: normalize_process_name(p) in background_process_union
            )

            filtered_grouped_df = grouped_df[
                (~grouped_df['is_background']) | (grouped_df['detected'])
            ]

            filtered_total = len(filtered_grouped_df)
            filtered_detected = int(filtered_grouped_df['detected'].sum())
            excluded_non_fp_background = int((grouped_df['is_background'] & ~grouped_df['detected']).sum())
            included_background_fp = int((filtered_grouped_df['is_background'] & filtered_grouped_df['detected']).sum())

            print(
                f'benign, filtered false positive rate: {format_rate(filtered_detected, filtered_total)}% '
                f'({filtered_detected}/{filtered_total})'
            )
            print(f'benign, excluded background non-FP groups: {excluded_non_fp_background}')
            print(f'benign, included background FP groups: {included_background_fp}')
        elif label == 'ransomware':
            print(f'ransomware: recall: {format_rate(cnt_detected, total)}% ({cnt_detected}/{total})')
            print(
                'ransomware, avg counters (mapped ransomware process): '
                f"branchinstructions={safe_mean(df['branchinstructions_sum']):.2f}, "
                f"branchmispredicts={safe_mean(df['branchmispredicts_sum']):.2f}"
            )

            predicted_ransomware_df = df[df['result']]
            predicted_benign_df = df[~df['result']]
            print(
                'ransomware, avg counters (predicted ransomware): '
                f"branchinstructions={safe_mean(predicted_ransomware_df['branchinstructions_sum']):.2f}, "
                f"branchmispredicts={safe_mean(predicted_ransomware_df['branchmispredicts_sum']):.2f}"
            )
            print(
                'ransomware, avg counters (predicted benign): '
                f"branchinstructions={safe_mean(predicted_benign_df['branchinstructions_sum']):.2f}, "
                f"branchmispredicts={safe_mean(predicted_benign_df['branchmispredicts_sum']):.2f}"
            )
            if args.print_ransomware_caught or args.print_ransomware_not_caught:
                caught_df = grouped_df[grouped_df['detected']].sort_values(['Sample', 'Process'])
                not_caught_df = grouped_df[~grouped_df['detected']].sort_values(['Sample', 'Process'])

                if args.print_ransomware_caught:
                    print('ransomware caught:')
                    if caught_df.empty:
                        print('  (none)')
                    else:
                        for _, row in caught_df.iterrows():
                            print(f"  {row['Sample']} | {row['Process']}")

                if args.print_ransomware_not_caught:
                    print('ransomware not caught:')
                    if not_caught_df.empty:
                        print('  (none)')
                    else:
                        for _, row in not_caught_df.iterrows():
                            print(f"  {row['Sample']} | {row['Process']}")


if __name__ == '__main__':
    main()
