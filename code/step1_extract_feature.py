from config import HPC_ROOT_path, IRP_ROOT_PATH, RAW_ROOT_PATH, FEATURE_PATH
from read_irp_file import read_irp_file

import argparse
import os
import pandas as pd
import numpy as np
import pandas as pd


WRITE_ENTROPY_FEATURES = [
    'write_entropy_byte_weighted_avg',
    'write_entropy_byte_max',
]

HPC_COUNTER_COLUMNS = [
    'InstructionsRetiredFixed',
    'BranchInstructionRetired',
    'BranchMispredictsRetired',
    'LLCReference',
    'LLCMisses',
]


def read_hpc_starttime(filepath):
    sample = os.path.basename(filepath).split('.')[0]
    dirname = os.path.dirname(filepath)
    starttime_path = f'{dirname}/{sample}_starttime.txt'
    if not os.path.exists(starttime_path):
        return None
    with open(starttime_path, 'r') as f:
        return int(f.read().strip())


def parse_hpc_timestamp_ms(series):
    text = series.astype('string')
    text = text.str.split('.', n=1).str[0].str.replace(',', '', regex=False)
    return pd.to_numeric(text, errors='coerce')


def iter_hpc_feature_chunks(filepath, chunksize):
    starttime = read_hpc_starttime(filepath)
    if starttime is None:
        return

    sample = os.path.basename(filepath).split('.')[0]
    usecols = ['Timestamp (ms)', 'Counter', 'Process']
    for chunk in pd.read_csv(
        filepath,
        usecols=lambda col: col in usecols,
        on_bad_lines='skip',
        chunksize=chunksize,
        dtype={'Timestamp (ms)': 'string', 'Counter': 'string', 'Process': 'string'},
    ):
        if not set(usecols).issubset(chunk.columns):
            continue
        chunk = chunk.dropna(subset=usecols).copy()
        if chunk.empty:
            continue
        relative_ms = parse_hpc_timestamp_ms(chunk['Timestamp (ms)'])
        chunk = chunk[relative_ms.notna()].copy()
        if chunk.empty:
            continue
        chunk['Timestamp'] = relative_ms.loc[chunk.index].astype(np.int64) * 10000 + starttime
        chunk['Sample'] = sample
        yield chunk[['Sample', 'Counter', 'Process', 'Timestamp']]


def aggregate_hpc_counters(df, group_cols):
    result = (
        df.groupby(group_cols + ['Counter'], sort=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    result.columns.name = None
    for col in HPC_COUNTER_COLUMNS:
        if col not in result.columns:
            result[col] = 0
    return result


def aggregate_counter_columns(df, group_cols):
    result = (
        df.groupby(group_cols, sort=False, as_index=False)[HPC_COUNTER_COLUMNS]
        .sum()
    )
    return result


def population_std(values):
    return float(np.asarray(values, dtype=np.float64).std(ddof=0))


# calculate features for every second
def calculate_1s_feature(filepath, targetpath, hpc_chunksize):  
    print(f'extract 1s feature for {targetpath}')

    if os.path.exists(targetpath):
        return

    partials = []
    for df in iter_hpc_feature_chunks(filepath, hpc_chunksize):
        df['Second'] = (df['Timestamp'] / 10000000).astype(np.int64)
        df['100ms'] = (df['Timestamp'] / 1000000).astype(np.int64)
        partials.append(aggregate_hpc_counters(df, ['Sample', 'Process', 'Second', '100ms']))

    if not partials:
        return

    ms100_df = aggregate_counter_columns(
        pd.concat(partials, ignore_index=True),
        ['Sample', 'Process', 'Second', '100ms'],
    )
    ms100_df = ms100_df[
        (ms100_df['InstructionsRetiredFixed'] != 0)
        & (ms100_df['BranchInstructionRetired'] != 0)
        & (ms100_df['LLCReference'] != 0)
    ].copy()
    if ms100_df.empty:
        return

    ms100_df['branchinstructionrate'] = ms100_df['BranchInstructionRetired'] / ms100_df['InstructionsRetiredFixed']
    ms100_df['branchmispredictsrate'] = ms100_df['BranchMispredictsRetired'] / ms100_df['BranchInstructionRetired']
    ms100_df['llcrefrate'] = ms100_df['LLCReference'] / ms100_df['InstructionsRetiredFixed']
    ms100_df['llcmissrate'] = ms100_df['LLCMisses'] / ms100_df['LLCReference']

    result_df = (
        ms100_df.groupby(['Sample', 'Process', 'Second'], sort=False, as_index=False)
        .agg(
            avg_branchinstructionrate=('branchinstructionrate', 'mean'),
            std_branchinstructionrate=('branchinstructionrate', population_std),
            avg_branchmispredictsrate=('branchmispredictsrate', 'mean'),
            std_branchmispredictsrate=('branchmispredictsrate', population_std),
            avg_llcrefrate=('llcrefrate', 'mean'),
            std_llcrefrate=('llcrefrate', population_std),
            avg_llcmissrate=('llcmissrate', 'mean'),
            std_llcmissrate=('llcmissrate', population_std),
        )
    )

    os.makedirs(os.path.dirname(targetpath), exist_ok=True)
    result_df.to_csv(targetpath)


# calculate features for every 100ms
def calculate_100ms_feature(filepath, targetpath, hpc_chunksize):
    print(f'calculate 100ms feature for {targetpath}')

    if os.path.exists(targetpath):
        return

    partials = []
    for df in iter_hpc_feature_chunks(filepath, hpc_chunksize):
        df['100ms'] = (df['Timestamp'] / 1000000).astype(np.int64)
        partials.append(aggregate_hpc_counters(df, ['Sample', 'Process', '100ms']))

    if not partials:
        return

    result_df = aggregate_counter_columns(
        pd.concat(partials, ignore_index=True),
        ['Sample', 'Process', '100ms'],
    )
    result_df['fromtime'] = result_df['100ms'] * 1000000
    result_df['totime'] = (result_df['100ms'] + 1) * 1000000
    result_df = result_df.rename(
        columns={
            'InstructionsRetiredFixed': 'instructions',
            'BranchInstructionRetired': 'branchinstructions',
            'BranchMispredictsRetired': 'branchmispredicts',
            'LLCReference': 'llcreferences',
            'LLCMisses': 'llcmisses',
        }
    )
    result_df = result_df[
        [
            'Sample',
            'Process',
            'fromtime',
            'totime',
            'instructions',
            'branchinstructions',
            'branchmispredicts',
            'llcreferences',
            'llcmisses',
        ]
    ]

    os.makedirs(os.path.dirname(targetpath), exist_ok=True)
    result_df.to_csv(targetpath)


# calculate feature for lstm
def write_entropy_features(irp_data):
    write_irp = irp_data[irp_data['is_write'] == 1]
    if write_irp.empty:
        return {
            'write_entropy_byte_weighted_avg': 0.0,
            'write_entropy_byte_max': 0.0,
        }

    buffer_lengths = pd.to_numeric(write_irp['buffer_length'], errors='coerce').fillna(0.0)
    entropies = pd.to_numeric(write_irp['entropy_byte_based'], errors='coerce').fillna(0.0)
    total_bytes = float(buffer_lengths.sum())
    weighted_avg = float((entropies * buffer_lengths).sum() / total_bytes) if total_bytes > 0 else 0.0
    return {
        'write_entropy_byte_weighted_avg': weighted_avg,
        'write_entropy_byte_max': float(entropies.max()) if len(entropies) else 0.0,
    }


def calculate_lstm_feature(irp_path, hpc_path, output_path, label, use_write_entropy_features=False):  
    print(f'calculate lstm feature for {output_path}')

    if os.path.exists(output_path):
        return
      
    feature_cols = ['sample', 'process', 'starttime', 'label']
    for i in range(10):
        feature_cols.append(f'read_{i}')
        feature_cols.append(f'write_{i}')
        feature_cols.append(f'rename_{i}')
        feature_cols.append(f'delete_{i}')
        feature_cols.append(f'query_information_{i}')
        feature_cols.append(f'filesize_{i}')
        feature_cols.append(f'instructions_{i}')
        feature_cols.append(f'branchinstructions_{i}')
        feature_cols.append(f'branchmispredicts_{i}')
        feature_cols.append(f'llcrefs_{i}')
        feature_cols.append(f'llcmisses_{i}')
        if use_write_entropy_features:
            for feature in WRITE_ENTROPY_FEATURES:
                feature_cols.append(f'{feature}_{i}')

    feature_df = pd.DataFrame(columns=feature_cols)
    
    # get sample name
    sample = os.path.basename(irp_path).split('.')[0]
    
    try:
        irp_df = read_irp_file(irp_path)
        hpc_df = pd.read_csv(hpc_path, index_col=0) 

        hpc_df['Second'] = hpc_df['fromtime'].apply(lambda x: int(x / 1e7))
        irp_df['file_basename'] = irp_df['file_name'].apply(lambda x: x.split('.')[0].lower())

        # consider every process separately
        for (sample, process), sub_df in hpc_df.groupby(['Sample', 'Process']):
            SecondList = list(sub_df['Second'].unique())
            
            # accessed files
            read_files = set()

            for second in SecondList:
                starttime = second * 10000000
                row_data = {}
                row_data['sample'] = sample
                row_data['process'] = process
                row_data['starttime'] = starttime
                row_data['label'] = 0 if label == 'benign' else 1
                for i in range(10):
                    ms_start = starttime + i * 1000000
                    ms_end = starttime + (i + 1) * 1000000
                    hpc_data = hpc_df[hpc_df['fromtime'] == ms_start]
                    irp_data = irp_df[(ms_start <= irp_df['time']) & (irp_df['time'] < ms_end)]

                    # files being read before
                    read_files.update(set(irp_data[irp_data['is_write']==0]['file_basename'].unique()))

                    row_data[f'read_{i}'] = irp_data['is_read'].sum()
                    row_data[f'write_{i}'] = irp_data['is_write'].sum()
                    row_data[f'rename_{i}'] = irp_data['is_rename'].sum()
                    row_data[f'delete_{i}'] = irp_data['is_delete'].sum()
                    row_data[f'query_information_{i}'] = irp_data['is_query_information'].sum()
                    row_data[f'filesize_{i}'] = irp_data[irp_data['file_basename'].isin(read_files)]['file_size'].sum()
                    row_data[f'instructions_{i}'] = hpc_data['instructions'].sum()
                    row_data[f'branchinstructions_{i}'] = hpc_data['branchinstructions'].sum()
                    row_data[f'branchmispredicts_{i}'] = hpc_data['branchmispredicts'].sum()
                    row_data[f'llcrefs_{i}'] = hpc_data['llcreferences'].sum()
                    row_data[f'llcmisses_{i}'] = hpc_data['llcmisses'].sum()
                    if use_write_entropy_features:
                        for feature, value in write_entropy_features(irp_data).items():
                            row_data[f'{feature}_{i}'] = value

                feature_df.loc[len(feature_df)] = row_data

        # save result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        feature_df.to_csv(output_path)
    except Exception as e:
        print(f'error while processing {irp_path} {hpc_path} {e}')
        return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--use-write-entropy-features',
        action='store_true',
        help='Add write entropy LSTM features and write LSTM output to features/lstm_entropy.',
    )
    parser.add_argument(
        '--hpc-chunksize',
        type=int,
        default=500000,
        help='Rows per chunk when reading raw HPC CSV files. Lower this if step1 still runs out of memory.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    lstm_dir_name = 'lstm_entropy' if args.use_write_entropy_features else 'lstm'
    for label in ['benign', 'ransomware']:
        for file in os.listdir(f'{RAW_ROOT_PATH}\\{label}'):
            sample = file.split('.')[0]

            calculate_1s_feature(
                f'{HPC_ROOT_path}\\{label}\\{sample}.csv',
                f'{FEATURE_PATH}\\1s\\{label}\\{sample}.csv',
                args.hpc_chunksize,
            )

            calculate_100ms_feature(
                f'{HPC_ROOT_path}\\{label}\\{sample}.csv',
                f'{FEATURE_PATH}\\100ms\\{label}\\{sample}.csv',
                args.hpc_chunksize,
            )
            
            calculate_lstm_feature(
                f'{IRP_ROOT_PATH}\\{label}\\{sample}.txt',
                f'{FEATURE_PATH}\\100ms\\{label}\\{sample}.csv',
                f'{FEATURE_PATH}\\{lstm_dir_name}\\{label}\\{sample}.csv',
                label,
                use_write_entropy_features=args.use_write_entropy_features,
            )


if __name__ == '__main__':
    main()
