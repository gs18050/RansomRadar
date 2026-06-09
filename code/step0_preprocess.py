from config import RAW_ROOT_PATH, HPC_ROOT_path, PARSER_PATH, PARSER_PROFILE_PATH
import argparse
import os
import shutil


TICKS_PER_SECOND = 10000000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--labels',
        default='ransomware,benign',
        help='Comma-separated labels to preprocess. Default: ransomware,benign.',
    )
    parser.add_argument(
        '--starttime-offset-seconds',
        type=float,
        default=0.0,
        help=(
            'Add this offset to generated *_starttime.txt values, in seconds. '
            'Use -3600 to compensate old UTC+8 parser output on UTC+9 captures.'
        ),
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing hpcdata CSV and *_starttime.txt outputs.',
    )
    parser.add_argument(
        '--overwrite-hpc',
        action='store_true',
        help='Overwrite existing hpcdata CSV outputs only.',
    )
    parser.add_argument(
        '--overwrite-starttime',
        action='store_true',
        help='Overwrite existing *_starttime.txt outputs only.',
    )
    parser.add_argument(
        '--adjust-existing-starttime',
        action='store_true',
        help=(
            'Apply --starttime-offset-seconds to existing *_starttime.txt files without '
            'rerunning the parser. Be careful not to apply the same offset twice.'
        ),
    )
    return parser.parse_args()


def run_command(command):
    exit_code = os.system(command)
    if exit_code != 0:
        raise RuntimeError(f'command failed with exit code {exit_code}: {command}')


def apply_starttime_offset(path, offset_seconds):
    if offset_seconds == 0:
        return
    offset_ticks = int(round(offset_seconds * TICKS_PER_SECOND))
    with open(path, 'r') as f:
        starttime = int(f.read().strip())
    adjusted = starttime + offset_ticks
    with open(path, 'w') as f:
        f.write(str(adjusted))


def main():
    args = parse_args()
    labels = [label.strip() for label in args.labels.split(',') if label.strip()]
    for label in labels:
        for file in os.listdir(f'{RAW_ROOT_PATH}\\{label}'):
            if not file.lower().endswith('etl'):
                continue

            sample = file.split('.')[0]
            hpc_csv_path = f'{HPC_ROOT_path}\\{label}\\{sample}.csv'
            hpc_temp_dir = f'{HPC_ROOT_path}\\{label}\\{sample}'
            starttime_path = f'{HPC_ROOT_path}\\{label}\\{sample}_starttime.txt'
            os.makedirs(f'{HPC_ROOT_path}\\{label}', exist_ok=True)
            
            print(f'start process {file}')
            try:
                # parse ETL file
                if args.overwrite or args.overwrite_hpc or not os.path.exists(hpc_csv_path):
                    if os.path.exists(hpc_temp_dir):
                        shutil.rmtree(hpc_temp_dir)
                    run_command(
                        f'wpaexporter.exe /tti -i {RAW_ROOT_PATH}\\{label}\\{file} '
                        f'-profile {PARSER_PROFILE_PATH} -outputfolder {hpc_temp_dir}'
                    )
                    os.replace(f'{hpc_temp_dir}\\PMC_Summary_Table_test.csv', hpc_csv_path)
                    shutil.rmtree(hpc_temp_dir)
                else:
                    print(f'skip existing HPC CSV: {hpc_csv_path}')

                # get ETL start time
                if args.overwrite or args.overwrite_starttime or not os.path.exists(starttime_path):
                    run_command(f'{PARSER_PATH} {RAW_ROOT_PATH}\\{label}\\{sample}.etl {starttime_path}')
                    apply_starttime_offset(starttime_path, args.starttime_offset_seconds)
                elif args.adjust_existing_starttime:
                    apply_starttime_offset(starttime_path, args.starttime_offset_seconds)
                    print(f'adjust existing starttime: {starttime_path}')
                else:
                    print(f'skip existing starttime: {starttime_path}')
            except Exception as e:
                print(f'error process {RAW_ROOT_PATH}\\{label}\\{file}')
                print(f'  reason: {e}')
                continue
            print(f'finish process {RAW_ROOT_PATH}\\{label}\\{file}')


if __name__ == '__main__':
    main()
