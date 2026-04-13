import pandas as pd


def read_irp_file(filepath):
    # Read as string first to avoid mixed-type parse failures. We will coerce
    # only the fields required by downstream logic and drop rows that remain invalid.
    df = pd.read_csv(
        filepath,
        delimiter='\t',
        on_bad_lines='skip',
        low_memory=False,
        dtype=str,
    )

    required_cols = ['time', 'major_opr', 'file_name', 'is_rename', 'is_delete', 'file_size']
    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA

    # Normalize text fields used by string operations.
    df['major_opr'] = df['major_opr'].fillna('').astype(str)
    df['file_name'] = df['file_name'].fillna('').astype(str)

    # Coerce numeric fields; rows with invalid time cannot be used and are ignored.
    for col in ['time', 'is_rename', 'is_delete', 'file_size']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df[df['time'].notna()].copy()
    df['time'] = df['time'].astype('int64')
    df['is_rename'] = df['is_rename'].fillna(0).astype('int64')
    df['is_delete'] = df['is_delete'].fillna(0).astype('int64')
    df['file_size'] = df['file_size'].fillna(0).astype('int64')

    df['is_read'] = (df['major_opr'] == 'IRP_MJ_READ').astype(int)
    df['is_write'] = (df['major_opr'] == 'IRP_MJ_WRITE').astype(int)
    df['is_query_information'] = (df['major_opr'] == 'IRP_MJ_QUERY_INFORMATION').astype(int)

    return df
