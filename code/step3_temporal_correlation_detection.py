import pandas as pd
from read_hpc_file import read_hpc_file
from read_irp_file import read_irp_file
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
from config import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import pandas as pd
import os
from torch.utils.data import Dataset
from joblib import load
import torch
from config import FEATURE_PATH, RESULT_PATH, MODEL_PATH


features = []
for i in range(10):
    features.append(f'read_{i}')
    features.append(f'write_{i}')
    features.append(f'rename_{i}')
    features.append(f'delete_{i}')
    features.append(f'filesize_{i}')
    features.append(f'instructions_{i}')
    features.append(f'branchinstructions_{i}')
    features.append(f'branchmispredicts_{i}')
    features.append(f'llcrefs_{i}')
    features.append(f'llcmisses_{i}')


def merge_dfs(dir):
    all_dfs = [pd.read_csv(os.path.join(dir, f), index_col=0) for f in os.listdir(dir)]
    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_df.reset_index(drop=True, inplace=True)
    return merged_df


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def print_score_stats(label, feature_df):
    print(f'=== {label} score statistics ===')
    for score_col in ['score_for_0', 'score_for_1']:
        series = feature_df[score_col]
        print(
            f'{label}, {score_col}: '
            f"avg={series.mean():.6f}, min={series.min():.6f}, "
            f"max={series.max():.6f}, std={series.std(ddof=0):.6f}"
        )
    print('')


def main():
    scaler = load(f'{MODEL_PATH}\\tc_detection_scaler.joblib')

    clf = LSTMModel(input_size=11, hidden_size=50, num_layers=1, num_classes=2)
    clf.load_state_dict(torch.load(f'{MODEL_PATH}\\tc_detection_clf.pth'))
    clf.eval()

    all_result_dfs = []

    for label in ['benign', 'ransomware']:
        feature_df = merge_dfs(f'{FEATURE_PATH}\\lstm\\{label}')

        X = feature_df[features].values
        # normalized
        X = scaler.transform(X)
        # transform to time series
        X = X.reshape(-1, 10, X.shape[1] // 10)
        # transform to tensor
        X = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            logits = clf(X)

        _, pred = torch.max(logits, 1)
        pred = pred.view(-1).tolist()
        feature_df['score_for_0'] = logits[:, 0].cpu().numpy()
        feature_df['score_for_1'] = logits[:, 1].cpu().numpy()
        feature_df['predict'] = pred
        feature_df['label'] = 0 if label == 'benign' else 1

        feature_df['Second'] = feature_df['starttime'].apply(lambda x: x // 10000000)

        print_score_stats(label, feature_df)

        feature_df.to_csv(f'{RESULT_PATH}\\tc_detection_result_{label}.csv')
        all_result_dfs.append(feature_df[['label', 'predict', 'score_for_0', 'score_for_1']].copy())

    combined_df = pd.concat(all_result_dfs, ignore_index=True)
    bucket_df_map = {
        'true positive (TP)': combined_df[(combined_df['label'] == 1) & (combined_df['predict'] == 1)],
        'false positive (FP)': combined_df[(combined_df['label'] == 0) & (combined_df['predict'] == 1)],
        'true negative (TN)': combined_df[(combined_df['label'] == 0) & (combined_df['predict'] == 0)],
        'false negative (FN)': combined_df[(combined_df['label'] == 1) & (combined_df['predict'] == 0)],
    }

    print('=== confusion-bucket score statistics ===')
    for bucket_name, bucket_df in bucket_df_map.items():
        print(f'[{bucket_name}] count={len(bucket_df)}')
        if len(bucket_df) == 0:
            print('  score_for_0: avg=0.000000, min=0.000000, max=0.000000, std=0.000000')
            print('  score_for_1: avg=0.000000, min=0.000000, max=0.000000, std=0.000000')
            continue
        for score_col in ['score_for_0', 'score_for_1']:
            series = bucket_df[score_col]
            print(
                f"  {score_col}: avg={series.mean():.6f}, min={series.min():.6f}, "
                f"max={series.max():.6f}, std={series.std(ddof=0):.6f}"
            )


if __name__ == '__main__':
    main()
