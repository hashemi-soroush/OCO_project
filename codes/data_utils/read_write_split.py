import os
import json

import numpy as np

from codes.data_utils import SPLIT_DATASET_ROOT, MAX_FEATURE_SIZE


def read_split_dataset(split_dataset_file_path):
    with open(split_dataset_file_path) as f:
        split_dataset = json.load(f)

        X = np.zeros((len(split_dataset), MAX_FEATURE_SIZE), np.float32)
        G = []
        for i, sample in enumerate(split_dataset):
            x = sample['features']
            g = sample['label']

            X[i, :len(x)] = x
            G.append(g)

        return X, G


def read_online_dataset():
    online_dataset_flie_path = os.path.join(SPLIT_DATASET_ROOT, 'online_dataset.json')
    return read_split_dataset(online_dataset_flie_path)


def read_offline_dataset():
    offline_dataset_flie_path = os.path.join(SPLIT_DATASET_ROOT, 'offline_dataset.json')
    return read_split_dataset(offline_dataset_flie_path)
