import os
import json

import numpy as np

from codes.data_utils import SPLIT_DATASET_ROOT


def read_split_dataset(split_dataset_file_path):
    with open(split_dataset_file_path) as f:
        split_dataset = json.load(f)

        x, g = [], []
        for sample in split_dataset:
            x.append(sample['features'])
            g.append(sample['label'])

        x = np.asarray(x, np.float32)
        g = np.asarray(g, str)

        return x, g


def read_online_dataset():
    online_dataset_flie_path = os.path.join(SPLIT_DATASET_ROOT, 'online_dataset.json')
    return read_split_dataset(online_dataset_flie_path)


def read_offline_dataset():
    offline_dataset_flie_path = os.path.join(SPLIT_DATASET_ROOT, 'offline_dataset.json')
    return read_split_dataset(offline_dataset_flie_path)
