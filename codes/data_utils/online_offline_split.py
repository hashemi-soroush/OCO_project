import os
import json

import numpy as np
from scipy import random

from codes.data_utils import SPLIT_DATASET_ROOT


def split_online_offline(dataset, online_ratio):
    online_size = int(len(dataset) * online_ratio)
    online_inds = random.choice(list(range(len(dataset))), online_size, replace=False)
    online_mask = np.zeros(len(dataset))
    online_mask[online_inds] = 1

    offline_mask = np.ones(len(dataset))
    offline_mask[online_inds] = 0.0

    dataset = np.asarray(dataset)
    online_dataset = dataset[online_mask == 1.0]
    offline_dataset = dataset[offline_mask == 1.0]

    return online_dataset, offline_dataset


def save_split_dataset(split_dataset, name):
    writable_dataset = []
    for sample in split_dataset:
        sample['features'] = sample['features'].tolist()
        writable_dataset.append(sample)

    split_dataset_file_path = os.path.join(SPLIT_DATASET_ROOT, '{0}_dataset.json'.format(name))
    with open(split_dataset_file_path, 'wt') as f:
        json.dump(writable_dataset, f, indent=4, sort_keys=True)
