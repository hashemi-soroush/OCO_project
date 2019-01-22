import os

import numpy as np

from codes.data_utils import ORIGINAL_DATASET_ROOT_PATH


def read_original_sample(sample_file_path):
    label = os.path.basename(sample_file_path)[0]
    features = []
    diam = 0.0
    f = open(sample_file_path)
    for line in f:
        line = line.split()
        line = [float(line[i]) for i in range(3, len(line))]
        features += line[:-1]
        diam = line[-1]

    features.append(diam)
    features = np.array(features)
    return label, features


def read_original_dataset(dataset_root_path):
    dataset = []
    for sample_file_name in os.listdir(dataset_root_path):
        sample_file_path = os.path.join(dataset_root_path, sample_file_name)

        label, features = read_original_sample(sample_file_path)

        new_data = {
            'label': label,
            'features': features
        }
        dataset.append(new_data)

    return dataset


def read_original_datasets():
    original_datasets_root_path = [
        os.path.join(ORIGINAL_DATASET_ROOT_PATH, 'test'),
        os.path.join(ORIGINAL_DATASET_ROOT_PATH, 'learn'),
    ]
    original_datasets = []
    for original_dataset_root_path in original_datasets_root_path:
        original_dataset = read_original_dataset(original_dataset_root_path)
        original_datasets += original_dataset

    return original_datasets
