import os
import torch
import numpy as np
from torch.utils.data import Dataset


class BCIDataset(Dataset):
    """PyTorch Dataset wrapper for BCI Competition III-IIIb EEG data."""
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = torch.tensor(self.data[index, ...], dtype=torch.float32)
        label = torch.tensor(self.label[index], dtype=torch.int64)
        return data, label

    def __len__(self):
        return self.data.shape[0]


def read_bci_data(data_dir='./data'):
    """
    Loads and preprocesses the BCI Competition III-IIIb dataset (S4b and X11b).

    Args:
        data_dir (str): Directory containing the .npz data files.

    Returns:
        tuple: (train_data, train_label, test_data, test_label)
            - train_data, test_data shape: (N, 1, 2, 750)
            - train_label, test_label shape: (N,)
    """
    S4b_train = np.load(os.path.join(data_dir, 'S4b_train.npz'))
    X11b_train = np.load(os.path.join(data_dir, 'X11b_train.npz'))
    S4b_test = np.load(os.path.join(data_dir, 'S4b_test.npz'))
    X11b_test = np.load(os.path.join(data_dir, 'X11b_test.npz'))

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    # Convert labels from {1, 2} to {0, 1}
    train_label = train_label - 1
    test_label = test_label - 1

    # Reshape from (N, 750, 2) -> (N, 1, 2, 750)
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    # Replace NaN values with the mean of valid entries
    train_mask = np.where(np.isnan(train_data))
    train_data[train_mask] = np.nanmean(train_data)

    test_mask = np.where(np.isnan(test_data))
    test_data[test_mask] = np.nanmean(test_data)

    print(f"Loaded dataset shapes -> Train: {train_data.shape}, {train_label.shape} | Test: {test_data.shape}, {test_label.shape}")
    return train_data, train_label, test_data, test_label