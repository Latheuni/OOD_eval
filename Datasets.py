# Datasets
import os
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import lightning as L


### Helper functions
def filter_counts_h5ad(data, labels, n):
    """Filter celltypes with less than 10 instances
    Parameters
    ----------
    data : annData
    labels : Series
    n : int
        celltypes with less than n occurences will be filtered out
    """
    s = labels.value_counts() < n
    celltype_to_filter = s.index[s].to_list()
    idx_keep = [l not in celltype_to_filter for l in labels]

    return (data[idx_keep, :], labels[idx_keep])


## Dataset functions
class PancreasDataset(Dataset):
    def __init__(self, data_h5ad, tech_list, min_celltypes):
        h5ad = sc.read_h5ad(data_h5ad)  # Should delete this from memory
        self.tech_list = tech_list
        self.labels = h5ad.obs["celltype"]
        self.data = sparse.csr_matrix(h5ad.X)
        self.tech_features = h5ad.obs["tech"]
        self.cut_off = min_celltypes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Select correct subpart of the data

        idx_data = [t in self.tech_list for t in self.tech_features]
        data_tech = self.data[idx_data, :]

        # Filter celltypes with less than 10 occurences
        data_filter, labels_filter = filter_counts_h5ad(
            self.data, self.labels, self.cut_off
        )

        # Retrieve sample
        data_filter_array = np.array(
            data_filter[idx, :].todense()
        ).flatten()  # convert numpy matrix to array and flatten from [[]] format
        return torch.from_numpy(data_filter_array), labels_filter[idx]


class LitPancreasDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        train_techs,
        test_techs,
        validation_size,
        min_celltypes,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch = batch_size
        self.train_techs = train_techs
        self.test_techs = test_techs
        self.val_size = validation_size
        self.min_celltypes = min_celltypes

        def train_validation_split(self, train_data_x, train_data_y):
            X_train, X_val, y_train, y_val = train_test_split(
                train_data_x.numpy(),
                train_data_y,
                test_size=self.val_size,
                stratify=train_data_y,
                random_state=0,
            )
            val_data = torch.from_numpy(X_val), y_val
            train_data = torch.from_numpy(X_train), y_train
            return train_data, val_data

        def setup(self):
            train_val_x_tensor, train_val_y = PancreasDataset(
                self.data_dir, self.train_techs, self.min_celltypes
            )
            data_train, data_val = self.train_validation_split(
                train_val_x_tensor, train_val_y
            )

            self.data_train = data_train
            self.data_val = data_val
            self.data_test = PancreasDataset(
                self.data_dir, self.test_techs, self.min_celltypes
            )

        def train_dataloader(self):
            return DataLoader(self.data_train, batchsize=self.batch)

        def val_dataloader(self):
            return DataLoader(self.data_val, batchsize=self.batch)

        def test_dataloader(self):
            return DataLoader(self.data_test, batchsize=self.batch)
