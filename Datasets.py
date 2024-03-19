# Datasets
import os
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pytorch_lightning as L
import json


### Helper functions

## Dataset functions
class PancreasDataset(Dataset):  # Input for the LightningDataModule
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def n_classes(self):
        return len(np.unique(self.labels))

    def __getitem__(self, idx):
        return self.data[idx,:], self.labels[idx]


class LitPancreasDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir,
        label_conversion_file,
        batch_size,
        train_techs,
        test_techs,
        validation_size,
        min_celltypes,
    ):
        super().__init__()
        self.data_dir = data_dir
        with open(label_conversion_file) as json_file:
            self.conversion_dict = json.load(json_file)
        self.batch = batch_size
        self.train_techs = train_techs
        self.test_techs = test_techs
        self.val_size = validation_size
        self.min_celltypes = min_celltypes

    def prepare_data(self):
        h5ad = sc.read_h5ad(self.data_dir)
        self.tech_features = h5ad.obs["tech"]
        self.labels = h5ad.obs["celltype"]
        self.data = sparse.csr_matrix(h5ad.X)

    def setup(self, stage):
        idx_train = [t in self.train_techs for t in self.tech_features]
        idx_test = [t in self.test_techs for t in self.tech_features]


        labels_train, data_train = self.labels[idx_train], self.data[idx_train,:]
        data_trainfilter, labels_trainfilter = self.filter_counts_h5ad(
            data_train, labels_train, self.min_celltypes
        )

        X_train, X_val, y_train, y_val = train_test_split(
        data_trainfilter,
        labels_trainfilter,
        test_size=self.val_size,
        stratify=labels_trainfilter,
        random_state=0,
        )

        labels_train = [self.conversion_dict[i] for i in y_train]
        self.data_train = PancreasDataset(torch.from_numpy(X_train.todense()), torch.from_numpy(np.array(labels_train))) #Check this operation locally

        labels_val = [self.conversion_dict[i] for i in y_val]
        self.data_val = PancreasDataset(torch.from_numpy(X_val.todense()), torch.from_numpy(np.array(labels_val)))

        labels_test, data_test = self.labels[idx_test], self.data[idx_test,:]
        data_testfilter, labels_testfilter = self.filter_counts_h5ad(
            data_test, labels_test, self.min_celltypes
        )
        # Convert labels to categories and store the transition
        print(type(data_testfilter.todense()))
        print(data_testfilter.shape)
        labels_converted = [self.conversion_dict[i] for i in labels_testfilter]
        self.data_test = PancreasDataset(torch.from_numpy(data_testfilter.todense()), torch.from_numpy(np.array(labels_converted)))

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch)
    
    def predict_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch)

    def filter_counts_h5ad(self, data, labels, n):
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
    def n_classes(self):
        h5ad = sc.read_h5ad(self.data_dir)
        self.tech_features = h5ad.obs["tech"]
        self.labels = h5ad.obs["celltype"]
        self.data = sparse.csr_matrix(h5ad.X)
        data_filter, labels_filter = self.filter_counts_h5ad(
            self.data, self.labels, self.min_celltypes
        )
        return len(np.unique(labels_filter))