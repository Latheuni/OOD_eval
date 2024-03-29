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



## Lightning 
class PancreasDataset(Dataset):  # Input for the LightningDataModule
    """Input format needed for Lightning
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def n_classes(self):
        return len(np.unique(self.labels))

    def __getitem__(self, idx):
        return self.data[idx,:], self.labels[idx]


## DataModules
class LitPancreasDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir,
        data_file,
        label_conversion_file,
        batch_size,
        train_techs,
        OOD_techs,
        test_size, 
        validation_size,
        min_celltypes,
        num_workers,
        name,
        verbose = "True"
    ):
        """
        Test data consists out of 20% ID data (from the entire dataset) and the OOD data
        Val data consists out of 20% ID data
        Train data consists out of 80% ID data
        """

        super().__init__()
        self.data_dir = data_dir
        self.data_file = data_file
        with open(label_conversion_file) as json_file:
            self.conversion_dict = json.load(json_file)
        self.batch = batch_size
        self.train_techs = train_techs
        self.OOD_techs = OOD_techs
        self.test_size = test_size
        self.val_size = validation_size
        self.min_celltypes = min_celltypes
        self.cpus = num_workers
        self.name = name
        self.verbose = verbose

    def prepare_data(self):
        h5ad = sc.read_h5ad(self.data_dir + self.data_file)
        self.tech_features = h5ad.obs["tech"]
        self.labels = h5ad.obs["celltype"]
        self.data = sparse.csr_matrix(h5ad.X)

    def setup(self, stage):
        # Data setup train
        if self.verbose == "True":
            print('\n')
            print('--- Data characteristics ---')
            print('Total size data', self.data.shape)
        train_ratio = 1 - (self.val_size + self.test_size)
        idx_ID = [t in self.train_techs for t in self.tech_features]
        

        # ID dataset: split in train-val-test
        labels_train, data_train = self.labels[idx_ID], self.data[idx_ID,:] # Note currently not 60 - 20 - 20. Is this a problem?
        data_trainfilter, labels_trainfilter = self.filter_counts_h5ad(
            data_train, labels_train, self.min_celltypes
        )

        X_train, X_val_test, y_train, y_val_test = train_test_split(
        data_trainfilter,
        labels_trainfilter,
        test_size=1-train_ratio,
        stratify=labels_trainfilter,
        random_state=0,
        )

        X_val, X_test, y_val, y_test = train_test_split(
        X_val_test,
        y_val_test,
        test_size=self.test_size/(self.val_size + self.test_size),
        stratify=y_val_test,
        random_state=0,
        )
        
        labels_train = [self.conversion_dict[i] for i in y_train]
        self.data_train = PancreasDataset(torch.from_numpy(X_train.todense()), torch.from_numpy(np.array(labels_train))) 
        print('Cell type classes', np.unique(labels_train))


        labels_val = [self.conversion_dict[i] for i in y_val]
        self.data_val = PancreasDataset(torch.from_numpy(X_val.todense()), torch.from_numpy(np.array(labels_val)))

        labels_test_conv = [self.conversion_dict[i] for i in y_test]
        data_test = torch.from_numpy(X_test.todense())
        labels_test = torch.from_numpy(np.array(labels_test_conv))
        if self.verbose == "True":
            print('Size training data', X_train.shape)
            print('Size validation data', X_val.shape)
            print('Size test data', X_test.shape)

        # Data setup OOD
        idx_OOD = np.array([t in self.OOD_techs for t in self.tech_features])

        labels_OOD, data_OOD= self.labels[idx_OOD], self.data[idx_OOD,:]
        data_OODfilter, labels_OODfilter = self.filter_counts_h5ad(
            data_OOD, labels_OOD, self.min_celltypes
        )
        labels_OODconverted = [self.conversion_dict[i] for i in labels_OODfilter]
        data_OOD = torch.from_numpy(data_OODfilter.todense())
        labels_OOD = torch.from_numpy(np.array(labels_OODconverted))

        if self.verbose == "True":
            print('Size OOD data', data_OODfilter.shape)

        # Setup test data
        data_test_total = torch.cat((data_test, data_OOD),0)
        labels_test_total = torch.cat((labels_test, labels_OOD),0)

        ## indicators OOD dataset
        OOD_ind = pd.DataFrame([1] *data_test.size(dim=0) + [-1]*data_OOD.size(dim=0)) 
        if not os.path.exists(self.data_dir + 'OOD_ind_pancreas'+ '_dataset_' + str(self.name) + '.csv'):
            OOD_ind.to_csv(self.data_dir + 'OOD_ind_pancreas'+ '_dataset_' + str(self.name) + '.csv')

        ## indicator OOD celltypes
        OOD_ind = pd.DataFrame([-1  if i not in np.unique(labels_train) else 1 for i in labels_test_total ])
        OOD_celltypes = [i.numpy() for i in labels_test_total if i.numpy() not in np.unique(labels_train)]
        if OOD_celltypes == []:
            OOD_celltypes = pd.DataFrame([None])
        else:
            OOD_celltypes = pd.DataFrame(OOD_celltypes)
            if self.verbose == "True":
                print('OOD celltypes', np.unique(OOD_celltypes))
                print('percentage OOD celltypes', len(OOD_celltypes)/labels_test_total.size(dim=0))

        #if not os.path.exists(self.data_dir + 'OOD_ind_pancreas'+ '_celltypes_' + str(self.name) + '.csv'):
        OOD_celltypes.to_csv(self.data_dir + 'OOD_ind_pancreas'+ '_celltypes_' + str(self.name) + '.csv')
        if self.verbose == "True":
            print('Size total test data', data_test_total.size())
            print ('\n')
        self.data_test = PancreasDataset(data_test_total, labels_test_total )
       

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch, num_workers = self.cpus)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch, num_workers = self.cpus)

    def test_dataloader(self):
        test_loader = DataLoader(self.data_test, batch_size=self.batch, num_workers = self.cpus)
        return [test_loader]
    
    def predict_dataloader(self):
        test_loader = DataLoader(self.data_test, batch_size=self.batch, num_workers = self.cpus)
        return [test_loader]

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
        h5ad = sc.read_h5ad(self.data_dir + self.data_file)
        self.tech_features = h5ad.obs["tech"]
        self.labels = h5ad.obs["celltype"]
        self.data = sparse.csr_matrix(h5ad.X)
        data_filter, labels_filter = self.filter_counts_h5ad(
            self.data, self.labels, self.min_celltypes
        )
        return len(np.unique(labels_filter))
    
    
    
