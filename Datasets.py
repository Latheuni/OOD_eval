# Datasets
import os
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pytorch_lightning as L
import json


## Lightning
class BaseDataset(Dataset):  # Input for the LightningDataModule
    """Input format needed for Lightning DataModules"""

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def n_classes(self):
        return len(np.unique(self.labels))

    def __getitem__(self, idx):
        return self.data[idx, :], self.labels[idx]

    def get_data(self):
        return self.data

    def get_labels(self):
        return self.labels


## DataModules
class LitLungDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir,
        data_file,
        label_conversion_file,
        scenario,
        batch_size,
        val_size,
        test_size,
        num_workers,
        name,
        min_celltypes=10,
        verbose="True",
        integrated = "False",

    ):
        """Lightning DataModule for the Lung dataset

        Parameters
        ----------
        data_dir : str
            path to data directory
        data_file : str
            filename of h5ad datafile
        label_conversion_file : str
            filename and dir where conversion dictionary of labels is saved
        scenario : str
            train/test split scenario
            options: patient_1, patient_2, patient_3, patient_4, patient_5, patient_6, protocol, tissue
        batch_size : int
        val_size : float
            fraction of training data used for validation
        test_size : float
            fraction of total data used for testing
        num_workers : int
            number of available cpu cores
        name : str
            name of the analysis
        min_celltypes : int, optional
            minimum amount of observations that need to be present to prevent filtering for every cell type
        verbose : str, optional
            by default "True"
        """
        super().__init__()
        self.data_dir = data_dir
        self.data_file = data_file
        self.verbose = verbose
        self.scenario = scenario
        self.val_size = val_size
        self.test_size = test_size
        self.min_celltypes = min_celltypes
        self.name = name
        self.batch = batch_size
        self.cpus = num_workers
        with open(label_conversion_file) as json_file:
            self.conversion_dict = json.load(json_file)
        self.integrated = integrated

    def filter_counts_h5ad(self, data, labels, n):
        """Filter celltypes with less than n instances
        Parameters
        ----------
        data : annData
        labels : Series
        n : int
            celltypes with less than n occurences will be filtered out
        """
        if isinstance(labels, pd.DataFrame):
            s = labels["cell_type"].value_counts() < n
            celltype_to_filter = s.index[s].to_list()
            idx_keep = [l not in celltype_to_filter for l in labels["cell_type"]]
            return (data[idx_keep, :], labels.iloc[idx_keep, :])
        elif isinstance(labels, pd.Series):
            s = labels.value_counts() < n
            celltype_to_filter = s.index[s].to_list()
            idx_keep = [l not in celltype_to_filter for l in labels.values]
            return (data[idx_keep, :], labels.iloc[idx_keep])
        else:
            print('Error with the type of labels, which is:', type(labels)) 

    def prepare_data(self):
        """Read in the Data"""
        h5ad = sc.read_h5ad(self.data_dir + self.data_file)
        if self.integrated == "True":
            print('Data integration in process')
            sc.pp.highly_variable_genes(h5ad)
            scvi.model.SCVI.setup_anndata(h5ad, layer="counts", batch_key="batch")
            model = scvi.model.SCVI(h5ad, n_layers=2, n_latent=30, gene_likelihood="nb")
            model.train(max_epochs=400)
            SCVI_LATENT_KEY = "X_scVI"
            h5ad.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()
            if SCVI_LATENT_KEY in h5ad.obsm:
                self.data =sparse.csr_matrix(h5ad.obsm[SCVI_LATENT_KEY])
                if self.verbose == "True":
                    print('Look into integrated data', self.data.todense().shape)

            else:
                raise ValueError('Something went wrong with integration, no X_scVI key in anndata object')
        else:
            self.data = sparse.csr_matrix(h5ad.X)
        self.obs = h5ad.obs

    def pick_scenario(self):
        train_ratio = 1 - (self.val_size + self.test_size)
        if self.scenario.startswith("patient"):
            # Scenario: split acros patients of the 10X_Transplant data (test one one patient, train on the other 5)
            obs_patient = self.obs.iloc[self.obs["dataset"].values == "10x_Transplant",:]
            data_patient = self.data[
                self.obs["dataset"].values == "10x_Transplant",
            ]

            if not hasattr(self, "patients_ID"):
                self.patients_ID = np.unique(obs_patient["batch"])

            current_sc = int(self.scenario.split("_")[1]) -1

            # Read in OOD data
            X_OOD = data_patient[
                obs_patient["batch"].values == self.patients_ID[current_sc]
            ]
            y_OOD = obs_patient.iloc[
                obs_patient["batch"].values == self.patients_ID[current_sc],:
            ]["cell_type"]

            # Read in train_val(_test) data (= ID data) 
            X_train_val = data_patient[
                obs_patient["batch"].values != self.patients_ID[current_sc]
            ]
            y_train_val = obs_patient.iloc[
                obs_patient["batch"].values != self.patients_ID[current_sc],:
            ]["cell_type"]

            # filter all data
            X_train_val,  y_train_val = self.filter_counts_h5ad(
                 X_train_val,  y_train_val, self.min_celltypes
            )
            X_OOD,  y_OOD = self.filter_counts_h5ad(
                 X_OOD, y_OOD, self.min_celltypes
            )
            if self.verbose:
                print(' \n')
                print('Distribution labels train-val', y_train_val.value_counts())
                print('Distribution labels OOD', y_OOD.value_counts())

            # Split "ID" data
            X_train, X_val_test, y_train_, y_val_test = train_test_split(
                X_train_val,
                y_train_val,
                test_size=1-train_ratio,
                stratify=y_train_val.values,
                random_state=0,
            )
            if self.verbose:
                print('Distribution labels val-test', y_val_test.value_counts())
                print(' \n')

            X_val, X_test, y_val_, y_test = train_test_split(
                X_val_test,
                y_val_test,
                test_size=self.test_size / (self.val_size + self.test_size),
                stratify=y_val_test.values,
                random_state=0,
            )

            # Convert train and val to torch plus convert str labels to int
            y_train = [self.conversion_dict[i] for i in y_train_.values]
            y_val = [self.conversion_dict[i] for i in y_val_.values]
            X_train = torch.from_numpy(X_train.todense())
            X_val = torch.from_numpy(X_val.todense())
            y_train = torch.from_numpy(np.array(y_train))
            y_val  = torch.from_numpy(np.array(y_val))

            # Convert test ID and OOD data to dense and concatenate
            data_test = torch.from_numpy(X_test.todense())
            y_test_ = [self.conversion_dict[i] for i in y_test.values]
            labels_test = torch.from_numpy(np.array(y_test_))
            data_OOD = torch.from_numpy(X_OOD.todense())
            self.data_OOD = data_OOD
            y_OOD_ = [self.conversion_dict[i] for i in y_OOD.values]
            labels_OOD = torch.from_numpy(np.array(y_OOD_))
            data_test_total = torch.cat((data_test, data_OOD), 0)
            labels_test_total = torch.cat((labels_test, labels_OOD), 0)

            # Make an OOD indicator file
            OOD_ind = pd.DataFrame(
            [1] * data_test.size(dim=0) + [-1] * data_OOD.size(dim=0)
            )
            OOD_ind.to_csv(
                self.data_dir
                + "OOD_ind_lung"
                + "_dataset_"
                + str(self.name)
                + ".csv"
            )

            return X_train, X_val, data_test_total, y_train, y_val, labels_test_total

        elif self.scenario.startswith("protocol"):
            # Scenario: ID data is Sun study, OOD data is Villani study

            # Read in, filter and split ID data
            obs_train_val = self.obs.iloc[
                self.obs["dataset"].values == "10x_Transplant",:
            ]
            data_train_val = self.data[
                self.obs["dataset"].values == "10x_Transplant",
            ]
            data_filter_train_val, obs_filter_train_val = self.filter_counts_h5ad(
                data_train_val, obs_train_val, self.min_celltypes
            )
            if self.verbose:
                print('Distribution labels train-val', obs_filter_train_val["cell_type"].value_counts())
           
            X_train, X_val_test, y_train_, y_val_test = train_test_split(
                data_filter_train_val,
                obs_filter_train_val["cell_type"],
                test_size= 1 - train_ratio,
                stratify= obs_filter_train_val["cell_type"].values,
                random_state=0,
            )
            if self.verbose:
                print('Distribution labels val-test', y_val_test.value_counts())
            
            X_val, X_test, y_val_, y_test = train_test_split(
                X_val_test,
                y_val_test,
                test_size=self.test_size / (self.val_size + self.test_size),
                stratify=y_val_test.values,
                random_state=0,
            )
            
            # Convert train and val to torch plus convert str labels to int
            y_train = [self.conversion_dict[i] for i in y_train_.values]
            y_val = [self.conversion_dict[i] for i in y_val_.values]
            X_train = torch.from_numpy(X_train.todense())
            X_val = torch.from_numpy(X_val.todense())
            y_train = torch.from_numpy(np.array(y_train))
            y_val  = torch.from_numpy(np.array(y_val))

            # Read in and filter OOD data
            obs_OOD = self.obs.iloc[
                self.obs["dataset"].values == "Dropseq_transplant",:
            ]
            data_OOD = self.data[
                self.obs["dataset"].values == "Dropseq_transplant",
            ]
            X_OOD, y_OOD = self.filter_counts_h5ad(
                data_OOD, obs_OOD, self.min_celltypes
            )
            if self.verbose:
                print('Distribution labels OOD', y_OOD['cell_type'].value_counts())

            # Convert test ID and OOD data to dense and concatenate
            data_test = torch.from_numpy(X_test.todense())
            y_test_ = [self.conversion_dict[i] for i in y_test.values]
            labels_test = torch.from_numpy(np.array(y_test_))
            data_OOD = torch.from_numpy(X_OOD.todense())
            self.data_OOD = data_OOD
            y_OOD_ = [self.conversion_dict[i] for i in y_OOD['cell_type'].values]
            labels_OOD = torch.from_numpy(np.array(y_OOD_))
            data_test_total = torch.cat((data_test, data_OOD), 0)
            labels_test_total = torch.cat((labels_test, labels_OOD), 0)

            # Make an OOD indicator file
            OOD_ind = pd.DataFrame(
            [1] * data_test.size(dim=0) + [-1] * data_OOD.size(dim=0)
            )

            OOD_ind.to_csv(
                self.data_dir
                + "OOD_ind_lung"
                + "_dataset_"
                + str(self.name)
                + ".csv"
            )
            return X_train, X_val, data_test_total, y_train, y_val, labels_test_total

        elif self.scenario.startswith("tissue"):
            # Scenario=:Train/Val data is Sun study, Test data is Oetjen study

            # Read in ID data, filter and split
            obs_train_val = self.obs.iloc[
                self.obs["dataset"].values == "10x_Transplant",:
            ]
            data_train_val = self.data[
                self.obs["dataset"].values == "10x_Transplant",
            ]
            data_filter_train_val, obs_filter_train_val = self.filter_counts_h5ad(
                data_train_val, obs_train_val, self.min_celltypes
            )

            X_train, X_val_test, y_train_, y_val_test = train_test_split(
                data_filter_train_val,
                obs_filter_train_val["cell_type"],
                test_size=1-train_ratio,
                stratify=obs_filter_train_val["cell_type"].values,
                random_state=0,
            )

            X_val, X_test, y_val_, y_test = train_test_split(
                X_val_test,
                y_val_test,
                test_size=self.test_size / (self.val_size + self.test_size),
                stratify=y_val_test.values,
                random_state=0,
            )

            y_train = [self.conversion_dict[i] for i in y_train_.values]
            y_val = [self.conversion_dict[i] for i in y_val_.values]
            X_train = torch.from_numpy(X_train.todense())
            X_val = torch.from_numpy(X_val.todense())
            y_train = torch.from_numpy(np.array(y_train))
            y_val  = torch.from_numpy(np.array(y_val))

            # OOD data is 10X biopsy
            obs_OOD = self.obs.iloc[self.obs["dataset"].values == "10x_Biopsy",:]
            data_OOD = self.data[self.obs["dataset"].values == "10x_Biopsy",]
            X_OOD, y_OOD= self.filter_counts_h5ad(
                data_OOD, obs_OOD["cell_type"], self.min_celltypes
            )

            # Convert test ID and OOD data to dense and concatenate
            data_test = torch.from_numpy(X_test.todense())
            y_test_ = [self.conversion_dict[i] for i in y_test.values]
            labels_test = torch.from_numpy(np.array(y_test_))
            data_OOD = torch.from_numpy(X_OOD.todense())
            self.data_OOD = data_OOD
            y_OOD_ = [self.conversion_dict[i] for i in y_OOD.values]
            labels_OOD = torch.from_numpy(np.array(y_OOD_))
            data_test_total = torch.cat((data_test, data_OOD), 0)
            labels_test_total = torch.cat((labels_test, labels_OOD), 0)

            if self.verbose:
                print(' \n')
                print('Distribution labels train-val', y_train_.value_counts())
                print('Distribution labels OOD', y_OOD.value_counts())
                print('Distribution val-test', y_val_test.value_counts())
                print(' \n')

            # Make an OOD indicator file
            OOD_ind = pd.DataFrame(
            [1] * data_test.size(dim=0) + [-1] * data_OOD.size(dim=0)
            )

            OOD_ind.to_csv(
                self.data_dir
                + "OOD_ind_lung"
                + "_dataset_"
                + str(self.name)
                + ".csv"
            )

            return X_train, X_val, data_test_total, y_train, y_val, labels_test_total

    def check_for_OOD(self, y_train, y_test):
        """Check if OOD labels/celltypes are present in train/test split

        Parameters
        ----------
        y_train
        y_test

        Returns
        -------
        ind: boolean
            Are there OOD labels?
        idx_OOD: list
            boolean list: TRUE/FALSE this label is OOD
        """
        un_test = np.unique(y_test)
        un_train = np.unique(y_train)
        print('un_train', un_train)
        print('un_test', un_test)
        self.un_test = un_test
        self.un_train = un_train

        if sum([t not in un_train for t in un_test]) > 0:
            ind = True
            idx_OOD = pd.DataFrame([-1 if t not in un_train else 1 for t in y_test.numpy()])
        else:
            ind = False
            idx_OOD = pd.DataFrame([None])
        return ind, idx_OOD

    def setup(self,stage):
        """General Data setup"""
        # Data setup train
    
        X_train, X_val, X_test, y_train, y_val, y_test = self.pick_scenario()
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test


        self.data_test = BaseDataset( # Already in dense format since concatenating with OOD data
            X_test,
            y_test,
        )

        self.data_train = BaseDataset(
            X_train,
            y_train,
        )

        self.data_val = BaseDataset(
            X_val,
            y_val,
        )

        OOD_idx, OOD_ind = self.check_for_OOD(y_train, y_test)
        self.OOD_idx = OOD_idx
        self.OOD_ind = OOD_ind
        if self.verbose: 
            print("OOD celltypes?", OOD_idx)
            if OOD_ind.iloc[0,0] != None:
                print("Size OOD data?", sum([1 for i in OOD_ind.iloc[:,0].values if i == -1]))
            print(" \n")
        # if not os.path.exists(
        #     self.data_dir + "OOD_ind_lung" + "_celltype_" + str(self.name) + ".csv"
        # ):
        OOD_ind.to_csv(
            self.data_dir
            + "OOD_ind_lung"
            + "_celltypes_"
            + str(self.name)
            + ".csv"
        )

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch, num_workers=self.cpus, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch, num_workers=self.cpus)

    def test_dataloader(self):
        test_loader = DataLoader(
            self.data_test, batch_size=self.batch, num_workers=self.cpus
        )
        return [test_loader]

    def predict_dataloader(self):
        predict_loader = DataLoader(
            self.data_test, batch_size=self.batch, num_workers=self.cpus
        )
        return [predict_loader]
    def n_classes(self):
        """Returns the number of unqiue label classes"""
        print('n_classes dataloader: ', len(list((self.conversion_dict.keys()))) )
        return len(list((self.conversion_dict.keys())))

    def n_features(self):
        """Return the number of features"""
        h5ad = sc.read_h5ad(self.data_dir + self.data_file)
        if self.integrated == "True":
            sc.pp.highly_variable_genes(h5ad)
            scvi.model.SCVI.setup_anndata(h5ad, layer="counts", batch_key="batch")
            model = scvi.model.SCVI(h5ad, n_layers=2, n_latent=30, gene_likelihood="nb")
            model.train(max_epochs=400)
            SCVI_LATENT_KEY = "X_scVI"
            h5ad.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()
            if SCVI_LATENT_KEY in h5ad.obsm:
               data =sparse.csr_matrix(h5ad.obsm[SCVI_LATENT_KEY])
            else:
                raise ValueError('Something went wrong with integration, no X_scVI key in anndata object')
        else:
             data = sparse.csr_matrix(h5ad.X)

        if self.verbose == "True":
            print('n_features dataloader', data.shape[1] )
        return data.shape[1]

    def display_data_characteristics(self):
        print("\n")
        print("--- Data characteristics ---")
        print("Total size data", self.data.shape)

        print("Size training data", self.X_train.shape)
        print("Size validation data", self.X_val.shape)
        print("size OOD data",self.data_OOD.shape )
        print("Size test data (total)", self.X_test.shape)
        
        print("OOD celltypes?", self.OOD_idx)
        print(self.OOD_ind.iloc[0,0])
        if isinstance(self.OOD_ind.iloc[0,0], int):
            print("Size OOD celltypes?", sum([i == -1 for i in self.OOD_ind.iloc[:,0].values]))
            print('unique celltypes train', self.un_train)
            print('unique celltypes test', self.un_test)

class LitImmuneDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir,
        data_file,
        label_conversion_file,
        scenario,
        batch_size,
        val_size,
        test_size,
        num_workers,
        name,
        min_celltypes=10,
        verbose="True",
        integrated ="False",
    ):
        """Lightning DataModule for the immune dataset

        Parameters
        ----------
        data_dir : str
            path to data directory
        data_file : str
            filename of h5ad datafile
        label_conversion_file : str
            filename and dir where conversion dictionary of labels is saved
        scenario : str
            train/test split scenario
            options: intra_1, intra_2, intra_3, intra_4, patient_1, patient_2, patient_3, patient_4, protocol, tissue
        batch_size : int
        val_size : float
            fraction of training data used for validation
        test_size : float
            for intra scenario= fraction of the data used for testing
        num_workers : int
            number of available cpu cores
        name : _str
            name of the analysis
        min_celltypes : int
            minimum amount of observations that need to be present to prevent filtering for every cell type
        verbose : str, optional
            by default "True"
        """

        super().__init__()
        self.data_dir = data_dir
        self.data_file = data_file
        self.verbose = verbose
        self.scenario = scenario
        self.val_size = val_size
        self.test_size = test_size
        self.min_celltypes = min_celltypes
        self.name = name
        self.batch = batch_size
        self.cpus = num_workers
        with open(label_conversion_file) as json_file:
            self.conversion_dict = json.load(json_file)
        self.integrated = integrated

    def filter_counts_h5ad(self, data, labels, n):
        """Filter celltypes with less than 10 instances
        Parameters
        ----------
        data : annData
        labels : Series
        n : int
            celltypes with less than n occurences will be filtered out
        """
        s = labels["final_annotation"].value_counts() < n
        celltype_to_filter = s.index[s].to_list()
        idx_keep = [l not in celltype_to_filter for l in labels.iloc[:, 0]]

        return (data[idx_keep, :], labels.iloc[idx_keep, :])

    def prepare_data(self):
        """Read in the Data"""
        h5ad = sc.read_h5ad(self.data_dir + self.data_file)
        if self.integrated == "True":
            print('Data integration in process')
            sc.pp.highly_variable_genes(h5ad, subset = True)
            sc.pp.pca(h5ad)
            sc.external.pp.scanorama_integrate(h5ad, key = "batch")
            print('Integrated?', 'X_scanorama' in h5ad.obsm)
            if 'X_scanorama' in h5ad.obsm:
                print(h5ad.obsm["X_scanorama"])
                self.data =sparse.csr_matrix(h5ad.obsm["X_scanorama"])
                if self.verbose:
                    print('Look into integrated data', self.data.todense().shape)

            else:
                raise ValueError('Something went wrong with integration, no X_scanorama key in anndata object')
        else:
            self.data = sparse.csr_matrix(h5ad.X)
        self.obs = h5ad.obs

    def pick_scenario(self):
        """Decide train-test split based on setup senario specified in tself.scenario

        Returns
        -------
        X_train, X_val, X_test, y_train, y_val, y_test
        """
        train_ratio = 1 - (self.val_size + self.test_size)
        if self.scenario.startswith("patient"):
            # Scenario: split acros patients of the Sun study (test one one patient, train on the other three)
            obs_patient = self.obs.iloc[self.obs["study"].values == "Sun",:]
            data_patient = self.data[self.obs["study"].values == "Sun",:]

            if not hasattr(self, "patients_ID"):
                self.patients_ID = np.unique(obs_patient["batch"])

            current_sc = int(self.scenario.split("_")[1])-1

            # Read in OOD data
            X_OOD = data_patient[
                obs_patient["batch"].values == self.patients_ID[current_sc],
            ]

            y_OOD = obs_patient.iloc[
                obs_patient["batch"].values == self.patients_ID[current_sc],:
            ]

            # Read in train_val(_test) data (= ID data)
            X_train_val = data_patient[
                obs_patient["batch"].values != self.patients_ID[current_sc],
            ]
            y_train_val = obs_patient.iloc[
                obs_patient["batch"].values != self.patients_ID[current_sc],:
            ]
            
            # filter all data
            X_train_val,  y_train_val = self.filter_counts_h5ad(
                 X_train_val,  y_train_val, self.min_celltypes
            )
            X_OOD,  y_OOD = self.filter_counts_h5ad(
                 X_OOD, y_OOD, self.min_celltypes
            )

            # Split "ID" data
            X_train, X_val_test, y_train_, y_val_test = train_test_split(
                X_train_val,
                y_train_val["final_annotation"],
                test_size=1-train_ratio,
                stratify=y_train_val["final_annotation"].values,
                random_state=0,
            )

            X_val, X_test, y_val_, y_test = train_test_split(
                X_val_test,
                y_val_test,
                test_size=self.test_size / (self.val_size + self.test_size),
                stratify=y_val_test.values,
                random_state=0,
            )
            # Convert train and val to torch plus convert str labels to int
            y_train = [self.conversion_dict[i] for i in y_train_.values]
            y_val = [self.conversion_dict[i] for i in y_val_.values]
            X_train = torch.from_numpy(X_train.todense())
            X_val = torch.from_numpy(X_val.todense())
            y_train = torch.from_numpy(np.array(y_train))
            y_val  = torch.from_numpy(np.array(y_val))

            # Convert test ID and OOD data to dense and concatenate
            data_test = torch.from_numpy(X_test.todense())
            y_test_ = [self.conversion_dict[i] for i in y_test.values]
            labels_test = torch.from_numpy(np.array(y_test_))
            data_OOD = torch.from_numpy(X_OOD.todense())
            self.data_OOD = data_OOD
            y_OOD_ = [self.conversion_dict[i] for i in y_OOD["final_annotation"].values]
            labels_OOD = torch.from_numpy(np.array(y_OOD_))
            data_test_total = torch.cat((data_test, data_OOD), 0)
            labels_test_total = torch.cat((labels_test, labels_OOD), 0)
            

            if self.verbose :
                print(' \n')
                print('Distribution labels train-val', y_train_val["final_annotation"].value_counts())
                print('Distribution labels OOD', y_OOD["final_annotation"].value_counts())
                print('Distribution val-test', y_val_test.value_counts())
                print(' \n')

            # Make an OOD indicator file
            OOD_ind = pd.DataFrame(
            [1] * data_test.size(dim=0) + [-1] * data_OOD.size(dim=0)
            )
            OOD_ind.to_csv(
                self.data_dir
                + "OOD_ind_immune"
                + "_dataset_"
                + str(self.name)
                + ".csv"
            )

            return X_train, X_val, data_test_total, y_train, y_val, labels_test_total

        elif self.scenario.startswith("protocol"):
            # Scenario: ID data is Sun study, OOD data is Villani study

            # Read in ID data, filter and split
            obs_train_val = self.obs.iloc[self.obs["study"].values == "Sun",]
            data_train_val = self.data[self.obs["study"].values == "Sun",]
            data_filter_train_val, obs_filter_train_val = self.filter_counts_h5ad(
                data_train_val, obs_train_val, self.min_celltypes
            )

            X_train, X_val_test, y_train_, y_val_test = train_test_split(
                data_filter_train_val,
                obs_filter_train_val["final_annotation"],
                test_size=1-train_ratio,
                stratify=obs_filter_train_val["final_annotation"].values,
                random_state=0,
            )
            
            X_val, X_test, y_val_, y_test = train_test_split(
                X_val_test,
                y_val_test,
                test_size=self.test_size / (self.val_size + self.test_size),
                stratify=y_val_test.values,
                random_state=0,
            )
            # Convert train and val to torch plus convert str labels to int
            y_train = [self.conversion_dict[i] for i in y_train_.values]
            y_val = [self.conversion_dict[i] for i in y_val_.values]
            X_train = torch.from_numpy(X_train.todense())
            X_val = torch.from_numpy(X_val.todense())
            y_train = torch.from_numpy(np.array(y_train))
            y_val  = torch.from_numpy(np.array(y_val))

            # Read in OOD data and filter
            obs_OOD = self.obs.iloc[self.obs["study"].values == "Villani",]
            data_OOD = self.data[self.obs["study"].values == "Villani",]
            X_OOD, y_OOD = self.filter_counts_h5ad(
                data_OOD, obs_OOD, self.min_celltypes
            )

            # Convert test ID and OOD data to torch and concatenate
            data_test = torch.from_numpy(X_test.todense())
            y_test_ = [self.conversion_dict[i] for i in y_test.values]
            labels_test = torch.from_numpy(np.array(y_test_))
            data_OOD = torch.from_numpy(X_OOD.todense())
            self.data_OOD = data_OOD
            y_OOD_ = [self.conversion_dict[i] for i in y_OOD['final_annotation'].values]
            labels_OOD = torch.from_numpy(np.array(y_OOD_))
            data_test_total = torch.cat((data_test, data_OOD), 0)
            labels_test_total = torch.cat((labels_test, labels_OOD), 0)
            
            if self.verbose:
                print(' \n')
                print('Distribution labels train-val', y_train_.value_counts())
                print('Distribution labels OOD', y_OOD['final_annotation'].value_counts())
                print('Distribution val-test', y_val_test.value_counts())
                print(' \n')

            # Make an OOD indicator file
            OOD_ind = pd.DataFrame(
            [1] * data_test.size(dim=0) + [-1] * data_OOD.size(dim=0)
            )
            if not os.path.exists(
            self.data_dir + "OOD_ind_immune" + "_dataset_" + str(self.name) + ".csv"
            ):
                OOD_ind.to_csv(
                    self.data_dir
                    + "OOD_ind_immune"
                    + "_dataset_"
                    + str(self.name)
                    + ".csv"
                )

            return X_train, X_val, data_test_total, y_train, y_val, labels_test_total

        elif self.scenario.startswith("tissue"):
            # Scenario=:ID data is Sun study, OOD data is Oetjen study

            # Read in ID data, filter and split
            obs_train_val = self.obs.iloc[self.obs["study"].values == "Sun",:]
            data_train_val = self.data[self.obs["study"].values == "Sun",:]
            data_filter_train_val, obs_filter_train_val = self.filter_counts_h5ad(
                data_train_val, obs_train_val, self.min_celltypes
            )

            X_train, X_val_test, y_train_, y_val_test = train_test_split(
                data_filter_train_val,
                obs_filter_train_val["final_annotation"],
                test_size=1-train_ratio,
                stratify=obs_filter_train_val["final_annotation"].values,
                random_state=0,
            )

            X_val, X_test, y_val_, y_test = train_test_split(
                X_val_test,
                y_val_test,
                test_size=self.test_size / (self.val_size + self.test_size),
                stratify=y_val_test.values,
                random_state=0,
            )
            y_train = [self.conversion_dict[i] for i in y_train_.values]
            y_val = [self.conversion_dict[i] for i in y_val_.values]
            X_train = torch.from_numpy(X_train.todense())
            X_val = torch.from_numpy(X_val.todense())
            y_train = torch.from_numpy(np.array(y_train))
            y_val  = torch.from_numpy(np.array(y_val))

            # OOD data is Oetjen study(Bone_marrow)
            obs_OOD = self.obs.iloc[self.obs["study"].values == "Oetjen",:]
            data_OOD = self.data[self.obs["study"].values == "Oetjen",:]
            X_OOD, y_OOD = self.filter_counts_h5ad(
                data_OOD, obs_OOD, self.min_celltypes
            )
            # Convert test ID and OOD data to dense and concatenate
            data_test = torch.from_numpy(X_test.todense())
            y_test_ = [self.conversion_dict[i] for i in y_test.values]
            labels_test = torch.from_numpy(np.array(y_test_))
            
            data_OOD = torch.from_numpy(X_OOD.todense())
            self.data_OOD = data_OOD
            y_OOD_ = [self.conversion_dict[i] for i in y_OOD['final_annotation'].values]
            labels_OOD = torch.from_numpy(np.array(y_OOD_))
            data_test_total = torch.cat((data_test, data_OOD), 0)
            labels_test_total = torch.cat((labels_test, labels_OOD), 0)
            if self.verbose:
                print(' \n')
                print('Distribution labels train-val', y_train_.value_counts())
                print('Distribution labels OOD', y_OOD['final_annotation'].value_counts())
                print('Distribution val-test', y_val_test.value_counts())
                print(' \n')

            # Make an OOD indicator file
            OOD_ind = pd.DataFrame(
            [1] * data_test.size(dim=0) + [-1] * data_OOD.size(dim=0)
            )
            self.OOD_ind = OOD_ind
            OOD_ind.to_csv(
                self.data_dir
                + "OOD_ind_immune"
                + "_dataset_"
                + str(self.name)
                + ".csv"
            )
            return X_train, X_val, data_test_total, y_train, y_val, labels_test_total

    def check_for_OOD(self, y_train, y_test):
        """Check if OOD labels are present in train/test split

        Parameters
        ----------
        y_train
        y_test

        Returns
        -------
        ind: boolean
            Are there OOD labels?
        idx_OOD: list
            boolean list: TRUE/FALSE this label is OOD
        """
        y_test = y_test.detach().numpy()
        y_train = y_train.detach().numpy()
        
        un_test = np.unique(y_test)
        un_train = np.unique(y_train)
        self.un_test = un_test
        self.un_train = un_train

        if sum([t not in un_train for t in un_test]) > 0:
            ind = True
            idx_OOD = pd.DataFrame([-1 if t not in un_train else 1 for t in y_test])
        else:
            ind = False
            idx_OOD = pd.DataFrame([None])
        if self.verbose:
            print('n OOD', sum(idx_OOD))
        return ind, idx_OOD

    def setup(self,stage):
        """General Data setup"""
        # Data setup train
        X_train, X_val, X_test, y_train, y_val, y_test = self.pick_scenario()
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        # Convert str labels to int

        self.data_test = BaseDataset(
            X_test,
            y_test,
        )

        self.data_train = BaseDataset(
            X_train,
            y_train,
        )

        self.data_val = BaseDataset(
            X_val,
            y_val,
        )

        OOD_idx, OOD_ind = self.check_for_OOD(y_train, y_test)
        self.OOD_idx = OOD_idx
        self.OOD_ind = OOD_ind
        if self.verbose:
            print("OOD celltypes?", OOD_idx)
            if OOD_ind.iloc[0,0] != None:
                print("Size OOD data?", sum(OOD_ind.iloc[:,0].values))
            print(" \n")
        # if not os.path.exists(
        #     self.data_dir + "OOD_ind_immune" + "_celltypes_" + str(self.name) + ".csv"
        # ):
        OOD_ind.to_csv(
            self.data_dir
            + "OOD_ind_immune"
            + "_celltypes_"
            + str(self.name)
            + ".csv"
        )

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch, num_workers=self.cpus, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch, num_workers=self.cpus)

    def test_dataloader(self):
        test_loader = DataLoader(
            self.data_test, batch_size=self.batch, num_workers=self.cpus
        )
        return [test_loader]

    def predict_dataloader(self):
        predict_loader = DataLoader(
            self.data_test, batch_size=self.batch, num_workers=self.cpus
        )
        return [predict_loader]

    def make_UMAP_manifold(self, data, labels, techs, filename):
        # Read in data
        adata = ad.AnnData(data.numpy())
        adata.obs["celltype"] = labels
        adata.obs["tech"] = techs

        # Compute UMAP
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)

        adata.write_h5ad(self.data_dir + filename)
        self.anndata_umap = adata
    def n_classes(self):
        """Returns the number of unqiue label classes"""
        return len(list((self.conversion_dict.keys())))
    
    def n_features(self):
        """Return the number of features"""
        h5ad = sc.read_h5ad(self.data_dir + self.data_file)
        if self.integrated == "True":
            sc.pp.highly_variable_genes(h5ad, subset = True)
            sc.pp.pca(h5ad)
            sc.external.pp.scanorama_integrate(h5ad, key = "batch")
            print('Integrated?', 'X_scanorama' in h5ad.obsm)
            if 'X_scanorama' in h5ad.obsm:
                print(h5ad.obsm["X_scanorama"])
                data =sparse.csr_matrix(h5ad.obsm["X_scanorama"])

            else:
                raise ValueError('Something went wrong with integration, no X_scanorama key in anndata object')
        else:
            data = sparse.csr_matrix(h5ad.X)



        if self.verbose:
            print('n_features dataloader', data.shape[1] )
        return data.shape[1]
    def display_data_characteristics(self):
        print("\n")
        print("--- Data characteristics ---")
        print("Total size data", self.data.shape)

        print("Size training data", self.X_train.shape)
        print("Size validation data", self.X_val.shape)
        print("size OOD data",self.data_OOD.shape )
        print("Size test data (total)", self.X_test.shape)
        
        print("OOD celltypes?", self.OOD_idx)
        print(self.OOD_ind.iloc[0,0])
        if isinstance(self.OOD_ind.iloc[0,0], int):
            print("Size OOD celltypes?", sum([i == -1 for i in self.OOD_ind.iloc[:,0].values]))
            print('unique celltypes train', self.un_train)
            print('unique celltypes test', self.un_test)
        

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
        val_size,
        min_celltypes,
        num_workers,
        name,
        verbose="True",
    ):
        """Lightning DataModule for the pancreas dataset

        Parameters
        ----------
        data_dir : str
            path to data directory
        data_file : str
            filename of h5ad datafile
        label_conversion_file : str
            filename and dir where conversion dictionary of labels is saved
        batch_size : int
        train_techs : list
            sequencing technologies used for training
        OOD_techs : list
            sequencing technologies used for OOD/test
        test_size : float
            fraction of ID data (training_techs) also used for testing
        val : float
            fraction of ID data (training_techs) also used for validation
        min_celltypes : int
            minimum amount of observations that need to be present to prevent filtering for every cell type
        num_workers : int
            number of available cpu cores
        name : _str
            name of the analysis
        verbose : str, optional
            by default "True"
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
        self.val_size = val_size
        self.min_celltypes = min_celltypes
        self.cpus = num_workers
        self.name = name
        self.verbose = verbose

    def prepare_data(self):
        h5ad = sc.read_h5ad(self.data_dir + self.data_file)
        self.tech_features = h5ad.obs["tech"]
        self.labels = h5ad.obs[["celltype", "tech"]]  # numpy matrix
        self.data = sparse.csr_matrix(h5ad.X)

    def setup(self, stage):
        # Data setup train
        if self.verbose:
            print("\n")
            print("--- Data characteristics ---")
            print("Total size data", self.data.shape)
        train_ratio = 1 - (self.val_size + self.test_size)
        idx_ID = [t in self.train_techs for t in self.tech_features]

        # ID dataset: split in train-val-test
        labels_train, data_train = (
            self.labels.iloc[idx_ID, :],
            self.data[idx_ID, :],
        )  
        data_trainfilter, labels_trainfilter = self.filter_counts_h5ad(
            data_train, labels_train, self.min_celltypes
        )

        X_train, X_val_test, y_train, y_val_test = train_test_split(
            data_trainfilter,
            labels_trainfilter,
            test_size=1 - train_ratio,
            stratify=labels_trainfilter.iloc[:, 0],
            random_state=0,
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_val_test,
            y_val_test,
            test_size=self.test_size / (self.val_size + self.test_size),
            stratify=y_val_test.iloc[:, 0],
            random_state=0,
        )

        labels_train = [self.conversion_dict[i] for i in y_train.iloc[:, 0]]
        self.tech_train = y_train.iloc[:, 1]
        self.data_train = BaseDataset(
            torch.from_numpy(X_train.todense()),
            torch.from_numpy(np.array(labels_train)),
        )
        if self.verbose:
            print("Cell type classes", np.unique(labels_train))

        labels_val = [self.conversion_dict[i] for i in y_val.iloc[:, 0]]
        self.tech_val = y_val.iloc[:, 1]
        self.data_val = BaseDataset(
            torch.from_numpy(X_val.todense()), torch.from_numpy(np.array(labels_val))
        )

        labels_test_conv = [self.conversion_dict[i] for i in y_test.iloc[:, 0]]
        data_test = torch.from_numpy(X_test.todense())
        labels_test = torch.from_numpy(np.array(labels_test_conv))
        if self.verbose:
            print("Size training data", X_train.shape)
            print("Size validation data", X_val.shape)
            print("Size test data", X_test.shape)

        # Data setup OOD
        idx_OOD = np.array([t in self.OOD_techs for t in self.tech_features])

        labels_OOD, data_OOD = self.labels.iloc[idx_OOD, :], self.data[idx_OOD, :]

        data_OODfilter, labels_OODfilter = self.filter_counts_h5ad(
            data_OOD, labels_OOD, self.min_celltypes
        )
        labels_OODconverted = [
            self.conversion_dict[i] for i in labels_OODfilter.iloc[:, 0]
        ]
        data_OOD = torch.from_numpy(data_OODfilter.todense())
        labels_OOD = torch.from_numpy(np.array(labels_OODconverted))

        if self.verbose:
            print("Size OOD data", data_OODfilter.shape)

        # Setup test data
        data_test_total = torch.cat((data_test, data_OOD), 0)
        labels_test_total = torch.cat((labels_test, labels_OOD), 0)
        self.tech_test = np.concatenate(
            (y_test.iloc[:, 1], labels_OODfilter.iloc[:, 1])
        )

        ## indicators OOD dataset
        OOD_ind = pd.DataFrame(
            [1] * data_test.size(dim=0) + [-1] * data_OOD.size(dim=0)
        )
        if not os.path.exists(
            self.data_dir + "OOD_ind_pancreas" + "_dataset_" + str(self.name) + ".csv"
        ):
            OOD_ind.to_csv(
                self.data_dir
                + "OOD_ind_pancreas"
                + "_dataset_"
                + str(self.name)
                + ".csv"
            )

        ## indicator OOD celltypes
        OOD_ind = pd.DataFrame(
            [-1 if i not in np.unique(labels_train) else 1 for i in labels_test_total]
        )
        OOD_celltypes = [
            i.numpy()
            for i in labels_test_total
            if i.numpy() not in np.unique(labels_train)
        ]
        OOD_celltypes_ind = [
            -1 if i.numpy() not in np.unique(labels_train) else 1
            for i in labels_test_total
        ]
        if OOD_celltypes == []:
            if self.verbose :
                print("No OOD celtypes")
            OOD_celltypes_ind = pd.DataFrame([None])
        else:
            OOD_celltypes_ind = pd.DataFrame(OOD_celltypes_ind)
            if self.verbose:
                print("OOD celltypes", np.unique(OOD_celltypes))
                print(
                    "percentage OOD celltypes",
                    len(OOD_celltypes) / labels_test_total.size(dim=0),
                )

        # if not os.path.exists(self.data_dir + 'OOD_ind_pancreas'+ '_celltypes_' + str(self.name) + '.csv'):
        OOD_celltypes_ind.to_csv(
            self.data_dir + "OOD_ind_pancreas" + "_celltypes_" + str(self.name) + ".csv"
        )
        if self.verbose:
            print(
                "percentage OOD dataset",
                len(labels_OOD) / labels_test_total.size(dim=0),
            )
            print("Size total test data", data_test_total.size())
            print("\n")
        self.data_test = BaseDataset(data_test_total, labels_test_total)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch, num_workers=self.cpus, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch, num_workers=self.cpus)

    def test_dataloader(self):
        test_loader = DataLoader(
            self.data_test, batch_size=self.batch, num_workers=self.cpus
        )
        return [test_loader]

    def predict_dataloader(self):
        test_loader = DataLoader(
            self.data_test, batch_size=self.batch, num_workers=self.cpus
        )
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
        s = labels["celltype"].value_counts() < n
        celltype_to_filter = s.index[s].to_list()
        idx_keep = [l not in celltype_to_filter for l in labels.iloc[:, 0]]

        return (data[idx_keep, :], labels.iloc[idx_keep, :])

    def n_classes(self):
        """Returns the number of unqiue label classes"""
        h5ad = sc.read_h5ad(self.data_dir + self.data_file)
        self.tech_features = h5ad.obs["tech"]
        self.labels = h5ad.obs[["celltype", "tech"]]
        self.data = sparse.csr_matrix(h5ad.X)
        data_filter, labels_filter = self.filter_counts_h5ad(
            self.data, self.labels, self.min_celltypes
        )
        return len(np.unique(labels_filter.iloc[:, 0]))

    def n_features(self):
        """Return the number od features"""
        h5ad = sc.read_h5ad(self.data_dir + self.data_file)
        data = sparse.csr_matrix(h5ad.X)
        return data.shape[1]

    def make_UMAP_manifold(self, data, labels, techs, filename):
        # Read in data
        adata = ad.AnnData(data.numpy())
        adata.obs["celltype"] = labels
        adata.obs["tech"] = techs

        # Compute UMAP
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)

        adata.write_h5ad(self.data_dir + filename)
        self.anndata_umap = adata

    def return_data_UMAP(self):
        return self.tech_test, self.data_test.get_data(), self.data_test.get_labels()

    def plot_UMAP_colour(self, filename, colours_ind):
        if not os.path.exists(self.data_dir + filename):
            fig = sc.pl.umap(
                self.anndata_umap, color=colours_ind, palette="tab20", return_fig=True
            )
            fig.savefig(self.data_dir + filename)
        else:
            if self.verbose:
                print("UMAP already exists")

    def plot_UMAP(self, filename):
        if not os.path.exists(self.data_dir + filename):
            fig = sc.pl.umap(
                self.anndata_umap, color=["tech"], palette="tab20", return_fig=True
            )
            fig.savefig(self.data_dir + filename)
        else:
            if self.verbose:
                print("UMAP already exists")

class LitCOPDDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir,
        data_file,
        label_conversion_file, 
        scenario,
        batch_size,
        val_size, 
        test_size, 
        num_workers,
        name,
        min_celltypes = 10,
        min_cells_patient = 500,
        verbose= True,
    ):
        """Lightning DataModule for the COPD dataset

        Parameters
        ----------
        data_dir : str
            path to data directory
        data_file : str
            filename of h5ad datafile
        label_conversion_file : str
            filename and dir where conversion dictionary of labels is saved
        scenario : str
            train/test split scenario
            options: patient_1 - patient_20, COPD, IPF an ds_2, s_5 and s_10
        batch_size : int
        val_size : float
            fraction of training data used for validation
        test_size: float
            fraction of total data used for testing
        num_workers : int
            number of available cpu cores
        name : _str
            name of the analysis
        min_celltypes : int
            minimum amount of observations that need to be present for every cell type
        min_cells_patient: int
            minimum amount of observations that need to be present per patient
        verbose : str, optional
            by default "True"
        """
        super().__init__()
        self.data_dir = data_dir
        self.data_file = data_file
        self.verbose = verbose
        self.scenario = scenario
        self.val_size = val_size
        self.test_size = test_size
        self.min_celltypes = min_celltypes
        self.min_cells_patient = min_cells_patient
        self.name = name
        self.batch = batch_size
        self.cpus = num_workers
        with open(label_conversion_file) as json_file:
            self.conversion_dict = json.load(json_file)
        

    def default(obj):
        if type(obj).__module__ == np.__name__:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj.item()
        elif type(obj) == pd.core.frame.DataFrame:
            return obj.to_json()
        elif type(obj) == pd.core.series.Series:
            return obj.to_json()
            
    def save_dict_to_json(self, d_, name):
        with open(name, "w") as f1:
            json.dump(d_, f1, default=self.default)

    def prepare_data(self):
        # Read in the data and normalize
        h5ad = sc.read_h5ad(self.data_dir + self.data_file)
        h5ad.var = pd.read_csv('/data/gent/438/vsc43883/Data/COPD/GSE136831_AllCells.GeneIDs.txt', sep = "\t")
        sc.pp.normalize_total(h5ad, target_sum = 10000)
        sc.pp.log1p(h5ad)

        # Save fixed order of patients to file
        if self.scenario.startswith("patient"):
            #if not os.path.exists( self.data_dir + "COPD_control_patients.json"):
            patients = np.unique(h5ad.obs.iloc[[i == "Control" for i in h5ad.obs['Disease_Identity']],:]["Subject_Identity"])
            self.patients = {str(count+1): value for count,value in enumerate([i for i in patients if i not in ['084C', '244C']]) } 
            self.save_dict_to_json(self.patients, self.data_dir + "COPD_control_patients.json")

        self.data_ID, self.labels_ID, self.data_OOD, self.labels_OOD = self.filter_counts_h5ad(sparse.csr_matrix(h5ad.X), h5ad.obs)
       
        
    def filter_counts_h5ad(self, data, obs):
        """Note this function is structured slightly different than the other modules, filtering of everything starts as beginning step"""

        # Select correct subset of data
        if self.scenario.startswith('patient'):
            scen_obs = obs.iloc[[i == "Control" for i in obs["Disease_Identity"]],:]
            scen_data = data[[i == "Control" for i in obs["Disease_Identity"]],:]

            # Filter out patients with less than self.min_cells_patients observations
            patients =  np.unique(scen_obs['Subject_Identity'])
            self.patients_remove = []
            for p in patients:
                c = sum([i == p for i in scen_obs["Subject_Identity"]])
                if c < self.min_cells_patient:
                    self.patients_remove.append(p)

            scen_obs_fp = scen_obs.iloc[[i not in self.patients_remove for i in scen_obs["Subject_Identity"]],:]
            scen_data_fp = scen_data[[i not in self.patients_remove for i in scen_obs["Subject_Identity"]],:]
            self.data = scen_data_fp

        elif self.scenario == "COPD":
            scen_obs_fp = obs.iloc[[i in ["Control", "COPD"] for i in obs["Disease_Identity"]],:]
            scen_data_fp = data[[i in ["Control", "COPD"] for i in obs["Disease_Identity"]],:]
            self.data = scen_data_fp

        elif self.scenario == "IPF":
            scen_obs_fp = obs.iloc[[i in ["Control", "IPF"] for i in obs["Disease_Identity"]],:]
            scen_data_fp = data[[i in ["Control", "IPF"] for i in obs["Disease_Identity"]],:]
            self.data = scen_data_fp

        elif self.scenario.startswith('s_'):
            scen_obs = obs.iloc[[i == "Control" for i in obs["Disease_Identity"]],:]
            scen_data = data[[i == "Control" for i in obs["Disease_Identity"]],:]
            self.data = scen_data
        else:
            print('Invalid scenario')
        
        
        
        # Split into ID and OOD data
        if self.scenario.startswith('patient'):
            current_patient = self.scenario.split('_')[-1]
            data_OOD = scen_data_fp[scen_obs_fp["Subject_Identity"].values == self.patients[current_patient],:]
            obs_OOD = scen_obs_fp.iloc[scen_obs_fp["Subject_Identity"].values == self.patients[current_patient],:]["Subclass_Cell_Identity"]
            data_ID = scen_data_fp[scen_obs_fp["Subject_Identity"].values != self.patients[current_patient],:]
            obs_ID = scen_obs_fp.iloc[scen_obs_fp["Subject_Identity"].values != self.patients[current_patient],:]["Subclass_Cell_Identity"]

            # Filter out cell types with less than self.min_celltypes observations
            # For OOD
            labels_OOD = obs_OOD.values
            s = labels_OOD.value_counts() < self.min_celltypes
            celltype_to_filter = s.index[s].to_list()
            obs_OOD_fc = obs_OOD.iloc[[i not in celltype_to_filter for i in obs_OOD.values]]
            data_OOD_fc = data_OOD[[i not in celltype_to_filter for i in obs_OOD.values],:]
            self.celltypes_filtered_OOD = celltype_to_filter
    

            # For ID
            labels_ID = obs_ID.values
            s = labels_ID.value_counts() < self.min_celltypes
            celltype_to_filter = s.index[s].to_list()
            obs_ID_fc = obs_ID.iloc[[i not in celltype_to_filter for i in obs_ID]]
            data_ID_fc = data_ID[[i not in celltype_to_filter for i in obs_ID],:]
            self.celltypes_filtered_ID = celltype_to_filter

        elif self.scenario == "COPD":
            obs_OOD = scen_obs_fp.iloc[[i == "COPD" for i in scen_obs_fp["Disease_Identity"]],:]["Subclass_Cell_Identity"]
            data_OOD = scen_data_fp[[i == "COPD" for i in scen_obs_fp["Disease_Identity"]],:]
            obs_ID  = scen_obs_fp.iloc[[i == "Control" for i in scen_obs_fp["Disease_Identity"]],:]["Subclass_Cell_Identity"]
            data_ID = scen_data_fp[[i ==  "Control" for i in scen_obs_fp["Disease_Identity"]],:]

            # Filter out cell types with less than self.min_celltypes observations
            # For OOD
            labels_OOD = obs_OOD.values
            s = labels_OOD.value_counts() < self.min_celltypes
            celltype_to_filter = s.index[s].to_list()
            obs_OOD_fc = obs_OOD.iloc[[i not in celltype_to_filter for i in obs_OOD.values]]
            data_OOD_fc = data_OOD[[i not in celltype_to_filter for i in obs_OOD.values],:]
            self.celltypes_filtered_OOD = celltype_to_filter
    

            # For ID
            labels_ID = obs_ID.values
            s = labels_ID.value_counts() < self.min_celltypes
            celltype_to_filter = s.index[s].to_list()
            obs_ID_fc = obs_ID.iloc[[i not in celltype_to_filter for i in obs_ID]]
            data_ID_fc = data_ID[[i not in celltype_to_filter for i in obs_ID],:]
            self.celltypes_filtered_ID = celltype_to_filter
        elif self.scenario == "IPF":
            obs_OOD = scen_obs_fp.iloc[[i == "IPF" for i in scen_obs_fp["Disease_Identity"]],:]["Subclass_Cell_Identity"]
            data_OOD = scen_data_fp[[i == "IPF" for i in scen_obs_fp["Disease_Identity"]],:]
            obs_ID  = scen_obs_fp.iloc[[i == "Control" for i in scen_obs_fp["Disease_Identity"]],:]["Subclass_Cell_Identity"]
            data_ID = scen_data_fp[[i ==  "Control" for i in scen_obs_fp["Disease_Identity"]],:]

            # Filter out cell types with less than self.min_celltypes observations
            # For OOD
            labels_OOD = obs_OOD.values
            s = labels_OOD.value_counts() < self.min_celltypes
            celltype_to_filter = s.index[s].to_list()
            obs_OOD_fc = obs_OOD.iloc[[i not in celltype_to_filter for i in obs_OOD.values]]
            data_OOD_fc = data_OOD[[i not in celltype_to_filter for i in obs_OOD.values],:]
            self.celltypes_filtered_OOD = celltype_to_filter

            # For ID
            labels_ID = obs_ID.values
            s = labels_ID.value_counts() < self.min_celltypes
            celltype_to_filter = s.index[s].to_list()
            obs_ID_fc = obs_ID.iloc[[i not in celltype_to_filter for i in obs_ID]]
            data_ID_fc = data_ID[[i not in celltype_to_filter for i in obs_ID],:]
            self.celltypes_filtered_ID = celltype_to_filter

        elif self.scenario.startswith('s_'):
            # Filter with less than 10 celltypes
            labels_obs = scen_obs["Subclass_Cell_Identity"]
            print(self.min_celltypes/self.test_size)
            s = labels_obs.value_counts() < self.min_celltypes/self.test_size
            celltype_to_filter = s.index[s].to_list()
            scen_obs_ = scen_obs.iloc[[i not in celltype_to_filter for i in scen_obs["Subclass_Cell_Identity"]],:]
            scen_data_= scen_data[[i not in celltype_to_filter for i in scen_obs["Subclass_Cell_Identity"]],:]

            # Split the data
            data_ID, data_OOD, obs_ID, obs_OOD = train_test_split(
            scen_data_,
            scen_obs_["Subclass_Cell_Identity"],
            test_size=self.test_size,
            stratify=scen_obs_["Subclass_Cell_Identity"],
            random_state=0)

        else:
            print('Invalid scenario')

        # Filter out cell types with less than self.min_celltypes observations for s_
        if self.scenario.startswith('s_'):
            len_tot = len(scen_obs["Subclass_Cell_Identity"])

            # Identify labels to exclude
            if self.scenario.split('_')[1] == "10":
                n = int(round(len_tot*0.1))
                cs = obs_ID.value_counts(ascending = True).cumsum()
                idx = np.where([i > n for i in cs])[0][0]
                labels_to_exclude = list(cs[:idx].index)
            elif self.scenario.split('_')[1] == "15":
                n = int(round(len_tot*0.15))
                cs = obs_ID.value_counts(ascending = True).cumsum()
                idx = np.where([i > n for i in cs])[0][0]
                labels_to_exclude = list(cs[:idx].index)
            elif self.scenario.split('_')[1] == "5":
                n = int(round(len_tot*0.05))
                cs = obs_ID.value_counts(ascending = True).cumsum()
                idx = np.where([i > n for i in cs])[0][0]
                labels_to_exclude = list(cs[:idx].index)
            elif self.scenario.split('_')[1] == "2":
                n = int(round(len_tot*0.02))
                cs = obs_ID.value_counts(ascending = True).cumsum()
                idx = np.where([i > n for i in cs])[0][0]
                labels_to_exclude = list(cs[:idx].index)

            self.OOD_labels_excluded = labels_to_exclude
            obs_ID_fcs = obs_ID.iloc[[i not in labels_to_exclude for i in obs_ID.values]]
            data_ID_fcs = data_ID[[i not in labels_to_exclude for i in obs_ID.values],:]

            print('length labels to exclude', len(labels_to_exclude))
            print('unique obs_id_FCS', len(np.unique(obs_ID_fcs)))
            print('unique obs_ood_FC', len(np.unique(obs_OOD)))
            return data_ID_fcs, obs_ID_fcs, data_OOD, obs_OOD

        else:
            return data_ID_fc, obs_ID_fc, data_OOD_fc, obs_OOD_fc

    def pick_scenario(self):
        if self.scenario.startswith('s_'):
            X_train, X_val, y_train_, y_val_ = train_test_split(
                self.data_ID,
                self.labels_ID,
                test_size= self.val_size,
                stratify=self.labels_ID.values,
                random_state=0,
            )
    
            # Convert train and val to torch plus convert str labels to int
            y_train = [self.conversion_dict[i] for i in y_train_.values]
            y_val = [self.conversion_dict[i] for i in y_val_.values]
            X_train = torch.from_numpy(X_train.todense())
            X_val = torch.from_numpy(X_val.todense())
            y_train = torch.from_numpy(np.array(y_train))
            y_val  = torch.from_numpy(np.array(y_val))


            # Convert OOD data to dense 
            data_OOD = torch.from_numpy(self.data_OOD.todense())
            self.data_OOD = data_OOD
            y_OOD_ = [self.conversion_dict[i] for i in self.labels_OOD.values]
            labels_OOD = torch.from_numpy(np.array(y_OOD_))

            OOD_ind = pd.DataFrame(
                [-1 if i in [self.conversion_dict[j] for j in self.OOD_labels_excluded] else 1 for i in labels_OOD]
            )
            OOD_ind.to_csv(
                self.data_dir
                + "OOD_ind_COPD"
                + "_dataset_"
                + str(self.name)
                + ".csv"
            )
            return X_train, X_val, self.data_OOD, y_train, y_val, labels_OOD

        else:
            train_ratio = 1 - (self.val_size + self.test_size)
            # Split "ID" data
            X_train, X_val_test, y_train_, y_val_test = train_test_split(
                self.data_ID,
                self.labels_ID,
                test_size=1-train_ratio,
                stratify=self.labels_ID.values,
                random_state=0,
            )
            self.y_val_test =  y_val_test
    
            X_val, X_test, y_val_, y_test = train_test_split(
                X_val_test,
                y_val_test,
                test_size=self.test_size / (self.val_size + self.test_size),
                stratify=y_val_test.values,
                random_state=0,
            )

            # Convert train and val to torch plus convert str labels to int
            y_train = [self.conversion_dict[i] for i in y_train_.values]
            y_val = [self.conversion_dict[i] for i in y_val_.values]
            X_train = torch.from_numpy(X_train.todense())
            X_val = torch.from_numpy(X_val.todense())
            y_train = torch.from_numpy(np.array(y_train))
            y_val  = torch.from_numpy(np.array(y_val))

            # Convert test ID and OOD data to dense and concatenate
            data_test = torch.from_numpy(X_test.todense())
            y_test_ = [self.conversion_dict[i] for i in y_test.values]
            labels_test = torch.from_numpy(np.array(y_test_))
            data_OOD = torch.from_numpy(self.data_OOD.todense())
            self.data_OOD = data_OOD
            y_OOD_ = [self.conversion_dict[i] for i in self.labels_OOD.values]
            labels_OOD = torch.from_numpy(np.array(y_OOD_))
            data_test_total = torch.cat((data_test, data_OOD), 0)
            labels_test_total = torch.cat((labels_test, labels_OOD), 0)

            # Make an OOD indicator file
            OOD_ind = pd.DataFrame(
                [1] * data_test.size(dim=0) + [-1] * data_OOD.size(dim=0)
            )
            OOD_ind.to_csv(
                self.data_dir
                + "OOD_ind_COPD"
                + "_dataset_"
                + str(self.name)
                + ".csv"
            )
            return X_train, X_val, data_test_total, y_train, y_val, labels_test_total

    def check_for_OOD(self, y_train, y_test):
        """Check if OOD labels/celltypes are present in train/test split

        Parameters
        ----------
        y_train
        y_test

        Returns
        -------
        ind: boolean
            Are there OOD labels?
        idx_OOD: list
            boolean list: TRUE/FALSE this label is OOD
        """
        un_test = np.unique(y_test)
        un_train = np.unique(y_train)
        self.un_test = un_test
        self.un_train = un_train

        if sum([t not in un_train for t in un_test]) > 0:
            ind = True
            idx_OOD = pd.DataFrame([-1 if t not in un_train else 1 for t in y_test.numpy()])
        else:
            ind = False
            idx_OOD = pd.DataFrame([None])
        return ind, idx_OOD

    def setup(self,stage):
        # Read in data and perform filtering
        self.prepare_data()

        # Perform train-test split
        X_train, X_val, X_test, y_train, y_val, y_test = self.pick_scenario()
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        
        self.data_test = BaseDataset( # Already in dense format since concatenating with OOD data
            X_test,
            y_test,
        )

        self.data_train = BaseDataset(
            X_train,
            y_train,
        )

        self.data_val = BaseDataset(
            X_val,
            y_val,
        )

        OOD_idx, OOD_ind = self.check_for_OOD(y_train, y_test)
        self.OOD_idx = OOD_idx
        self.OOD_ind = OOD_ind
        if self.verbose:
            print("OOD celltypes?", OOD_idx)
            if OOD_ind.iloc[0,0] != None:
                print("Size OOD data?", sum([1 for i in OOD_ind.iloc[:,0].values if i == -1]))
            print(" \n")

        OOD_ind.to_csv(
            self.data_dir
            + "OOD_ind_COPD"
            + "_celltypes_"
            + str(self.name)
            + ".csv"
        )

    def train_dataloader(self):
            return DataLoader(self.data_train, batch_size=self.batch, num_workers=self.cpus, shuffle = True, pin_memory = True, pin_memory_device = "cuda")

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch, num_workers=self.cpus)

    def test_dataloader(self):
        test_loader = DataLoader(
            self.data_test, batch_size=self.batch, num_workers=self.cpus, pin_memory = True, pin_memory_device = "cuda"
        )
        return [test_loader]

    def predict_dataloader(self):
        predict_loader = DataLoader(
            self.data_test, batch_size=self.batch, num_workers=self.cpus
        )
        return [predict_loader]
    def n_classes(self):
        """Returns the number of unique label classes"""
        return len(list((self.conversion_dict.keys())))
    def n_features(self):
        """Return the number od features"""
        h5ad = sc.read_h5ad(self.data_dir + self.data_file)
        data = sparse.csr_matrix(h5ad.X)
        if self.verbose:
            print('n_features dataloader', data.shape[1] )
        return data.shape[1]
    
    def display_data_characteristics(self):
        print("\n")
        print('Shape ID data', self.data_ID.shape)
        print('Shape OOD data', self.data_OOD.shape)
        print('n celltypes OOD', len(np.unique(self.labels_OOD)))
        print('n celltypes ID', len(np.unique(self.labels_ID)))