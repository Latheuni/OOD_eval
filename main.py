## packages

import os
import time
import numpy as np
import pandas as pd
#import wandb
import torch
import argparse
import datetime
from torch import nn
import pytorch_lightning as L

from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    BasePredictionWriter,
    LearningRateMonitor
)
from torcheval.metrics import MulticlassAccuracy
from pytorch_lightning.loggers import TensorBoardLogger
from Metrics import auc_and_fpr_recall
## Own imports
from utils import read_config
from Datasets import LitPancreasDataModule
from Networks import LinearNetwork, NonLinearNetwork
from Trainers import LitBasicNN
from Losses import LogitNormLoss

# Code
## basic module
import torch
from pytorch_lightning.callbacks import BasePredictionWriter


class CustomWriter(BasePredictionWriter):
    def __init__(self, config_file):
        super().__init__(
            main_config["write_interval"]
        )  # 'batch', 'epoch', 'batch_and_epoch'
        self.output_dir = main_config["output_dir"]

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        torch.save(
            prediction, os.path.join(self.output_dir, dataloader_idx, f"{batch_idx}.pt")
        )

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))


class LitBasicNN(L.LightningModule):
    def __init__(self, NN, loss_function, learning_rate, n_classes, decay = 0.95):
        super().__init__()
        self.NN = NN
        self.loss_function = loss_function
        self.lr = learning_rate
        self.decay = decay
        self.accuracy = MulticlassAccuracy(num_classes=n_classes, average="micro") # is not callable
        self.balanced_accuracy = MulticlassAccuracy(
            num_classes=n_classes, average="macro"
        )
        self.save_hyperparameters()
        
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        scores = self.NN(x)
        loss = self.loss_function(scores, y)
        self.log(
            "train_loss", loss
        )  # on_epoch acculumate and rduces all metric to the end of the epoch, on_step that specific call will not accumulate metrics
        self.accuracy.update(scores, y)
        self.balanced_accuracy.update(scores, y)
        self.log("training accuracy", self.accuracy.compute())
        self.log("training balanced accuracy", self.balanced_accuracy.compute())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        scores = self.NN(x)
        val_loss = self.loss_function(scores, y)
        self.log("val_loss", val_loss)
        self.accuracy.update(scores, y)
        self.balanced_accuracy.update(scores, y)
        self.log("validation accuracy", self.accuracy.compute())
        self.log("validation balanced accuracy", self.balanced_accuracy.compute())

    def test_step(self, batch, batch_idx):
        x, y = batch
        scores = self.NN(x)
        test_loss = self.loss_function(scores, y)
        self.log("test_loss", test_loss)
        self.accuracy.update(scores, y)
        self.balanced_accuracy.update(scores, y)
        self.log("accuracy", self.accuracy.compute())
        self.log("balanced accuracy", self.balanced_accuracy.compute())  
        if batch_idx == 0:
            self.ytrue = y
            self.scores = scores
        else:
            self.ytrue = torch.cat((self.ytrue, y),0 )
            self.scores = torch.cat((self.scores, scores),0 )
        return scores, y

    def predict_step(self, batch, batch_idx): # Assume loss needs to be minimized
        x,y = batch
        scores = self.NN(x)
        if batch_idx == 0:
            self.predictions = torch.argmin(scores, dim=1)
        else:
            self.predictions = torch.cat((self.predictions, torch.argmin(scores, dim=1)),0) 
        return torch.argmin(scores, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=float(self.lr)) # can still add weight decay
        lambd = lambda epoch: self.decay
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=lambd
        )
        return optimizer

def train_step(config_file, train_test_together = False):

    # Read in parameters
    main_config, dataset_config, network_config, training_config = read_config(
        config_file
    )
    verbose = main_config["verbose"]
    if training_config["loss_function"] == "logitnorm":
        loss_function = LogitNormLoss()

    # Set up directionary to save the results
    if verbose == "True":
        print("Set up direcionary and environment")
    if not os.path.exists(main_config["output_dir"]):
        os.mkdir(main_config["output_dir"])

    # Define the dataset
    if dataset_config["name"] == "Pancreas":
        DataLoader = LitPancreasDataModule(
            dataset_config["data_dir"],
            dataset_config["data_file"],
            dataset_config["label_conversion_file"],
            dataset_config["batch_size"],
            dataset_config["train_techs"],
            dataset_config["OOD_techs"],
            dataset_config["test_size"],
            dataset_config["validation_size"],
            dataset_config["min_celltypes"],
            main_config['cpus'],
            main_config['name'],
            main_config['verbose'],
        )
        n_classes = DataLoader.n_classes()

    # Define network
    if network_config["model"] == "linear":
        network = LinearNetwork(
            network_config["input_dim"],
            network_config["output_dim"],
            network_config["nodes_per_layer"],
            network_config["num_hidden_layer"],
        )
    else:
        network = NonLinearNetwork(
            network_config["input_dim"],
            network_config["output_dim"],
            network_config["nodes_per_layer"],
            network_config["num_hidden_layer"],
            network_config["activation"],
        )

    # Define model
    if verbose == "True":
        print("Start training")  # optional: progress bar
    model = LitBasicNN(network, loss_function, training_config["learning_rate"], n_classes)

    # Logger
    Logger = TensorBoardLogger(main_config["output_dir"] + 'tb_logger/', name =main_config['name'] )

    # Define callbacks
    checkpoint_train = ModelCheckpoint(
        monitor="train_loss",
        dirpath=main_config["output_dir"],
        filename = main_config['name'] + '_train_loss'
    )
    checkpoint_val = ModelCheckpoint(
        monitor="val_loss",
        dirpath=main_config["output_dir"],
        filename = main_config['name'] + '_val_loss'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    earlystopping = EarlyStopping(monitor="val_loss", mode="min")  # for cross-entropy
    callbacks_list = [checkpoint_val, checkpoint_train, earlystopping, lr_monitor]

    # Train and validate model
    if main_config["debug"] == "True":
        debug = True
    else:
        debug = False
    trainer = Trainer(
        fast_dev_run=debug,
        max_epochs=training_config["max_epochs"],
        logger=Logger,
        callbacks=callbacks_list,
        default_root_dir=main_config["output_dir"],
        enable_progress_bar=False,
        #accelerator=training_config["accelerator"],
        #devices = training_config['devices'],
    )
    trainer.fit(model, DataLoader)
    
    if train_test_together:
        return model
def test_step(config_file, model):
    """
    model can be or path to save checkpoint or an actual model
    """
    # Read in Config
    main_config, dataset_config, network_config, training_config = read_config(
        config_file)
    verbose = main_config["verbose"]

    # Define the dataset
    if verbose == "True":
        print('Read in model and set up analysis')
    if dataset_config["name"] == "Pancreas":
        dataset = LitPancreasDataModule(
            dataset_config["data_dir"],
            dataset_config["data_file"],
            dataset_config["label_conversion_file"],
            dataset_config["batch_size"],
            dataset_config["train_techs"],
            dataset_config["OOD_techs"],
            dataset_config["test_size"],
            dataset_config["validation_size"],
            dataset_config["min_celltypes"],
            main_config['cpus'],
            main_config['name'],
            main_config['verbose'],
        )
        OOD_label_dataset = pd.read_csv(dataset_config["data_dir"] + 'OOD_ind_pancreas'+ '_dataset_' + main_config['name'] + '.csv', index_col = 0)
        OOD_label_celltype = pd.read_csv(dataset_config["data_dir"] + 'OOD_ind_pancreas'+ '_celltypes_' + main_config['name'] + '.csv', index_col = 0)

    # load model
    if type(model) is str:
        os.chdir(main_config["output_dir"])
        model = LitBasicNN.load_from_checkpoint(filename)

    # Logger
    Logger = TensorBoardLogger(main_config["output_dir"] + 'tb_logger/', name =main_config['name'])

    # Define callbacks
    checkpoint_test = ModelCheckpoint(
        monitor="test_loss",
        dirpath=main_config["output_dir"],
        filename = main_config['name'] + '_test_loss_' + "{epoch}"
    )
    pred_writer = CustomWriter(
        main_config
    )
    callbacks_list = [checkpoint_test, pred_writer]

    # Trainer
    trainer = Trainer(
        max_epochs=training_config["max_epochs"],
        logger=Logger,
        callbacks=callbacks_list,
        default_root_dir=main_config["output_dir"],
        enable_progress_bar=False,
        #accelerator=main_config["accelerator"],
    )
    # Test
    if verbose == "True":
        print('Start testing')
    trainer.test(model, datamodule=dataset)

    # predict
    if verbose == "True":
        print('Start predicting')
    trainer.predict(model, datamodule=dataset)

    # Calculate statistics
    confidence = torch.max(model.scores,1).values

    results_dict = {}
    auroc_dataset, aupr_in_dataset, aupr_out_dataset, fpr_dataset = auc_and_fpr_recall(confidence.numpy(), OOD_label_dataset.iloc[:,0].values, 0.95)
    results_dict['dataset'] = {"auroc":auroc_dataset, "aupr_in": aupr_in_dataset, "aupr_out": aupr_out_dataset, "fpr": fpr_dataset}
    print(' \n')
    print('-------')
    print('Results')
    print('-------')
    print("For the dataset")
    print('auroc', auroc_dataset)
    print('aupr_in', aupr_in_dataset)
    print('aupr_out', aupr_out_dataset)
    print('fpr', fpr_dataset)
    
    if not np.isnan(OOD_label_celltype.iloc[0,0]):
        auroc_celltype, aupr_in_celltype, aupr_out_celltype, fpr_celltype = auc_and_fpr_recall(confidence.numpy(), OOD_label_celltype.iloc[:,0].values, 0.95)
        results_dict['celltype'] = {"auroc":auroc_celltype, "aupr_in": aupr_in_celltype, "aupr_out": aupr_out_celltype, "fpr": fpr_celltype}
        print('-------')
        print("For the celltypes")
        print('auroc', auroc_celltype)
        print('aupr_in', aupr_in_celltype)
        print('aupr_out', aupr_out_celltype)
        print('fpr', fpr_celltype)
    else:
        print('No OOD celltypes, so no celltype analysis')

    print('\n')
    # save results
    save_dict_to_json(results_dict, main_config['name'])

    return model.scores, model.ytrue, model.predictions

import json
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

def save_dict_to_json(d_results, name_analysis):     
    with open(str(name_analysis) + "_results.json", "w") as f1:
            json.dump(d_results, f1, default = default)

#######################
### Actual analysis ###
#######################

# Arguments
## Parameters
parser = argparse.ArgumentParser()
parser.add_argument("config_file", type=str)
parser.add_argument("Run_step", type=str)
parser.add_argument("filename", type=str, nargs='?')
args = parser.parse_args()
main_config, dataset_config, network_config, training_config = read_config(
    args.config_file
)
print('Cuda available?:', torch.cuda.is_available())
if args.Run_step == "train":
    start = time.time()
    train_step(args.config_file)
    end = time.time()
    if main_config['verbose'] == "True":
        print('Total training time', end - start)

elif args.Run_step == "test":
    start = time.time()
    print(main_config['output_dir'] + args.filename)
    predictions = test_step(args.config_file, main_config['output_dir'] + args.filename)
    end = time.time()
    if main_config['verbose'] == "True":
        print('Total OOD testing time', end - start)

elif args.Run_step == "all":
    # Training
    start = time.time()
    model = train_step(args.config_file, train_test_together = True)
    end = time.time()
    if main_config['verbose'] == "True":
        print('Total training time', end - start)

    # Testing
    start = time.time()
    scores, ytrue, predictions = test_step(args.config_file,  model)
    end = time.time()
    if main_config['verbose'] == "True":    
        print('Total OOD testing time', end - start)
    print('predictions', predictions.size())

