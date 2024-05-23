## packages

import os
import time
import numpy as np
import pandas as pd
import argparse
import datetime

import pytorch_lightning as L
from pytorch_lightning import Trainer
from torch.nn.functional import softmax
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    BasePredictionWriter,
    LearningRateMonitor,
    DeviceStatsMonitor,
)
from torchmetrics.classification import (
    Accuracy,
)  # MultiClassAccuracy in newer versions of lightning (here 0.9.3)
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import BasePredictionWriter
import torch
from torch import nn

print("Pytorch Lighntning version used is", L.__version__)
if L.__version__ > "0.9.3":
    from torchmetrics.classification import (
        MulticlassAccuracy,
    )
else:
    from torchmetrics.classification import (
        Accuracy,
    )

## Own imports
from utils import read_config
from Datasets import LitPancreasDataModule
from Networks import LinearNetwork, NonLinearNetwork
from Trainers import LitBasicNN
from Losses import *
from Metrics import *


# Code
## basic module


# Writes predictions in .pt format
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
    def __init__(self, NN, loss_function, learning_rate, n_classes, decay=0.95):
        super().__init__()
        self.NN = NN
        self.loss_function = loss_function
        self.lr = learning_rate
        self.decay = decay

        if L.__version__ > "0.9.3":
            self.accuracy = MulticlassAccuracy(
                num_classes=n_classes, average="micro"
            )  # is not callable
            self.balanced_accuracy = MulticlassAccuracy(
                num_classes=n_classes, average="macro"
            )
            self.train_accuracy = MulticlassAccuracy(
                num_classes=n_classes, average="micro"
            )
            self.val_accuracy = MulticlassAccuracy(
                num_classes=n_classes, average="micro"
            )
            self.test_accuracy = MulticlassAccuracy(
                num_classes=n_classes, average="micro"
            )
            self.val_balanced_accuracy = MulticlassAccuracy(
                num_classes=n_classes, average="macro"
            )
            self.test_balanced_accuracy = MulticlassAccuracy(
                num_classes=n_classes, average="macro"
            )
        else:
            self.train_accuracy = Accuracy(num_classes=n_classes, average="micro")
            self.val_accuracy = Accuracy(num_classes=n_classes, average="micro")
            self.test_accuracy = Accuracy(num_classes=n_classes, average="micro")
            self.val_balanced_accuracy = Accuracy(
                num_classes=n_classes, average="macro"
            )
            self.test_balanced_accuracy = Accuracy(
                num_classes=n_classes, average="macro"
            )
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        scores = self.NN(x)
        loss = self.loss_function(scores, y)
        scores = F.softmax(scores, dim=-1)
        self.log(
            "train_loss", loss, on_step=True
        )  # on_epoch acculumate and rduces all metric to the end of the epoch, on_step that specific call will not accumulate metrics
        self.train_accuracy(scores, y)
        self.log("training accuracy", self.train_accuracy, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        scores = self.NN(x)
        val_loss = self.loss_function(scores, y)
        scores = F.softmax(scores, dim=-1)
        self.log("val_loss", val_loss, on_step=True)
        self.val_accuracy(scores, y)
        self.val_balanced_accuracy(scores, y)
        self.log("validation accuracy", self.val_accuracy, on_step=True)
        self.log(
            "validation balanced accuracy", self.val_balanced_accuracy, on_step=True
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        scores = self.NN(x)
        test_loss = self.loss_function(scores, y)
        scores = F.softmax(scores, dim=-1)
        self.log("test_loss", test_loss, on_step=True)
        self.test_accuracy(scores, y)
        self.test_balanced_accuracy(scores, y)
        self.log("test accuracy", self.test_accuracy, on_step=True)
        self.log("test balanced accuracy", self.test_balanced_accuracy, on_step=True)
        if batch_idx == 0:
            self.ytrue = y
            self.scores = scores  # Unsure if this is correct: check!
        else:
            self.ytrue = torch.cat((self.ytrue, y), 0)
            self.scores = torch.cat((self.scores, scores), 0)
        return scores, y

    def predict_step(
        self, batch, batch_idx
    ):  # Loss needs to be minimized, max scores are correct label
        x, y = batch
        scores = self.NN(x)
        scores = F.softmax(scores, dim=-1)
        if batch_idx == 0:
            self.predictions = torch.argmax(scores, dim=1)
        else:
            self.predictions = torch.cat(
                (self.predictions, torch.argmax(scores, dim=1)), 0
            )
        return torch.argmax(scores, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=float(self.lr)
        )  # can still add weight decay
        lambd = lambda epoch: self.decay
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=lambd
        )
        return optimizer


def train_step(config_file, train_test_together=False):
    # Read in parameters
    main_config, dataset_config, network_config, training_config = read_config(
        config_file
    )
    verbose = main_config["verbose"]
    if training_config["loss_function"] == "logitnorm":
        loss_function = LogitNormLoss()
    elif training_config["loss_function"] in ["cross-entropy", "cross entropy"]:
        loss_function = CrossEntropyLoss()

    # Set up directionary to save the results
    if verbose == "True":
        print("Set up direcionary and environment")
    if not os.path.exists(main_config["output_dir"] + main_config["name"] + "/"):
        os.mkdir(main_config["output_dir"] + main_config["name"] + "/")

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
            training_config["cpus"],
            main_config["name"],
            main_config["verbose"],
        )
        n_classes = DataLoader.n_classes()
        n_features = DataLoader.n_features()

    # Define network
    if network_config["model"] == "linear":
        network = LinearNetwork(
            n_features,
            n_classes,
            network_config["nodes_per_layer"],
            network_config["num_hidden_layer"],
        )
    else:
        network = NonLinearNetwork(
            n_features,
            n_classes,
            network_config["nodes_per_layer"],
            network_config["num_hidden_layer"],
            network_config["activation"],
        )

    # Define model
    if verbose == "True":
        print("Start training")  # optional: progress bar
    model = LitBasicNN(
        network, loss_function, training_config["learning_rate"], n_classes
    )

    # Logger
    Logger = TensorBoardLogger(
        main_config["output_dir"] + "tb_logger/", name=main_config["name"]
    )

    # Define callbacks
    checkpoint_train = ModelCheckpoint(
        monitor="train_loss",
        dirpath=main_config["output_dir"] + main_config["name"] + "/",
        filename=main_config["name"] + "_train_loss",
    )
    checkpoint_val = ModelCheckpoint(
        monitor="val_loss",
        dirpath=main_config["output_dir"] + main_config["name"] + "/",
        filename=main_config["name"] + "_val_loss",
    )
    device_stats = DeviceStatsMonitor()
    lr_monitor = LearningRateMonitor(logging_interval="step")
    earlystopping = EarlyStopping(monitor="val_loss", mode="min")  # for cross-entropy
    callbacks_list = [
        checkpoint_val,
        checkpoint_train,
        earlystopping,
        lr_monitor,
        device_stats,
    ]

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
        default_root_dir=main_config["output_dir"] + main_config["name"] + "/",
        enable_progress_bar=False,
        accelerator=training_config["accelerator"],
        devices=training_config["devices"],
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
        config_file
    )
    verbose = main_config["verbose"]

    # Define the dataset
    if verbose == "True":
        print("Read in model and set up analysis")
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
            training_config["cpus"],
            main_config["name"],
            main_config["verbose"],
        )
        OOD_label_dataset = pd.read_csv(
            dataset_config["data_dir"]
            + "OOD_ind_pancreas"
            + "_dataset_"
            + main_config["name"]
            + ".csv",
            index_col=0,
        )
        OOD_label_celltype = pd.read_csv(
            dataset_config["data_dir"]
            + "OOD_ind_pancreas"
            + "_celltypes_"
            + main_config["name"]
            + ".csv",
            index_col=0,
        )

    # load model
    if type(model) is str:
        os.chdir(main_config["output_dir"] + main_config["name"] + "/")
        model = LitBasicNN.load_from_checkpoint(model)

    # Logger
    Logger = TensorBoardLogger(
        main_config["output_dir"] + "tb_logger/", name=main_config["name"]
    )

    # Define callbacks
    checkpoint_test = ModelCheckpoint(
        monitor="test_loss",
        dirpath=main_config["output_dir"] + main_config["name"] + "/",
        filename=main_config["name"] + "_test_loss_" + "{epoch}",
    )
    pred_writer = CustomWriter(main_config)
    device_stats = DeviceStatsMonitor()
    callbacks_list = [checkpoint_test, pred_writer, device_stats]

    # Trainer
    trainer = Trainer(
        max_epochs=training_config["max_epochs"],
        logger=Logger,
        callbacks=callbacks_list,
        default_root_dir=main_config["output_dir"] + main_config["name"] + "/",
        enable_progress_bar=False,
        accelerator=training_config["accelerator"],
        devices=training_config["devices"],
    )

    # Test
    if verbose == "True":
        print("Start testing")
    trainer.test(model, datamodule=dataset)

    # predict
    if verbose == "True":
        print("Start predicting")
    trainer.predict(model, datamodule=dataset)

    # Calculate statistics
    confidence = torch.max(model.scores, 1).values

    results_dict = {}
    auroc_dataset, aupr_in_dataset, aupr_out_dataset, fpr_dataset = auc_and_fpr_recall(
        confidence.cpu().numpy(), OOD_label_dataset.iloc[:, 0], 0.95
    )
    acc_OOD, acc_ID, bacc_OOD, bacc_ID = general_metrics(
        model.scores.cpu().numpy(),
        OOD_label_dataset.iloc[:, 0].values,
        model.predictions.cpu().numpy(),
        model.ytrue.cpu().numpy(),
        verbose,
    )
    results_dict["dataset"] = {
        "auroc": auroc_dataset,
        "aupr_in": aupr_in_dataset,
        "aupr_out": aupr_out_dataset,
        "fpr": fpr_dataset,
        "acc_in": acc_ID,
        "acc_out": acc_OOD,
        "bacc_in": bacc_ID,
        "bacc_out": bacc_OOD,
    }
    print(" \n")
    print("-------")
    print("Results")
    print("-------")
    print("For the dataset")
    print("auroc", auroc_dataset)
    print("aupr_in", aupr_in_dataset)
    print("aupr_out", aupr_out_dataset)
    print("fpr", fpr_dataset)

    if not np.isnan(OOD_label_celltype.iloc[0, 0]):
        auroc_celltype, aupr_in_celltype, aupr_out_celltype, fpr_celltype = (
            auc_and_fpr_recall(
                confidence.cpu().numpy(), OOD_label_celltype.iloc[:, 0].values, 0.95
            )
        )
        acc_OOD, acc_ID, bacc_OOD, bacc_ID = general_metrics(
            model.scores.cpu().numpy(),
            OOD_label_celltype.iloc[:, 0].values,
            model.predictions.cpu().numpy(),
            model.ytrue.cpu().numpy(),
            verbose,
        )
        results_dict["celltype"] = {
            "auroc": auroc_celltype,
            "aupr_in": aupr_in_celltype,
            "aupr_out": aupr_out_celltype,
            "fpr": fpr_celltype,
            "acc_in": acc_ID,
            "acc_out": acc_OOD,
            "bacc_in": bacc_ID,
            "bacc_out": bacc_OOD,
        }
        print("-------")
        print("For the celltypes")
        print("auroc", auroc_celltype)
        print("aupr_in", aupr_in_celltype)
        print("aupr_out", aupr_out_celltype)
        print("fpr", fpr_celltype)
    else:
        results_dict["celltype"] = None
        print("No OOD celltypes, so no celltype analysis")
    max_elem, max_ind = torch.max(model.scores.cpu(), dim=1)
    R = Accuracy_reject_curves(
        max_elem, model.ytrue.cpu().numpy(), model.predictions.cpu().numpy()
    )
    plot_AR_curves(
        R,
        main_config["output_dir"]
        + main_config["name"]
        + "/"
        + main_config["name"]
        + "_AR_plot.png",
    )
    R.to_csv(
        main_config["output_dir"]
        + main_config["name"]
        + "/"
        + main_config["name"]
        + "_AR.csv"
    )
    print("\n")
    # save results
    save_dict_to_json(
        results_dict,
        main_config["output_dir"] + main_config["name"] + "/" + main_config["name"],
    )

    return (
        model.scores.cpu().numpy(),
        model.ytrue.cpu().numpy(),
        model.predictions.cpu().numpy(),
    )


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
        json.dump(d_results, f1, default=default)


def save_numpy_array(obj, file_dir):
    if torch.is_tensor(obj):
        obj = obj.numpy()

    df = pd.DataFrame(obj)
    df.to_csv(file_dir)


#######################
### Actual analysis ###
#######################

# Arguments
## Parameters
parser = argparse.ArgumentParser()
parser.add_argument("config_file", type=str)
parser.add_argument("Run_step", type=str)
parser.add_argument("filename", type=str, nargs="?")
args = parser.parse_args()
main_config, dataset_config, network_config, training_config = read_config(
    args.config_file
)
print("Cuda available?:", torch.cuda.is_available())
if args.Run_step == "train":
    start = time.time()
    train_step(args.config_file)
    end = time.time()
    if main_config["verbose"] == "True":
        print("Total training time", end - start)

elif args.Run_step == "test":
    start = time.time()
    scores, ytrue, predictions = test_step(
        args.config_file, main_config["output_dir"] + args.filename
    )
    end = time.time()
    if main_config["verbose"] == "True":
        print("Total OOD testing time", end - start)

    # Saving
    save_numpy_array(
        scores, main_config["output_dir"] + main_config["name"] + "_scores.csv"
    )
    save_numpy_array(
        ytrue, main_config["output_dir"] + main_config["name"] + "_ytrue.csv"
    )
    save_numpy_array(
        predictions,
        main_config["output_dir"] + main_config["name"] + "_predictions.csv",
    )

elif args.Run_step == "all":
    # Training
    start = time.time()
    model = train_step(args.config_file, train_test_together=True)
    end = time.time()
    if main_config["verbose"] == "True":
        print("Total training time", end - start)

    # Testing
    start = time.time()
    scores, ytrue, predictions = test_step(args.config_file, model)
    end = time.time()
    if main_config["verbose"] == "True":
        print("Total OOD testing time", end - start)

    # Saving
    save_numpy_array(
        scores,
        main_config["output_dir"]
        + main_config["name"]
        + "/"
        + main_config["name"]
        + "_scores.csv",
    )
    save_numpy_array(
        ytrue,
        main_config["output_dir"]
        + main_config["name"]
        + "/"
        + main_config["name"]
        + "_ytrue.csv",
    )
    save_numpy_array(
        predictions,
        main_config["output_dir"]
        + main_config["name"]
        + "/"
        + main_config["name"]
        + "_predictions.csv",
    )

# to save tensorboard results: https://www.tensorflow.org/tensorboard/dataframe_api
