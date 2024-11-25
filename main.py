## packages

import os
import time
import numpy as np
import pandas as pd
import argparse
import datetime

import torchmetrics
import pytorch_lightning as L
import torch.nn.functional as F
from pytorch_lightning import Trainer
from torch.nn.functional import softmax
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    BasePredictionWriter,
    LearningRateMonitor,
    DeviceStatsMonitor,
)
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import BasePredictionWriter
import torch
from torch import nn

print("Torchmetrics version used is", torchmetrics.__version__)
if torchmetrics.__version__ > "0.9.3":
    from torchmetrics.classification import (
        MulticlassAccuracy,
    )
else:
    from torchmetrics.classification import (
        Accuracy,
    )

## Own imports
from utils import read_config, load_dataset, load_network, evaluate_OOD
from Losses import LogitNormLoss, CrossEntropyLoss
from Metrics import (
    general_metrics,
    accuracy_reject_curves,
    auc_and_fpr_recall,
    plot_AR_curves,
)
from Post_processors import base_postprocessor, dropout_postprocessor, EBO_postprocessor, Ensemble_postprocessor, KNN_postprocessor, Posterior_postprocessor
from LModule import *
from Networks import Posterior_network


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

# Utility function necessary for posterior networks
def get_N(labels, output_dim): # Saves class counts to vector for posterior networks
    class_index, class_count = np.unique(labels, return_counts=True)
    N = np.zeros(output_dim)
    N[class_index.astype(int)] = class_count
    N = torch.tensor(N)
    return(N)

def train_step(config_file, train_test_together=False):
    # Read in parameters
    main_config, dataset_config, network_config, training_config = read_config(
        config_file
    )
    verbose = main_config["verbose"]

    if verbose:
        print(" \n")
        print("-------")
        print("Start Training")
        print("-------")
    
    if verbose:
        print("Set up direcionary and environment")

    # Set up directory to save the results
    if not os.path.exists(main_config["output_dir"] + main_config["name"] + "/"):
        os.mkdir(main_config["output_dir"] + main_config["name"] + "/")

    # Define the dataset
    DataLoader = load_dataset(config_file, train = True)
    n_classes = DataLoader.n_classes()
    n_features = DataLoader.n_features()
    
    DataLoader.prepare_data()
    DataLoader.setup(stage="train")
    labels = DataLoader.data_train.labels

    # Define Loss function
    loss_dict = {
        "logitnorm": LogitNormLoss(),
        "dropout": CrossEntropyLoss(),
        "EBO": CrossEntropyLoss(),
        "Ensembles": CrossEntropyLoss(),
        "Knn": CrossEntropyLoss(),
        "Posterior": None, # Has it's own build in loss
    }
    loss_function = loss_dict[training_config["OOD_strategy"]]
    
    if training_config["OOD_strategy"] == "Ensembles":
        for i in range(0,10):
            # Define network
            network = load_network(config_file, n_features,  n_classes)

            # Define model
            model = LitBasicNN(
                network, loss_function, training_config["learning_rate"], n_classes
            )

            # Logger
            Logger = TensorBoardLogger(
                main_config["output_dir"] + "tb_logger/", name=main_config["name"]
            )

            # Define callbacks
            checkpoint_val = ModelCheckpoint(
                monitor="val_loss",
                dirpath=main_config["output_dir"] + main_config["name"] + "/",
                filename=main_config["name"] + '_' + str(i) + "_best_model",
            )
            device_stats = DeviceStatsMonitor()
            lr_monitor = LearningRateMonitor(logging_interval="step")
            earlystopping = EarlyStopping(monitor="val_loss", mode="min")  # for cross-entropy
            callbacks_list = [
                checkpoint_val,
                earlystopping,
                lr_monitor,
                device_stats,
            ]

            # Define Trainer
            trainer = Trainer(
                max_epochs=training_config["max_epochs"],
                logger=Logger,
                callbacks=callbacks_list,
                default_root_dir=main_config["output_dir"] + main_config["name"] + "/",
                enable_progress_bar=False,
                accelerator=training_config["accelerator"],
                devices=training_config["devices"],
                log_every_n_steps=2
            )

            # Train the model
            trainer.fit(model, DataLoader)

    else:
        if training_config["OOD_strategy"] == "Posterior":
            N = get_N(labels, n_classes) 
            # Define network and lightning module
            if "n_density" in training_config.keys():
                if "density_type" in training_config.keys():
                    network = Posterior_network(N, n_features, n_classes, network_config["nodes_per_layer"], network_config["num_hidden_layer"], "relu", network_config["model"], n_density = training_config["n_density"], density_type =training_config["density_type"] )
                else:
                    network = Posterior_network(N, n_features, n_classes, network_config["nodes_per_layer"], network_config["num_hidden_layer"], "relu", network_config["model"], n_density = training_config["n_density"])
            else:
                if "density_type" in training_config.keys():
                    network = Posterior_network(N, n_features, n_classes, network_config["nodes_per_layer"], network_config["num_hidden_layer"], "relu", network_config["model"], density_type =training_config["density_type"])

                else:
                    network = Posterior_network(N, n_features, n_classes, network_config["nodes_per_layer"], network_config["num_hidden_layer"], "relu", network_config["model"])
            
            
            if "regr" in training_config.keys():
                model = LitPostNN(N, network,n_classes, batch_size = dataset_config['batch_size'], lr = training_config["learning_rate"], regr = training_config["regr"])
            else:
                model = LitPostNN(N, network,n_classes, batch_size = dataset_config['batch_size'], lr = training_config["learning_rate"] )

           
        else:
            # Define network
            network = load_network(config_file, n_features,  n_classes)

            # Define model
            model = LitBasicNN(
                network, loss_function, training_config["learning_rate"], n_classes
            )

        # Logger
        Logger = TensorBoardLogger(
            main_config["output_dir"] + "tb_logger/", name=main_config["name"]
        )

        # Define callbacks
        checkpoint_val = ModelCheckpoint(
            monitor="val_loss",
            dirpath=main_config["output_dir"] + main_config["name"] + "/",
            filename=main_config["name"] + "_best_model",
        )
        device_stats = DeviceStatsMonitor()
        lr_monitor = LearningRateMonitor(logging_interval="step")
        earlystopping = EarlyStopping(monitor="val_loss", mode="min")  # for cross-entropy
        callbacks_list = [
            checkpoint_val,
            earlystopping,
            lr_monitor,
            device_stats,
        ]

        # Define the Trainer
        trainer = Trainer(
            max_epochs=training_config["max_epochs"],
            logger=Logger,
            callbacks=callbacks_list,
            default_root_dir=main_config["output_dir"] + main_config["name"] + "/",
            enable_progress_bar=False,
            accelerator=training_config["accelerator"],
            devices=training_config["devices"],
            log_every_n_steps=2
        )

        # Train the model
        trainer.fit(model, DataLoader)

    if verbose:
        DataLoader.display_data_characteristics()
        print('n_classes', n_classes)
        print('n_features', n_features)
        print(' \n')
        print("Training complete")

    if train_test_together:
        return trainer, DataLoader 


def test_step(config_file, dataLoader):
    """
    model can be or path to save checkpoint or an actual model
    """
    # Read in Config
    main_config, dataset_config, network_config, training_config = read_config(
        config_file
    )
    verbose = main_config["verbose"]

    if verbose:
        print("-------")
        print("Start Testing")
        print("-------")

    # Define the dataset
    if verbose:
        print("Reading in model and setting up the analysis")
    __ , OOD_label_dataset, OOD_label_celltype = load_dataset(config_file, train= False)
    
    
    test_X = dataLoader.test_dataloader()[0] # is a tensor
    y_true = dataLoader.data_test.labels
    
    if verbose:
        print('OOD in in main', OOD_label_celltype)
        print('OOD label dataset', len(OOD_label_dataset))
        print("shape y_true", y_true.shape)

    # Analyses specifically for Ensembles
    if training_config["OOD_strategy"] == "Ensembles":
        if verbose:
            print(' \n')
            print('Evaluating OOD dataset')

        #Postprocess networks
        postprocessor = Ensemble_postprocessor(main_config["output_dir"] + main_config["name"], main_config["name"] )
        postprocessor.setup()
        pred, conf = postprocessor.postprocess(test_X)
        
        # Calculate Measures
        results_dict = {}
        results_dict = evaluate_OOD(
            conf.detach().cpu().numpy(), pred.detach().cpu().numpy(), y_true.detach().cpu().numpy(), OOD_label_dataset.iloc[:,0].values, "dataset", results_dict
        )

        if not dataset_config["scenario"].startswith('s_'): # For the subsetting data no "celltype" scenarui, already included in "dataset scenario"
            if verbose:
                print(' \n')
                print('Evaluating OOD celltypes')

            if not np.isnan(OOD_label_celltype.iloc[0,0]):
                results_dict = evaluate_OOD(
                    conf.detach().cpu().numpy(), pred.detach().cpu().numpy(), y_true.detach().cpu().numpy(), OOD_label_celltype.iloc[:,0].values, "celltype", results_dict
                )
            else:
                results_dict["celltype"] = None
                if verbose:
                    print("No OOD celltypes, so no celltype analysis")
        
        # Save results
        save_dict_to_json(
            results_dict,
            main_config["output_dir"] + main_config["name"] + "/" + main_config["name"],
        )

    else:
        # load network (with ensembles in postprocessor)
        if training_config["OOD_strategy"] == "Posterior":
            os.chdir(main_config["output_dir"] + "/" + main_config["name"])
            model =  LitPostNN.load_from_checkpoint(main_config["output_dir"] + '/' + main_config["name"] + "/" + main_config["name"] + "_best_model.ckpt")

        else:
            model =  LitBasicNN.load_from_checkpoint(main_config["output_dir"] + main_config["name"] + "/" + main_config["name"] + "_best_model.ckpt")
        network = model.NN #model is the trainer instance, so change to network


        # Logger
        Logger = TensorBoardLogger(
            main_config["output_dir"] + "tb_logger/", name=main_config["name"]
        )

        # Define callbacks
        pred_writer = CustomWriter(main_config)
        device_stats = DeviceStatsMonitor()
        callbacks_list = [pred_writer, device_stats]

        # PostProcess
        postprocessor_dict = {
            "logitnorm": base_postprocessor(),
            "dropout": dropout_postprocessor(),
            "EBO": EBO_postprocessor(),
            "Posterior": Posterior_postprocessor(),
        }

        # Implement for Knn a sweep for values of K
        if training_config["OOD_strategy"] == "Knn":
            if verbose:
                print('Start Knn postprocessor sweep:')
            
            results_dict = dict()
            for k in [50,100,200,500,1000]:
                if verbose:
                    ('\t ' + str(k))
                postprocessor = KNN_postprocessor(k)
                postprocessor.setup(network, test_X)
                pred, conf = postprocessor.postprocess(
                        network, test_X)
                results_dict_ = {}
                if verbose:
                    print(' \n')
                    print('Evaluating OOD dataset')

                results_dict_ = evaluate_OOD(
                    conf.detach().cpu().numpy(), pred.detach().cpu().numpy(), y_true.detach().cpu().numpy(), OOD_label_dataset.iloc[:,0].values, "dataset", results_dict_
                )
                if not dataset_config["scenario"].startswith('s_'):
                    if verbose:
                        print(' \n')
                        print('Evaluating OOD celltypes')

                    if not np.isnan(OOD_label_celltype.iloc[0,0]):
                        results_dict_ = evaluate_OOD(
                            conf.detach().cpu().numpy(), pred.detach().cpu().numpy(), y_true.detach().cpu().numpy(), OOD_label_celltype.iloc[:,0].values, "celltype", results_dict_
                        )
                    else:
                        results_dict_["celltype"] = None
                        if verbose:
                            print("No OOD celltypes, so no celltype analysis")
                results_dict[k] = results_dict_
            
        # For Posterior natworks, the postprocessing is also different    
        elif training_config["OOD_strategy"] == "Posterior":
            postprocessor = postprocessor_dict[training_config["OOD_strategy"]]
            pred, conf = postprocessor.postprocess( 
                network, test_X
            )  # conf is the score of the prediction, scores returns everything

            # Calculate statistics
            results_dict = {}
            if verbose:
                print(' \n')
                print('Evaluating OOD dataset')

            results_dict = evaluate_OOD(
                conf, pred.detach().cpu().numpy(), y_true.detach().cpu().numpy(), OOD_label_dataset.iloc[:,0].values, "dataset", results_dict
            )
            if not dataset_config["scenario"].startswith('s_'):
                if verbose:
                    print(' \n')
                    print('Evaluating OOD celltypes')

                if not np.isnan(OOD_label_celltype.iloc[0,0]):
                    if not isinstance(conf, (np.ndarray, np.matrix)):
                        conf = conf.detach().cpu().numpy()
                    elif not isinstance(pred, np.ndarray):
                        pred = pred.detach().cpu().numpy()
                    elif not isinstance(ytrue, np.ndarray):
                        ytrue = y_true.detach().cpu().numpy()
                    results_dict = evaluate_OOD(
                        conf, pred, y_true, OOD_label_celltype.iloc[:,0].values, "celltype", results_dict
                    )
                else:
                    results_dict["celltype"] = None
                    if verbose:
        
                        print("No OOD celltypes, so no celltype analysis")
        # For EBO and Knn                
        else:
            postprocessor = postprocessor_dict[training_config["OOD_strategy"]]
            pred, conf, scores = postprocessor.postprocess( 
                network, test_X
            )  # conf is the score of the prediction, scores returns everything

            # Calculate statistics
            results_dict = {}
            if verbose:
                print(' \n')
                print('Evaluating OOD dataset')

            results_dict = evaluate_OOD(
                conf.detach().cpu().numpy(), pred.detach().cpu().numpy(), y_true.detach().cpu().numpy(), OOD_label_dataset.iloc[:,0].values, "dataset", results_dict
            )
            if not dataset_config["scenario"].startswith('s_'):
                if verbose:
                    print(' \n')
                    print('Evaluating OOD celltypes')

                if not np.isnan(OOD_label_celltype.iloc[0,0]):
                    results_dict = evaluate_OOD(
                        conf.detach().cpu().numpy(), pred.detach().cpu().numpy(), y_true.detach().cpu().numpy(), OOD_label_celltype.iloc[:,0].values, "celltype", results_dict
                    )
                else:
                    results_dict["celltype"] = None
                    if verbose:
                        print("No OOD celltypes, so no celltype analysis")
        print("\n")
        # save results
        save_dict_to_json(
            results_dict,
            main_config["output_dir"] + main_config["name"] + "/" + main_config["name"],
        )
        
        #trainer.test(model, datamodule=dataLoader, ckpt_path = "best")

    if verbose:
        print("Testing complete")

    if training_config["OOD_strategy"] in ["Ensembles", "Knn"]:
        return (conf.detach().cpu().numpy(), y_true, pred)
    elif training_config["OOD_strategy"] == "Posterior":
        return (conf, y_true, pred)
    else:
        return (
            scores, y_true, pred,
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
        obj = obj.detach().cpu().numpy()

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
    if main_config["verbose"]:
        print("Total training time", end - start)
        print(" \n")

elif args.Run_step == "test":
    start = time.time()
    DataLoader = load_dataset(args.config_file, train = True)
    DataLoader.prepare_data()
    DataLoader.setup(stage="train")
    scores, ytrue, predictions = test_step(
        args.config_file, DataLoader
    )
    end = time.time()
    if main_config["verbose"]:
        print("Total OOD testing time", end - start)
        print(" \n")

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
    model, dataLoader = train_step(args.config_file, train_test_together=True)
    end = time.time()
    if main_config["verbose"]:
        print("Total training time", end - start)
        print(" \n")

    # Testing
    start = time.time()
    scores, ytrue, predictions = test_step(args.config_file, dataLoader)
    end = time.time()
    if main_config["verbose"]:
        print("Total OOD testing time", end - start)
        print(" \n")

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
    if not isinstance(predictions, np.ndarray):
        predictions = predictions.detach().cpu().numpy()
    save_numpy_array(
        predictions,
        main_config["output_dir"]
        + main_config["name"]
        + "/"
        + main_config["name"]
        + "_predictions.csv",
    )

# to save tensorboard results: https://www.tensorflow.org/tensorboard/dataframe_api
