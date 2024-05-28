# packages
import yaml
from Datasets import *

def read_config(yaml_file):
    """very sepcific yaml file

    First level categories are:
    - Main
    - Dataset
    - Network
    - Training
    """
    with open(yaml_file, "r") as file:
        file = yaml.safe_load(file)

    return file["main"], file["dataset"], file["network"], file["training"]

def create_config(config_name,
                  config_dir,
                  write_interval = "epoch",
                  output_dir = "/data/gent/vo/000/gvo00070/vsc43883/Results_OOD/",
                  name = "Unkown_pancreas_nonlinear_logitnorm",
                  verbose = "True",
                  debug = "False",
                  cpus = 1,
                  dataset_name = "Pancreas",
                  scenario = "None"
                  data_dir = "/data/gent/438/vsc43883/Data/Pancreas/",
                  data_file = "human_pancreas_norm_complexBatch.h5ad",
                  label_conversion_file = "/data/gent/438/vsc43883/Data/Pancreas/Pancreas_conversion.json",
                  batch_size = 256,
                  train_techs = ["inDrop1"],
                  OOD_techs = ["celseq2"],
                  validation_size = 0.2,
                  test_size = 0.2,
                  min_celltypes = 10,
                  model = "non-linear",
                  nodes_per_layer = [100,100],
                  num_hidden_layers = 2,
                  activation = "relu",
                  OOD_strategy = "logitnorm",
                  learning_rate = 1e-04,
                  max_epochs = 150,
                  accelerator = "gpu",
                  devices = 1,
                  ):
    if dataset_name == "Pancreas":
        d = {"main": {
                "write_interval": write_interval,
                "output_dir": output_dir,
                "name": name,
                "verbose": verbose,
                "debug": debug},
            "dataset": {
                "name": dataset_name,
                "data_dir": data_dir,
                "data_file": data_file,
                "label_conversion_file": label_conversion_file,
                "batch_size": batch_size,
                "train_techs": train_techs,
                "OOD_techs": OOD_techs,
                "validation_size": validation_size,
                "test_size": test_size,
                "min_celltypes": min_celltypes},
            "network": {
                "model": model,
                "nodes_per_layer": nodes_per_layer,
                "num_hidden_layer": num_hidden_layers,
                "activation": activation},
            "training": {
                "OOD_strategy": OOD_strategy
                "learning_rate": learning_rate,
                "max_epochs": max_epochs,
                "cpus": cpus,
                "accelerator": accelerator,
                "devices": devices}
        }
    else:
        d = {"main": {
                "write_interval": write_interval,
                "output_dir": output_dir,
                "name": name,
                "verbose": verbose,
                "debug": debug},
            "dataset": {
                "name": dataset_name,
                "data_dir": data_dir,
                "data_file": data_file,
                "label_conversion_file": label_conversion_file,
                "batch_size": batch_size,
                "train_techs": train_techs,
                "OOD_techs": OOD_techs,
                "validation_size": validation_size,
                "test_size": test_size,
                "min_celltypes": min_celltypes,
                "scenario": scenario},
            "network": {
                "model": model,
                "nodes_per_layer": nodes_per_layer,
                "num_hidden_layer": num_hidden_layers,
                "activation": activation},
            "training": {
                "OOD_strategy": OOD_strategy
                "learning_rate": learning_rate,
                "max_epochs": max_epochs,
                "cpus": cpus,
                "accelerator": accelerator,
                "devices": devices}
        }
    with open(config_dir + config_name + '.yml', 'w') as ff:
        yaml.dump(d, ff)
    
def load_dataset(dataset_name, config_file):
    """Constructs the lightning datasetmodule and reads in OOD indicator csv files (for novel celltypes or OOD data), based on config files
    """
    main_config, dataset_config, network_config, training_config = read_config(
        config_file
    )

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
        return dataset, OOD_label_dataset, OOD_label_celltype
    elif dataset_config["name"] == "Lung":
        dataset = LitLungDataModule(
            dataset_config["data_dir"],
            dataset_config["data_file"],
            dataset_config["label_conversion_file"],
            dataset_config["scenario"],
            dataset_config["batch_size"],
            dataset_config["val_size"],
            dataset_config["test_size"],
            training_config["cpus"],
            main_config["name"],
            verbose = main_config["verbose"],
        )
        OOD_label_celltype = pd.read_csv(
            dataset_config["data_dir"]
            + "OOD_ind_lung"
            + "_celltypes_"
            + main_config["name"]
            + ".csv",
            index_col=0,
        )
        return dataset, OOD_label_celltype

    elif dataset_config["name"] == "Immune":
        dataset = LitImmuneDataModule(
            dataset_config["data_dir"],
            dataset_config["data_file"],
            dataset_config["label_conversion_file"],
            dataset_config["scenario"],
            dataset_config["batch_size"],
            dataset_config["val_size"],
            dataset_config["test_size"],
            training_config["cpus"],
            main_config["name"],
            verbose = main_config["verbose"],
        )
        OOD_label_celltype = pd.read_csv(
            dataset_config["data_dir"]
            + "OOD_ind_immune"
            + "_celltypes_"
            + main_config["name"]
            + ".csv",
            index_col=0,
        )
        return dataset, OOD_label_celltype
    
    def load_network(config_file):
        main_config, dataset_config, network_config, training_config = read_config(
        config_file)

        if network_config["model"] == "linear":
        network = LinearNetwork(
            n_features,
            n_classes,
            network_config["nodes_per_layer"],
            network_config["num_hidden_layer"],
        )
        else:
            if training_config['OOD strategy'] == "dropout": # loss should be cross-entropy
                network = DropoutNetwork(n_features, n_classes, network_config["nodes_per_layer"],
                network_config["num_hidden_layer"],
                network_config["activation"],
                0.5 )
            else:
                network = NonLinearNetwork(
                    n_features,
                    n_classes,
                    network_config["nodes_per_layer"],
                    network_config["num_hidden_layer"],
                    network_config["activation"],
                )
        return(network)
    
    def evaluate_OOD(conf, pred, ytrue, OOD_ind, OOD_scenario, results_dict, verbose = True):
        """Calculates all OOD metrics for OOD_ind labels, OOD_scenario has to be "dataset"
        or "celltypes"
        """
        auroc_dataset, aupr_in_dataset, aupr_out_dataset, fpr_dataset = auc_and_fpr_recall(
        conf, OOD_label_dataset.iloc[:, 0], 0.95)
        acc_OOD, acc_ID, bacc_OOD, bacc_ID = general_metrics(
            conf.cpu().numpy(),
            OOD_ind.iloc[:, 0].values,
            pred,
            ytrue,
            verbose,
        )

        results_dict[OOD_scenario] = {
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
        print("For the " + str(OOD_scenario))
        print("auroc", auroc_dataset)
        print("aupr_in", aupr_in_dataset)
        print("aupr_out", aupr_out_dataset)
        print("fpr", fpr_dataset)

        return(results_dict)