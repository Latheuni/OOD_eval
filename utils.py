# packages
import yaml


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

def Create_config(config_name,
                  config_dir,
                  write_interval = "epoch",
                  output_dir = "/data/gent/vo/000/gvo00070/vsc43883/Results_OOD/",
                  name = "Unkown_pancreas_nonlinear_logitnorm",
                  verbose = "True",
                  debug = "False",
                  cpus = 1,
                  dataset_name = "Pancreas",
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
                  loss_function = "logitnorm",
                  learning_rate = 1e-04,
                  max_epochs = 150,
                  accelerator = "gpu",
                  devices = 1,
                  ):
    
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
            "loss_function": loss_function,
            "learning_rate": learning_rate,
            "max_epochs": max_epochs,
            "cpus": cpus,
            "accelerator": accelerator,
            "devices": devices}
    }
    with open(config_dir + config_name + '.yml', 'w') as ff:
        yaml.dump(d, ff)
    