## packages
from utils import Create_config

## Function
def MakeConfigs(
    config_dir, config_name, name, train_data, test_data, num_hidden_layers, nodes_per_layer, activations, model
):
    ## Fixed parameters
    dataset = "Pancreas"
    output_dir_results = "/data/gent/vo/000/gvo00070/vsc43883/Results_OOD/LogitNorm/"
    cpus = 1
    data_file = "human_pancreas_norm_complexBatch.h5ad"
    label_conversion_file = "/data/gent/438/vsc43883/Data/Pancreas/Pancreas_conversion.json"
    loss_function = "logitnorm"
    accelerator = "gpu"
    devices = 1

    Create_config(
        config_name,
        config_dir,
        name = name,
        dataset_name=dataset,
        output_dir=output_dir_results,
        cpus=cpus,
        train_techs = train_data,
        OOD_techs = test_data,
        nodes_per_layer = nodes_per_layer, 
        activation = activations,
        label_conversion_file = label_conversion_file, 
        num_hidden_layers = num_hidden_layers,
        loss_function= loss_function,
        accelerator = accelerator,
        devices = devices)


# Variable parameters: name, train_techs, OOD_techs, model, nodes_per_layer, num_hidden_layers, activation

tech_splits = [
    [["inDrop1"], ["inDrop2", "inDrop3", "inDrop4"]],
    [["inDrop2"], ["inDrop1", "inDrop3", "inDrop4"]],
    [["inDrop3"], ["inDrop1", "inDrop2", "inDrop4"]],
    [["inDrop4"], ["inDrop1", "inDrop2", "inDrop3"]],
    [
        ["inDrop1", "inDrop2", "inDrop3", "inDrop4"],
        ["celseq", "celseq2", "fluidigmc1", "smarter", "smartseq2"],
    ],
]

# For the non-linear models
num_layers= [2, 5, 10]
nodes_per_layer = [50, 100, 1000, 5000]
activations = ["elu", "relu", "gelu"]
model = "non-linear"
config_dir = "/user/gent/438/vsc43883/OOD_eval/Configs/Try1_LogitNorm/"
name_dataset = ["inDrop1", "inDrop2","inDrop3","inDrop4","full"]

initial_act = True

for i, dataset_setup in enumerate(tech_splits):
    name = "LogitNorm_Pancreas_" + name_dataset[i] + '_'
    train_data = dataset_setup[1]
    test_data = dataset_setup[0]
    name__ = name + 'non-linear'
    for layers in num_layers:
        for nodes in nodes_per_layer:
            if initial_act:
                for act in activations: 
                    name_ = name__ + '_'+ act + '_l' + str(layers) + '_n' + str(nodes)
                    MakeConfigs(config_dir, "Config_" + name_,  name_, train_data, test_data,layers, nodes, act,'non-linear')
            elif name_dataset[i] == "full" and nodes == 2 and layers == 50:
                for act in activations: 
                    name_ = name__ + act + '_l' + str(layers) + '_n' + str(nodes)
                    MakeConfigs(config_dir, "Config_" + name_, name_, train_data, test_data,layers,nodes, act,'non-linear')
            else:
                name_ = name__ + 'relu' + '_l' + str(layers) + '_n' + str(nodes)
                MakeConfigs(config_dir, "Config_" + name_, name_, train_data, test_data,layers,nodes, "relu",'non-linear')


# For the linear model:
for i, dataset_setup in enumerate(tech_splits):
    name = "LogitNorm_Pancreas_" + name_dataset[i] + '_' + 'linear'
    train_data = dataset_setup[1]
    test_data = dataset_setup[0]

    MakeConfigs(config_dir, "Config_" + name,  name, train_data, test_data,  0, ['/'], '/', 'linear')

