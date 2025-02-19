# OOD_eval

*Note: The documentation of this repo is still under construction*

Repository containing code for the paper 'Evaluation of out-of-distribution detection methods for data shifts in single-cell transcriptomics' to reproduce the results. 

## Structure of the code
The code is constructed in a modular way: with the help of a config file and the main.py script all analyses, for three datasets over 6 methods with varying data splits can be performed. A helper function to construct a config file can be found back in the utils script. The main script takes as arguments the config file and the argument "test"/"train"/"all" to test/train or test and train the model.

## How to reproduce an analysis
1. Decide which dataset, which application setting and which OOD method you want to test.
2. Decide which underlying classification NN architecture you want to use and see how many computational resources are available <br/> (**Note**: the COPD dataset is significantly larger and will require more memory to run, esp. the disease application setting)
3. Go to utils and create a *Config file* to run the analysis with the help of the ```create_config``` function, all properties that need to be decided to run the analyses are input arguments of this function. <br/> (**Note**: to run any analyses a label_conversion_file is needed, this is just a json file (containing a python dictionary) that assigns for any textual label a numerical value (0, ..., K) with K the number of classes)
4. With the help of the ```main.py``` script and the created *Config file* it is now possible to run the analyses. The ```main.py``` script runs in three modes: *train, test* and *all*, indicating if the model is only to be trained, tested or trained and tested at once. The ```main.py``` script takes as arguments the *Config file* and the running mode as a string.
5. The output of the test step is a result dictionary containing for the OOD setting[^1]:
   -  the AUROC
   -  the AUPR_in/AUPR_out,
   -  the FPR@TPR95
   -  the (balanced) accuracy_in, (balanced) accuracy_out.

[^1]: In/out corresponds to the ID or OOD data part (which depends on the OOD setting). The metrics are calculated similarly to those used in a recent benchmark of OOD methods (OpenOOD)

