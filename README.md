# OOD_eval

*Note: The documentation of this repo is still under construction*

Repository containing code for the paper 'Evaluation of out-of-distribution detection methods for data shifts in single-cell transcriptomics' to reproduce the results. 

## Structure of the code
The code is constructed in a modular way: with the help of a config file and the main.py script all analyses, for three datasets over 6 methods with varying data splits can be performed. A helper function to construct a config file can be found back in the utils script. The main script takes as arguments the config file and the argument "test"/"train"/"all" to test/train or test and train the model.

## How to reproduce an analysis
1. Decide which dataset, which application setting and which OOD method you want to test.
2. Decide which underlying classification NN architecture you want to use and see how many computational resources are available (NOTE: the COPD dataset is significantly larger and will require more memory to run, esp. the disease application setting)

