# OOD_eval

*Note: The documentation of this repo is still under construction*

Repository containing code for the paper 'Evaluation of out-of-distribution detection methods for data shifts in single-cell transcriptomics' to reproduce the results. 

The code is constructed in a modular way: with the help of a config file and the main.py script all analyses, for three datasets over 6 methods with varying data splits can be performed. A helper function to construct a config file can be found back in the utils script. The main script takes as arguments the config file and the argument "test"/"train"/"all" to test/train or test and train the model.

