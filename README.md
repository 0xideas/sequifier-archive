## Overview
The sequifier package enables:
  - the extraction of sequences for training from a standardised format
  - the configuration and training of a transformer classification model
  - inference on data with a trained model
each of these steps is explained below


## Preprocessing of data into sequences for training

This step presupposes input data with three columns: sequenceId, itemId and timesort.
sequenceId and itemId identify user/item interaction, and the timesort column must
provide values that enable sequential sorting. Often this will simply be a timestamp.

The data can then be processed into training, validation and testing datasets of all
valid subsequences in the original data with the command:

> sequifier.py --preprocess --config_path=[CONFIG PATH] --project_path=[PROJECT PATH]

where the config path specifies the path to the preprocessing config and the project
path the path to the (preferably empty) folder where the output files of the different
steps are written to.

The default config can be found on this path:

> configs/preprocess/default.yaml


## Configuring and training the sequence classification model

The training step is executed with the command:

> sequifier.py --train --config_path=[CONFIG PATH] --project_path=[PROJECT PATH]

and configured using the config. The default config can be found here:

> configs/train/default.yaml


## Inferring on test data using the trained model

Inference is done using the command:

> sequifier.py --infer --config_path=[CONFIG PATH] --project_path=[PROJECT PATH]

and configured using a config file. The default version can be found here:

> configs/infer/default.yaml

