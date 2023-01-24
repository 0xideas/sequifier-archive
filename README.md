<img src="./design/logo.png">


## Overview
The sequifier package enables:
  - the extraction of sequences for training from a standardised format
  - the configuration and training of a transformer classification model
  - inference on data with a trained model

Each of these steps is explained below.


## Preprocessing of data into sequences for training

The preprocessing step is specifically designed for scenarios where for long series
of events, the prediction of the next event from the previous N events  is of interest.
In cases of sequences where only the last item is a valid target, the  preprocessing
step does not apply.

This step presupposes input data with three columns: sequenceId, itemId and timesort.
sequenceId and itemId identify sequence and item, and the timesort column must
provide values that enable sequential sorting. Often this will simply be a timestamp.

The data can then be processed into training, validation and testing datasets of all
valid subsequences in the original data with the command:

> sequifier --preprocess --config_path=[CONFIG PATH] --project_path=[PROJECT PATH]

The config path specifies the path to the preprocessing config and the project
path the path to the (preferably empty) folder the output files of the different
steps are written to.

The default config can be found on this path:

> configs/preprocess/default.yaml


## Configuring and training the sequence classification model

The training step is executed with the command:

> sequifier --train --config_path=[CONFIG PATH] --project_path=[PROJECT PATH]

If the data on which the model is trained comes from the preprocessing step, the flag

> --on-preprocessed

should also be added.

If the training data does not come from the preprocessing step, both train and validation
data have to take the form of a csv file with the columns:

> sequenceId, seq_length, seq_length-1,...,1, target

The training step is configured using the config. The two default configs can be found here:

> configs/train/default.yaml

> configs/train/default-on-preprocessed.yaml


## Inferring on test data using the trained model

Inference is done using the command:

> sequifier --infer --config_path=[CONFIG PATH] --project_path=[PROJECT PATH]

and configured using a config file. The default version can be found here:

> configs/infer/default.yaml


## Complete example how to use the repository

1. create a new project folder at PROJECT PATH and a "configs" subfolder
2. copy default configs from repository for preprocessing, training and inference and name them preprocess.yaml, train.yaml and infer.yaml
3. adapt preprocess config to take the path to the data you want to preprocess
4. run 
> sequifier --preprocess --config_path=[PROJECT PATH]/configs/preprocess.yaml --project_path=[PROJECT PATH]
5. adapt dd_config parameter in train.yaml and infer.yaml in to dd_config path from preprocessing
6. run 
> sequifier --train --on-preprocessed --config_path=[PROJECT PATH]/configs/train.yaml --project_path=[PROJECT PATH]
7. adapt inference_data_path in infer.yaml
8. run  
> sequifier --infer --config_path=PROJECT PATH]/configs/infer.yaml --project_path=[PROJECT PATH]
9. find your predictions at [PROJECT PATH]/outputs/predictions/sequifier-default-best_predictions.csv
